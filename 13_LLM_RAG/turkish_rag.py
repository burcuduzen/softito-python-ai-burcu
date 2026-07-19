"""Türkçe belgeler için chunking, embedding, retrieval ve kaynaklı RAG sistemi."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

@dataclass
class Document:
    source: str
    text: str
    metadata: dict | None = None

@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    start_word: int
    end_word: int
    metadata: dict

@dataclass
class SearchResult:
    chunk_id: str
    source: str
    text: str
    score: float
    metadata: dict

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text

def split_into_chunks(document: Document, chunk_size=90, overlap=20) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("Overlap, chunk_size değerinden küçük olmalıdır.")
    words = normalize_text(document.text).split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        selected = words[start:start + chunk_size]
        if not selected:
            break
        text = " ".join(selected)
        digest = hashlib.sha1(f"{document.source}:{start}:{text}".encode()).hexdigest()[:12]
        chunks.append(Chunk(
            chunk_id=digest,
            source=document.source,
            text=text,
            start_word=start,
            end_word=start + len(selected),
            metadata=document.metadata or {},
        ))
        if start + chunk_size >= len(words):
            break
    return chunks

def load_documents(directory: Path | None) -> list[Document]:
    if directory is None:
        return [
            Document("izin_politikasi.md", "Yıllık izin hakkı çalışanların kıdem süresine göre hesaplanır. İzin talebi çalışan portalı üzerinden yönetici onayına gönderilir. Beş güne kadar olan talepler en az üç iş günü önce oluşturulmalıdır.", {"department": "İK"}),
            Document("uzaktan_calisma.md", "Uzaktan çalışma günleri ekip yöneticisi ile aylık olarak planlanır. Bilgi güvenliği kuralları evden çalışırken de geçerlidir. Şirket cihazı ve kurumsal VPN kullanılması zorunludur.", {"department": "İK"}),
            Document("egitim_butcesi.md", "Her çalışan teknik eğitim ve sertifika için yıllık eğitim bütçesinden yararlanabilir. Talep formunda eğitimin işle ilişkisi açıklanmalı ve yönetici onayı alınmalıdır.", {"department": "Eğitim"}),
            Document("bilgi_guvenligi.md", "Şüpheli e-postalar açılmadan güvenlik ekibine bildirilmelidir. Parolalar başka kişilerle paylaşılmamalı, çok faktörlü kimlik doğrulama etkin tutulmalıdır.", {"department": "BT"}),
        ]
    documents = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            documents.append(Document(str(path.relative_to(directory)), path.read_text(encoding="utf-8"), {"path": str(path)}))
    if not documents:
        raise ValueError("Dizinde .txt veya .md belge bulunamadı.")
    return documents

class EmbeddingBackend:
    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))

class TfidfBackend(EmbeddingBackend):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).toarray()
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norm, 1e-12, None)

class VectorStore:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("İndekslenecek chunk bulunamadı.")
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self.backend.fit(texts)
        self.vectors = self.backend.encode(texts)

    def search(self, query: str, k=3, minimum_score=0.0) -> list[SearchResult]:
        if self.vectors is None:
            raise RuntimeError("Önce index() çağrılmalıdır.")
        query_vector = self.backend.encode([query])[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(scores)[::-1]
        results = []
        for index in indices:
            if scores[index] < minimum_score:
                continue
            chunk = self.chunks[index]
            results.append(SearchResult(
                chunk.chunk_id, chunk.source, chunk.text,
                float(scores[index]), chunk.metadata,
            ))
            if len(results) == k:
                break
        return results

def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[Kaynak {index}: {result.source}]\n{result.text}"
        for index, result in enumerate(results, start=1)
    )
    return f"""Sen kaynaklara bağlı kalan Türkçe bir asistansın.
Yalnızca aşağıdaki bağlamı kullan. Cevap bağlamda yoksa 'Belgelerde bu bilgi bulunamadı.' de.
Cevabın sonunda kullandığın kaynakları belirt.

BAĞLAM:
{context}

SORU:
{question}

CEVAP:"""

def extractive_answer(question: str, results: list[SearchResult]) -> dict:
    if not results or results[0].score <= 0:
        return {"answer": "Belgelerde bu bilgi bulunamadı.", "sources": []}
    best_sentences = []
    query_terms = set(question.lower().split())
    for result in results:
        sentences = re.split(r"(?<=[.!?])\s+", result.text)
        ranked = sorted(sentences, key=lambda s: len(query_terms & set(s.lower().split())), reverse=True)
        if ranked and ranked[0]:
            best_sentences.append(ranked[0])
    return {
        "answer": " ".join(best_sentences),
        "sources": sorted({result.source for result in results}),
    }

def create_backend(use_tfidf=False):
    if use_tfidf:
        return TfidfBackend()
    try:
        return SentenceTransformerBackend()
    except Exception as exc:
        print(f"Embedding modeli yüklenemedi, TF-IDF kullanılacak: {exc}")
        return TfidfBackend()

def run(question: str, document_dir: Path | None, output: Path, use_tfidf=False):
    documents = load_documents(document_dir)
    chunks = [chunk for document in documents for chunk in split_into_chunks(document)]
    store = VectorStore(create_backend(use_tfidf))
    store.index(chunks)
    results = store.search(question, k=3)
    answer = extractive_answer(question, results)
    payload = {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [asdict(result) for result in results],
        "prompt": build_prompt(question, results),
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "rag_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Türkçe RAG uygulaması")
    parser.add_argument("question", nargs="?", default="Teknik eğitim desteği nasıl alınır?")
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--tfidf", action="store_true", help="Embedding yerine hızlı TF-IDF kullan")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.question, args.documents, args.output, args.tfidf)

if __name__ == "__main__":
    main()
