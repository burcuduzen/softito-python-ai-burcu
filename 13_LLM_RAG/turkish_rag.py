"""Çok dilli embedding ile kaynak gösteren Türkçe semantik arama."""
from dataclasses import dataclass

@dataclass
class Document:
    source: str
    text: str

def semantic_search(query: str, documents: list[Document], k: int = 3):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    document_vectors = model.encode(
        [document.text for document in documents], normalize_embeddings=True
    )
    query_vector = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_vector, document_vectors)[0]
    indices = scores.argsort()[::-1][:k]
    return [(documents[index], float(scores[index])) for index in indices]

if __name__ == "__main__":
    documents = [
        Document("izin.md", "Çalışanların yıllık izin hakları kıdeme göre belirlenir."),
        Document("uzaktan.md", "Uzaktan çalışma günleri ekip yöneticisiyle planlanır."),
        Document("egitim.md", "Teknik eğitim bütçesi yılda bir kez kullanılabilir."),
    ]
    for document, score in semantic_search("Eğitim desteği nasıl alınır?", documents):
        print(round(score, 3), document.source, document.text)
