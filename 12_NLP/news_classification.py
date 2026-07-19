"""Türkçe haberlerde temizleme, EDA, TF-IDF ve kategori sınıflandırması."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
TURKISH_STOPWORDS = {
    "acaba", "ama", "artık", "aslında", "az", "bazı", "belki", "ben", "bile",
    "bir", "birçok", "biz", "bu", "böyle", "da", "daha", "de", "defa", "diye",
    "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "için", "ile", "ise",
    "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", "ne", "neden", "nerde",
    "nerede", "nereye", "niçin", "niye", "o", "sanki", "siz", "şu", "tüm",
    "ve", "veya", "ya", "yani", "çok", "çünkü",
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zçğıöşü0-9\s]", " ", text)
    text = re.sub(r"\d+", " sayi ", text)
    tokens = [
        token for token in text.split()
        if token not in TURKISH_STOPWORDS and len(token) > 1
    ]
    return " ".join(tokens)

def create_demo_corpus(repeats=35) -> pd.DataFrame:
    samples = {
        "ekonomi": [
            "Merkez Bankası politika faizini açıkladı",
            "İhracat geçen yılın aynı ayına göre yükseldi",
            "Borsa İstanbul günü primli tamamladı",
            "Şirket yeni fabrika yatırımı yapacağını duyurdu",
            "Enflasyon verileri piyasa beklentisinin altında kaldı",
        ],
        "teknoloji": [
            "Yeni yapay zeka modeli geliştiricilere açıldı",
            "Yazılım güncellemesi güvenlik açıklarını kapatıyor",
            "Yerli çip üretimi için araştırma merkezi kurulacak",
            "Bulut bilişim yatırımları hızla büyüyor",
            "Akıllı telefon üreticisi yeni cihazını tanıttı",
        ],
        "spor": [
            "Takım final karşılaşmasını uzatmalarda kazandı",
            "Transfer görüşmeleri resmi imzayla tamamlandı",
            "Milli sporcu dünya şampiyonasında altın madalya aldı",
            "Teknik direktör maç öncesi açıklama yaptı",
            "Lig fikstürünün yeni sezon tarihleri belli oldu",
        ],
        "kültür": [
            "Uluslararası film festivali bu hafta başlıyor",
            "Modern sanat sergisi ziyaretçilere açıldı",
            "Roman yılın edebiyat ödülünü kazandı",
            "Tiyatro oyunu yeni sezonda yeniden sahnelenecek",
            "Arkeoloji müzesindeki eserler restore edildi",
        ],
    }
    rows = []
    for category, sentences in samples.items():
        for index in range(repeats):
            sentence = sentences[index % len(sentences)]
            rows.append({"text": f"{sentence}. Ayrıntılar haber merkezinden aktarıldı {index}.", "category": category})
    return pd.DataFrame(rows)

def load_data(path: Path | None, text_column="text", target_column="category") -> pd.DataFrame:
    df = pd.read_csv(path) if path else create_demo_corpus()
    if not {text_column, target_column}.issubset(df.columns):
        raise ValueError(f"CSV içinde {text_column} ve {target_column} sütunları bulunmalı.")
    clean = df[[text_column, target_column]].dropna().drop_duplicates().copy()
    clean.columns = ["text", "category"]
    clean["clean_text"] = clean["text"].map(clean_text)
    clean = clean[clean["clean_text"].str.len() > 0]
    return clean

def corpus_report(df: pd.DataFrame) -> dict:
    token_counts = df["clean_text"].str.split().map(len)
    vocabulary = Counter(token for text in df["clean_text"] for token in text.split())
    return {
        "document_count": len(df),
        "category_distribution": df["category"].value_counts().to_dict(),
        "average_token_count": float(token_counts.mean()),
        "median_token_count": float(token_counts.median()),
        "vocabulary_size": len(vocabulary),
        "most_common_terms": vocabulary.most_common(20),
    }

def build_candidates():
    common_vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=.98,
        sublinear_tf=True,
        max_features=30000,
    )
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", common_vectorizer),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text, ngram_range=(1, 2),
                min_df=2, sublinear_tf=True, max_features=30000
            )),
            ("model", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

def evaluate_candidates(X_train, X_test, y_train, y_test):
    rows, fitted = [], {}
    for name, model in build_candidates().items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        rows.append({
            "model": name,
            "macro_f1": f1_score(y_test, prediction, average="macro"),
            "weighted_f1": f1_score(y_test, prediction, average="weighted"),
        })
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return comparison, fitted

def top_features(model, class_names, top_n=15) -> dict:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]
    if not hasattr(classifier, "coef_"):
        return {}
    names = np.asarray(vectorizer.get_feature_names_out())
    result = {}
    for index, class_name in enumerate(classifier.classes_):
        coefficients = classifier.coef_[index]
        result[str(class_name)] = names[np.argsort(coefficients)[-top_n:][::-1]].tolist()
    return result

def save_confusion(y_true, prediction, output: Path):
    labels = sorted(set(y_true))
    matrix = confusion_matrix(y_true, prediction, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Tahmin", ylabel="Gerçek", title="Haber Kategorisi Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=170)
    plt.close(fig)

def run(input_path: Path | None, output: Path) -> None:
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=.25,
        stratify=df["category"], random_state=RANDOM_STATE
    )
    comparison, fitted = evaluate_candidates(X_train, X_test, y_train, y_test)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    prediction = best_model.predict(X_test)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    report = {
        "corpus": corpus_report(df),
        "best_model": best_name,
        "metrics": classification_report(y_test, prediction, output_dict=True),
        "top_features": top_features(best_model, sorted(df["category"].unique())),
    }
    (output / "nlp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"text": X_test, "actual": y_test, "prediction": prediction}).to_csv(output / "predictions.csv", index=False)
    joblib.dump(best_model, output / "news_classifier.joblib")
    save_confusion(y_test, prediction, output)
    print(comparison.round(4).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Türkçe haber sınıflandırma")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
