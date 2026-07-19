"""TF-IDF ve Linear SVM ile haber kategorisi tahmini."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def create_demo_corpus():
    samples = {
        "ekonomi": [
            "Merkez bankası faiz kararını açıkladı",
            "İhracat geçen aya göre arttı",
            "Borsa günü yükselişle kapattı",
        ],
        "teknoloji": [
            "Yeni yapay zeka modeli tanıtıldı",
            "Yazılım güncellemesi yayınlandı",
            "Çip üretiminde yeni yatırım başladı",
        ],
        "spor": [
            "Takım final maçını kazandı",
            "Transfer görüşmeleri tamamlandı",
            "Milli sporcu altın madalya aldı",
        ],
        "kültür": [
            "Film festivali bu hafta başlıyor",
            "Yeni sergi sanatseverlerle buluştu",
            "Roman yılın ödülünü kazandı",
        ],
    }
    texts, labels = [], []
    for label, sentences in samples.items():
        for index in range(12):
            texts.append(sentences[index % 3] + f" Haber gelişmesi {index}")
            labels.append(label)
    return texts, labels

if __name__ == "__main__":
    texts, labels = create_demo_corpus()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=.25, stratify=labels, random_state=42
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LinearSVC()),
    ])
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), digits=4))
