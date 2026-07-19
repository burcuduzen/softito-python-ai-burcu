# Softito Python ve Yapay Zekâ Çalışmaları

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Projects](https://img.shields.io/badge/proje-15-2EA44F)
![Status](https://img.shields.io/badge/status-geliştiriliyor-F59E0B)

Softito Yapay Zekâ Yazılımcılığı eğitimi boyunca işlenen konuların, birbirinden farklı veri setleri ve gerçek iş problemleri üzerinde uygulandığı çalıştırılabilir Python projeleri.

Projeler; eğitimde işlenen konuları pekiştirmek, farklı veri setleri üzerinde uygulama deneyimi kazanmak ve uçtan uca bir yapay zekâ portföyü oluşturmak amacıyla hazırlanmıştır.

## İçerik

| No | Bölüm | Uygulama | Veri seti |
|---:|---|---|---|
| 01 | Python Temelleri | Öğrenci başarı analizi | Students Performance |
| 02 | İleri Python | E-ticaret sipariş yönetimi | Brazilian E-Commerce |
| 03 | EDA | Otel rezervasyon analizi | Hotel Booking Demand |
| 04 | Linear Regression | Ev fiyat tahmini | California Housing |
| 05 | Logistic Regression | Müşteri kaybı tahmini | Telco Customer Churn |
| 06 | Klasik ML | Hastalık tahmininde model karşılaştırması | Diagnostic Measurements |
| 07 | Denetimsiz Öğrenme | Müşteri segmentasyonu | Mall Customers |
| 08 | Anomali Tespiti | Kredi kartı dolandırıcılığı | Credit Card Fraud |
| 09 | Model İzleme | Veri dağılımı ve drift kontrolü | Weather AUS |
| 10 | Deep Learning | Moda ürünü sınıflandırma | Fashion-MNIST |
| 11 | Computer Vision | Trafik levhası tanıma | GTSRB |
| 12 | NLP | Haber kategorisi tahmini | AG News / Türkçe örnekler |
| 13 | LLM ve RAG | Türkçe doküman arama | Özgün bilgi tabanı |
| 14 | MLOps ve Docker | Tahmin API'si | Telco Churn |
| 15 | Big Data | Uçuş gecikme analizi | US Flight Delays |

## Repo yapısı

```text
softito-python-ai-burcu/
├── 01_Python_Temelleri/
├── 02_Ileri_Python/
├── 03_EDA/
├── 04_Linear_Regression/
├── 05_Logistic_Regression/
├── 06_Klasik_ML/
├── 07_Denetimsiz_Ogrenme/
├── 08_Anomali_Tespiti/
├── 09_Model_Izleme/
├── 10_Deep_Learning/
├── 11_Computer_Vision/
├── 12_NLP/
├── 13_LLM_RAG/
├── 14_MLOps_Docker/
├── 15_Big_Data/
├── datasets.yaml
├── download_datasets.py
└── requirements.txt
```

Her bölümde açıklayıcı bir `README.md` ve çalıştırılabilir Python dosyası bulunur. Küçük projeler örnek veriyle doğrudan çalışır. Büyük veri setleri için Kaggle kaynağı `datasets.yaml` dosyasında tanımlıdır.

## Kurulum

```bash
git clone https://github.com/burcuduzen/softito-python-ai-burcu.git
cd softito-python-ai-burcu
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kaggle verilerini indirme

```bash
python download_datasets.py --list
python download_datasets.py telco_churn
```

`kagglehub` ilk kullanımda Kaggle kimlik doğrulaması isteyebilir. Küçük demo verisi içeren projeler Kaggle indirmesi olmadan da çalışır.

## Çalıştırma örnekleri

```bash
python 01_Python_Temelleri/student_analysis.py
python 03_EDA/hotel_booking_eda.py
python 04_Linear_Regression/california_housing.py
python 06_Klasik_ML/model_comparison.py
python 12_NLP/news_classification.py
uvicorn 14_MLOps_Docker.app:app --reload
```

## Proje standardı

- Eksik değer ve veri tipi kontrolü
- Sabit `random_state` ile tekrarlanabilir deneyler
- Train/test ayrımı ve veri sızıntısını engelleyen Pipeline
- Probleme uygun metrikler: F1, ROC-AUC, MAE, RMSE ve R²
- Sonuçların `outputs/` klasörüne yazılması
- Türkçe açıklamalar ve okunabilir fonksiyon yapısı

## Hazırlayan

**Burcu Düzen**  
Bilgisayar Mühendisliği — Yapay Zekâ ve Veri Bilimi
