# Churn Prediction API

FastAPI ile doğrulanan giriş, health endpoint, tahmin endpoint'i ve Docker imajı.

## Çalıştırma

```bash
cd 14_MLOps_Docker
docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

Ana veri kaynağı ve indirme bilgisi için kök dizindeki `datasets.yaml` dosyasına bakın. Büyük veri bulunmadığında uygun bölümler tekrarlanabilir demo verisi üretir.
