# Spark Uçuş Gecikme Analizi

Uçuş CSV verisini Spark ile okuyup havayolu bazında gecikme ve iptal oranı üretir.

## Çalıştırma

```bash
cd 15_Big_Data
python flight_delay_spark.py data/flights.csv
```

Ana veri kaynağı ve indirme bilgisi için kök dizindeki `datasets.yaml` dosyasına bakın. Büyük veri bulunmadığında uygun bölümler tekrarlanabilir demo verisi üretir.
