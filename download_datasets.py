"""datasets.yaml içindeki Kaggle veri setlerini kagglehub ile indirir."""
from __future__ import annotations
import argparse
from pathlib import Path
import kagglehub
import yaml

CATALOG = Path(__file__).with_name("datasets.yaml")

def load_catalog() -> dict:
    with CATALOG.open(encoding="utf-8") as file:
        return yaml.safe_load(file)

def download(name: str) -> None:
    item = load_catalog()[name]
    path = kagglehub.dataset_download(item["kaggle"])
    print(f"{name} indirildi: {path}")
    if item.get("file"):
        print(f"Beklenen ana dosya: {item['file']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle veri seti indir")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    catalog = load_catalog()
    if args.list:
        print("\n".join(f"{key}: {value['kaggle']}" for key, value in catalog.items()))
    elif args.name in catalog:
        download(args.name)
    else:
        parser.error("Geçerli bir veri seti adı girin veya --list kullanın.")
