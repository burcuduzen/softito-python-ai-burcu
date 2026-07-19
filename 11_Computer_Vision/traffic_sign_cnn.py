"""GTSRB trafik levhaları için PyTorch görüntü sınıflandırma projesi."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms(image_size=48):
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(.08, .08), scale=(.9, 1.1)),
        transforms.ColorJitter(brightness=.20, contrast=.20, saturation=.15),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
    ])
    return train_transform, evaluation_transform

def create_loaders(data_dir: Path, batch_size=64, validation_ratio=.15, workers=2):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    train_transform, evaluation_transform = build_transforms()
    base = datasets.ImageFolder(data_dir)
    count = len(base)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(count, generator=generator).tolist()
    split = int(count * (1 - validation_ratio))
    train_indices, validation_indices = indices[:split], indices[split:]

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(data_dir, transform=evaluation_transform)
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    common = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_subset, shuffle=True, **common)
    validation_loader = DataLoader(validation_subset, shuffle=False, **common)
    return train_loader, validation_loader, base.classes

def build_model(class_count: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((3, 3)),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(.40),
        nn.Linear(256, class_count),
    )

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    import torch
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * len(images)
        correct += (predictions == labels).sum().item()
        total += len(images)
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": all_predictions,
        "labels": all_labels,
    }

def save_curves(history, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=170)
    plt.close(fig)

def train(data_dir: Path, output: Path, epochs=15, batch_size=64, workers=2):
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, classes = create_loaders(
        data_dir, batch_size=batch_size, workers=workers
    )
    model = build_model(len(classes)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=.5
    )
    history = {key: [] for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]}
    best_accuracy = -1.0
    patience_counter = 0
    output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_result = run_epoch(model, validation_loader, loss_fn, device)
        scheduler.step(validation_result["accuracy"])
        for key, value in [
            ("train_loss", train_result["loss"]),
            ("val_loss", validation_result["loss"]),
            ("train_accuracy", train_result["accuracy"]),
            ("val_accuracy", validation_result["accuracy"]),
        ]:
            history[key].append(value)
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_result['loss']:.4f} "
            f"val_loss={validation_result['loss']:.4f} "
            f"val_acc={validation_result['accuracy']:.4f}"
        )
        if validation_result["accuracy"] > best_accuracy:
            best_accuracy = validation_result["accuracy"]
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "image_size": 48,
            }, output / "best_traffic_sign_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping.")
                break

    checkpoint = torch.load(output / "best_traffic_sign_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final = run_epoch(model, validation_loader, loss_fn, device)
    report = classification_report(
        final["labels"], final["predictions"], output_dict=True, zero_division=0
    )
    metrics = {
        "device": str(device),
        "class_count": len(classes),
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "best_validation_accuracy": best_accuracy,
        "epochs_trained": len(history["train_loss"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(final["labels"], final["predictions"]).tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_curves(history, output)

def main():
    parser = argparse.ArgumentParser(description="GTSRB trafik levhası CNN")
    parser.add_argument("data_dir", type=Path, help="Sınıf klasörlerini içeren train dizini")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch_size, args.workers)

if __name__ == "__main__":
    main()
