"""GTSRB klasör düzenine uygun PyTorch CNN eğitim uygulaması."""
def train(data_dir: str = "data/train", epochs: int = 3) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(brightness=.15, contrast=.15),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(.25), nn.Linear(64 * 8 * 8, len(dataset.classes)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.4f}")
    torch.save(model.state_dict(), "traffic_sign_cnn.pt")

if __name__ == "__main__":
    train()
