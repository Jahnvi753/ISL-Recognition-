import torch
from torch.utils.data import DataLoader, random_split

from training.dataset import ISLDataset
from models.stgcn.model import STGCN


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    batch_size = 8
    epochs = 30
    learning_rate = 0.001

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = ISLDataset(
        root="data/processed",
        num_frames=100
    )

    print("Classes:", dataset.classes)
    print("Total samples:", len(dataset))

    # 80% training, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = STGCN(
        num_classes=len(dataset.classes)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -----------------------------
    # Training
    # -----------------------------
    best_val_accuracy = 0.0

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(outputs, y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == y
            ).sum().item()

            train_total += y.size(0)

        train_accuracy = (
            train_correct / train_total
        )

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == y
                ).sum().item()

                val_total += y.size(0)

        val_accuracy = (
            val_correct / val_total
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss: {train_loss / len(train_loader):.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Acc: {val_accuracy:.4f}"
        )

        # -----------------------------
        # Save best model
        # -----------------------------
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": dataset.classes,
                },
                "models/stgcn/best_model.pt"
            )

            print("Saved best model.")

    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
