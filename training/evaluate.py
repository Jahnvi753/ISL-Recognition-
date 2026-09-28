import torch
from torch.utils.data import DataLoader

from training.dataset import ISLDataset
from models.stgcn.model import STGCN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = ISLDataset(
        root="data/processed",
        num_frames=100
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    checkpoint = torch.load(
        "models/stgcn/best_model.pt",
        map_location=device
    )

    model = STGCN(
        num_classes=len(dataset.classes)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    correct = 0
    total = 0

    class_correct = [0] * len(dataset.classes)
    class_total = [0] * len(dataset.classes)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

            for true, pred in zip(y, predictions):
                class_total[true.item()] += 1

                if true.item() == pred.item():
                    class_correct[true.item()] += 1

    print("\nOverall accuracy:")
    print(f"{correct}/{total} = {correct / total:.4f}")

    print("\nPer-class accuracy:")

    for i, class_name in enumerate(dataset.classes):
        accuracy = (
            class_correct[i] / class_total[i]
            if class_total[i] > 0
            else 0
        )

        print(
            f"{class_name}: "
            f"{class_correct[i]}/{class_total[i]} "
            f"= {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()