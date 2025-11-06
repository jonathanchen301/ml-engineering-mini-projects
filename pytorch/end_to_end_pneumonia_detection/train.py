import argparse
import torch
import data.dataset as dataset
from models.cnn_baseline import create_cnn_baseline
from models.resnet_finetune import create_resnet_finetune
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a pneumonia detection model using chest X-ray images")

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn_baseline", "resnet_baseline", "resnet_finetune"],
        required=True,
        help="Model type to train (cnn_baseline, resnet_baseline, resnet_finetune)"
    )

    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray", help="Path to the dataset (Please download as described in README.md)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Path to save checkpoints and final models")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Path to save training plots")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train, dataset_val = dataset.create_datasets(args.data_dir)
    dataloader_train, dataloader_val = dataset.create_dataloaders(dataset_train, dataset_val, args.batch_size, args.num_workers)

    if args.model == "cnn_baseline":
        model = create_cnn_baseline()
    elif args.model == "resnet_baseline":
        model = create_resnet_finetune(weights=None)
    else:
        model = create_resnet_finetune(weights="IMAGENET1K_V1")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.1)

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float('inf')

    for epoch in range(args.epochs):

        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        model.train()
        for X, y in dataloader_train:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader_train)

        model.eval()
        for X, y in dataloader_val:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                val_acc += (y_pred.argmax(dim=1) == y).sum().item()

        val_loss /= len(dataloader_val)
        val_acc /= len(dataset_val)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            torch.save({
                "model": model.state_dict(),
                "metrics": metrics,
            }, f"{args.output_dir}/best_model_{args.model}.pth")
            best_val_loss = val_loss

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

    torch.save({
        "model": model.state_dict(),
        "metrics": metrics,
    }, f"{args.output_dir}/final_model_{args.model}.pth")

    print(f"Best Val Loss: {best_val_loss:.4f}")

# Plotting the training and validation curves
plt.figure(figsize=(10, 5))

plt.plot(metrics["train_loss"], label="Training Loss")
plt.plot(metrics["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Loss Curves - {args.model}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{args.plot_dir}/loss_curves_{args.model}.png")
plt.close()

# Plotting the accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(metrics["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Accuracy Curves - {args.model}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.savefig(f"{args.plot_dir}/accuracy_curves_{args.model}.png")
plt.close()