import argparse
import torch
import data.dataset as dataset
from models.cnn_baseline import create_cnn_baseline
from models.resnet_finetune import create_resnet_finetune
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a pneumonia detection model using chest X-ray images")

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn_baseline", "resnet_baseline", "resnet_finetune"],
        required=True,
        help="Model type to test (cnn_baseline, resnet_baseline, resnet_finetune)"
    )

    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray", help="Path to the dataset (Please download as described in README.md)")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Path to save evaluation plots")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train, dataset_test = dataset.create_datasets(args.data_dir)
    dataloader_train, dataloader_test = dataset.create_dataloaders(dataset_train, dataset_test, args.batch_size, args.num_workers)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    if args.model == "cnn_baseline":
        model = create_cnn_baseline()
    elif args.model == "resnet_baseline":
        model = create_resnet_finetune(weights=None)
    else:
        model = create_resnet_finetune(weights=None)

    model.to(device)
    model.load_state_dict(checkpoint["model"])

    y_preds = []
    y_true = []

    with torch.no_grad():
        model.eval()
        for X, y in dataloader_test:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_preds.append(y_pred.argmax(dim=1))
            y_true.append(y)
    
    y_preds = torch.cat(y_preds).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()

    acc = (y_preds == y_true).mean()
    precision = precision_score(y_true, y_preds, average="macro")
    recall = recall_score(y_true, y_preds, average="macro")
    f1 = f1_score(y_true, y_preds, average="macro")
    print(f"Model: {args.model}")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    cm = confusion_matrix(y_true, y_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"]).plot(cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{args.plot_dir}/confusion_matrix_{args.model}.png")
    plt.close()