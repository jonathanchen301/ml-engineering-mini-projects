import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.cnn_baseline import create_cnn_baseline
from models.resnet_finetune import create_resnet_finetune

inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for pneumonia detection model using chest X-ray images")

    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn_baseline", "resnet_baseline", "resnet_finetune"],
        required=True,
        help="Model type (cnn_baseline, resnet_baseline, resnet_finetune)"
    )
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the X-ray image to classify")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == "cnn_baseline":
        model = create_cnn_baseline()
    elif args.model == "resnet_baseline":
        model = create_resnet_finetune(weights=None)
    else:
        model = create_resnet_finetune(weights=None)
    
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = inference_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    class_names = ["NORMAL", "PNEUMONIA"]
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Image: {args.image_path}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass Probabilities:")
    print(f"  NORMAL:    {probabilities[0][0].item():.4f}")
    print(f"  PNEUMONIA: {probabilities[0][1].item():.4f}")
    print("="*50)