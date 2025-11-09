# End to End Pneumonia Detection

A complete deep learning pipeline for pneumonia detection from chest X-rays using PyTorch.

## Dataset

Download ChestXRay2017.zip from https://data.mendeley.com/datasets/rscbjbr9sj/2

Unzip and move to `data/chest_xray/`

## Usage

**Training:**
```bash
python train.py --model cnn_baseline --epochs 10 --batch_size 32 --lr 0.0001
python train.py --model resnet_finetune --epochs 10 --batch_size 32 --lr 0.0001
```

**Evaluation:**
```bash
python eval.py --model cnn_baseline --checkpoint_path checkpoints/best_model_cnn_baseline.pth
python eval.py --model resnet_finetune --checkpoint_path checkpoints/best_model_resnet_finetune.pth
```

**Inference:**
```bash
python inference.py --model cnn_baseline --checkpoint_path checkpoints/best_model_cnn_baseline.pth --image_path data/chest_xray/test/NORMAL/example.jpeg
```

## Results

Baseline CNN: Accuracy: 0.8349, Precision: 0.8643, Recall: 0.7902, F1: 0.8077
Resent Fine-tune: Accuracy: 0.7885, Precision: 0.8502, Recall: 0.7231, F1: 0.7370

# Key Findings

Custom CNN baseline actually outperformed ResNet feature extraction on the validation set. This was unexpected but informative. It demonstrates that the frozen ImageNet features by themselves weren't sufficient for medical X-ray classification.

The CNN could learn domain-specific features from scratch (it is a big network and overfits severely after the first epoch -- needs some tuning here in the future.) This highlights an important lesson: transfer learning isn't always guaranteed improvement, especially when using feature extraction (frozen backbone) rather than full fine-tuning on a pretty large specialized dataset.

# Future Improvements:

To improve this, I would use full fine-tuning rather than feature extraction (unfreezen the entire backbone) and do a two-stage training where I start with the frozen backbone and gradually unfreeze layers. I would use different learning rates for the backbone and the classifier head (higher for classifier head). Also, need some regularizatoin / lower learning rates in general to prevent the overfitting seen in the loss curves.

## Project Structure

```
end_to_end_pneumonia_detection/
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── cnn_baseline.py     # Custom CNN architecture
│   └── resnet_finetune.py  # ResNet with transfer learning
├── train.py                # Training pipeline
├── eval.py                 # Evaluation with metrics
├── inference.py            # Single-image prediction
├── checkpoints/            # Saved model weights
└── plots/                  # Training curves and confusion matrices
```