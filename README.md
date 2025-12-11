# Plant Disease Classification using Deep Learning

A production-ready PyTorch implementation for classifying plant diseases from leaf images using both custom CNN and transfer learning approaches.

## Features

- **Two Model Architectures**: Custom CNN and ResNet50 transfer learning
- **Complete Training Pipeline**: With early stopping, learning rate scheduling, and checkpointing
- **Interactive Web App**: Streamlit-based UI for real-time predictions
- **Production-Ready**: Error handling, logging, type hints, and modular code structure

## Dataset Structure

```
Dataset/
├── Training/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
├── Validation/
│   ├── Early_Blight/
│   ├── Healthy/
│   └── Late_Blight/
└── Testing/
    ├── Early_Blight/
    ├── Healthy/
    └── Late_Blight/
```

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train both models (Custom CNN and ResNet50):
```bash
python main.py --mode train --model all
```

Train specific model:
```bash
python main.py --mode train --model cnn
python main.py --mode train --model resnet
```

### Evaluation

Evaluate trained models:
```bash
python main.py --mode evaluate --model all
```

### Streamlit Application

Launch the web interface:
```bash
streamlit run app/app.py
```

## Project Structure

## Complete File Structure

```
Deep-Learning-AOL/
├── Dataset/                         # Dataset folders
│   ├── Training/
│   ├── Validation/
│   └── Testing/
│
├── notebooks/
│   ├── evaluate_checkpoint.ipynb        # Evaluation of Model Checkpoints
│   ├── experiment.ipynb                 # Model Training & Experimentation
│   └── exploration.ipynb                # Data Exploration & Visualization
│
├── checkpoints/                     # (Auto-created) Model weights
├── logs/                            # (Auto-created) Training logs
├── results/                         # (Auto-created) Evaluation outputs
├── src/
│   ├── config.py                        # Centralized configuration and hyperparameters
│   ├── data.py                          # Data pipeline & DataLoaders
│   ├── model.py                         # Model architectures (Custom CNN & ResNet50)
│   ├── train.py                         # Training pipeline
│   ├── evaluate.py                      # Evaluation & metrics
│   ├── utils.py                         # Utility functions
│   ├── main.py                          # Main execution script
│   └── examples.py                      # Example usage demonstrations
│
├── app/
│   └── app.py                           # Streamlit web application
│
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
├── QUICKSTART.md                    # Quick start guide
├── PROJECT_SUMMARY.md               # This file
└── .gitignore                       # Git ignore rules
```

## Model Architectures

### Custom CNN
- 4 convolutional layers with batch normalization
- MaxPooling and Dropout for regularization
- Fully connected layers with adaptive pooling

### ResNet50 Transfer Learning
- Pretrained on ImageNet
- Fine-tuning last 3 layers
- Custom classifier head

## Performance
Results and trained models are saved in:
- `checkpoints/` - Model weights
- `logs/` - TensorBoard logs
- `results/` - Evaluation plots and metrics

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use fewer workers: `--num-workers 2`

### Dataset Not Found
- Check Dataset folder structure
- Verify paths in `config.py`

### Model Checkpoint Not Found
- Train the model first before evaluation
- Check `checkpoints/` directory