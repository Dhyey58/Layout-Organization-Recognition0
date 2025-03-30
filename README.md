# Layout Organization Recognition

This project implements a deep learning solution for recognizing and segmenting layout elements in historical manuscripts using a U-Net architecture. The model is trained to identify different regions in manuscript images, such as text blocks, illustrations, and margins.

## Project Overview

The project focuses on the following tasks:
- Manuscript layout analysis
- Semantic segmentation of manuscript regions
- Training a U-Net model for pixel-wise classification

## Download Required Files

1. **Dataset**
   - Download the dataset from: [Dhyey56/dataset](https://huggingface.co/datasets/Dhyey56/dataset/tree/main)
   - Extract the downloaded zip file to the project root directory
   - The dataset should be named as `dataset` 

2. **Pre-trained Model**
   - Download the pre-trained model from: [[Model Link]](https://huggingface.co/Dhyey56/final_unet_trained/tree/main)
   - Place the downloaded `.pth` file in the project root directory
   - The model file should be named `final_unet_trained.pth`

## Dataset Structure

The dataset is organized as follows:
```
dataset/
├── img-CB55/
│   └── img/
│       ├── training/
│       ├── validation/
│       └── public-test/
├── img-CS18/
│   └── img/
│       ├── training/
│       ├── validation/
│       └── public-test/
├── img-CS863/
│   └── img/
│       ├── training/
│       ├── validation/
│       └── public-test/
├── PAGE-gt-CB55/
│   └── PAGE-gt/
│       ├── training/
│       ├── validation/
│       └── public-test/
├── PAGE-gt-CS18/
│   └── PAGE-gt/
│       ├── training/
│       ├── validation/
│       └── public-test/
├── PAGE-gt-CS863/
│   └── PAGE-gt/
│       ├── training/
│       ├── validation/
│       └── public-test/
└── pixel-level-gt-*/
    └── pixel-level-gt/
        ├── training/
        ├── validation/
        └── public-test/
```

## Model Architecture

The project uses a U-Net architecture with the following features:
- Encoder: EfficientNet-b0 backbone
- Decoder: Custom decoder with skip connections
- Output: 4-channel segmentation map

## Training Configuration

- Image Size: 256x256
- Batch Size: 4
- Epochs: 10
- Optimizer: Adam
- Loss Function: Combined loss (Dice + BCE)
- Learning Rate: 1e-4

## Dependencies

- Python 3.x
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- scikit-learn

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Dhyey58/Layout-Organization-Recognition0.git
cd Layout-Organization-Recognition0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```python
python train.py
```

4. Evaluate the model:
```python
python evaluate.py
```

## Model Performance

The model's performance is evaluated using two key metrics:

1. **F1 Score**
   - Harmonic mean of precision and recall
   - Provides a balanced measure between precision and recall
   - Particularly useful for imbalanced datasets

2. **Intersection over Union (IoU) / Jaccard Score**
   - Measures the overlap between predicted and ground truth segmentation
   - Calculated as: IoU = intersection(pred, gt) / union(pred, gt)
   - Standard metric for semantic segmentation tasks

## File Structure

```
.
├── training.ipynb          # Main training notebook
├── dataset_structure.txt   # Dataset organization details
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
