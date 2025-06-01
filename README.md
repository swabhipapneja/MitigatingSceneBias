# Mitigating Scene Bias in Action Recognition
This project is a reproducibility challenge implementation of the NeurIPS 2019 paper: **“Why Can’t I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition” by Jinwoo Choi, Chen Gao, Joseph C. E. Messou, and Jia-Bin Huang.**

## Project Summary
Action recognition models often learn spurious correlations, especially scene-related biases (e.g., associating a playground with baseball). This project implements a debiasing framework using adversarial training and human-mask loss to encourage models to focus on the actual actions, not background scenes.

We reproduce and evaluate the proposed debiasing method on multiple datasets and compare performance with a baseline model trained without any debiasing.

## Project Structure

```bash
MitigatingSceneBias/
├── data/                   # Contains datasets used for training and evaluation
├── models/                 # Contains model architectures and training scripts
├── utils/                  # Utility functions and helper scripts
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and instructions
``` 
## Setup Instructions

1. Clone the Repository

`git clone https://github.com/swabhipapneja/MitigatingSceneBias.git`
`cd MitigatingSceneBias`

2. Create a Virtual Environment (Optional but Recommended)

`python -m venv venv`
`source venv/bin/activate  # On Windows: venv\Scripts\activate`

3. Install Requirements

`pip install -r requirements.txt`

## Training

### Baseline Model Training

python models/train_baseline.py \
  --dataset mini_kinetics \
  --epochs 10 \
  --batch_size 32 \
  --save_path ./models/baseline.pth

### Debiasing Model Training

python models/train_debias.py \
  --dataset mini_kinetics \
  --epochs 10 \
  --alpha_scene 0.5 \
  --alpha_mask 0.5 \
  --save_path ./models/debiasing.pth

## Evaluation
Evaluate the pretrained models on target datasets:

python models/evaluate.py \
  --pretrained_weights ./models/debiasing.pth \
  --target_dataset ucf101 \
  --epochs 30

Supported target datasets:

- ucf101
- hmdb51
- diving48


## Results

| Model              | UCF101 (%) | HMDB51 (%) | Diving48 (%) |
|--------------------|------------|------------|--------------|
| Baseline (scratch) | 34.17      | 21.38      | 12.82        |
| Debiasing Model    | **34.43**  | **22.36**  | **14.21**    |


## Conclusion 
Debiasing improves generalization in transfer learning by reducing reliance on scene-specific features.

## Key Techniques Used

- ResNet3D-18 backbone for feature extraction
- Scene Adversarial Loss (via gradient reversal)
- Human-Mask Confusion Loss
- Frame extraction using opencv + mmcv
- Pseudo scene labels via Places365 model

