# PEFT Sentiment Analysis with DistilBERT using LoRA

This project demonstrates Parameter Efficient Fine Tuning (PEFT) using Low Rank Adaptation (LoRA) to fine-tune a pre-trained DistilBERT model for binary sentiment classification on the IMDb movie reviews dataset.

Instead of updating all model parameters, LoRA trains only a small number of additional adapter weights, making fine-tuning significantly more memory- and compute-efficient while maintaining strong performance.

## Features

* Hugging Face Transformers Trainer API.
* Early stopping and best model checkpointing.
* Confusion Matrix and ROC Curve visualization.

## Dataset

The project uses the IMDb Movie Reviews dataset available through the Hugging Face Datasets library.

- Training samples: 25,000
- Test samples: 25,000
- Labels:
  - 0 = Negative
  - 1 = Positive

## Model

Base Model

```
distilbert-base-uncased
```

Task

```
Binary Sentiment Classification
```

Fine-tuning Method

```
LoRA (Low Rank Adaptation)
```

LoRA Configuration

| Parameter | Value |
|-----------|------:|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.2 |
| Target Modules | q_lin, v_lin |

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 10 |
| Learning Rate | 3e-5 |
| Batch Size | 16 |
| Weight Decay | 0.1 |
| Scheduler | Cosine |
| Warmup Ratio | 0.1 |

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Installation

Clone the repository

```bash
git clone https://github.com/daniyalnazeer55/Distilbert_LoRA_Sentiment_Analysis.git
```

Move into the project directory

```bash
cd Distilbert_LoRA_Sentiment_Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
pre-trained-transformer-full-fine-tuning.ipynb
```

## Results

The notebook trains a LoRA-adapted DistilBERT model for sentiment classification and reports multiple evaluation metrics along with visualizations such as the confusion matrix and ROC curve.
