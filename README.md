# PEFT Sentiment Analysis with DistilBERT using LoRA

This project demonstrates Parameter Efficient Fine Tuning (PEFT) using Low Rank Adaptation (LoRA) to fine tune a pre trained DistilBERT model for binary sentiment classification on the IMDb movie reviews dataset.

Instead of updating all model parameters, LoRA trains only a small number of additional adapter weights, making fine tuning significantly more memory and compute efficient while maintaining strong performance.

## Features

* Uses DistilBERT as the base transformer model
* Parameter Efficient Fine Tuning with LoRA
* Binary sentiment classification on the IMDb dataset
* Hugging Face Transformers Trainer API
* Early stopping and best model checkpointing
* Performance evaluation using:
  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC AUC
* Confusion Matrix visualization
* ROC Curve visualization

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

Fine Tuning Method

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

## Project Workflow

1. Install required libraries
2. Load IMDb dataset
3. Tokenize reviews
4. Split training and validation data
5. Load DistilBERT model
6. Apply LoRA adapters
7. Train the model
8. Evaluate performance
9. Generate predictions
10. Plot Confusion Matrix and ROC Curve

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

Move into the project directory

```bash
cd your-repository
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

The notebook trains a LoRA adapted DistilBERT model for sentiment classification and reports multiple evaluation metrics along with visualizations such as the confusion matrix and ROC curve.

## License

This project is intended for educational and research purposes.
