# Text-Classification-for-Spam-Detection
# Spam Detection using Naive Bayes

## Project Overview
This project implements a spam detection system using the UCI Spambase dataset and a Naive Bayes classifier.

## Dataset
Spambase Dataset  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/spambase  

## Methodology
- Dataset loaded using ucimlrepo
- Train-test split (80/20)
- Gaussian Naive Bayes classifier
- Evaluation using Accuracy, Precision, Recall, F1-score
- Confusion Matrix visualization

## Results
The model achieves approximately 80–85% accuracy depending on data split.

## Technologies Used
- Python
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the script:
   python spam_detection.py
