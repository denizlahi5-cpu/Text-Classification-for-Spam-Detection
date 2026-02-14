# Text-Classification-for-Spam-Detection
# Spam Detection using Naive Bayes


## Project Overview

This project was developed as part of the course DLBAIPNLP01 – Project: NLP. The goal of the project is to build a machine learning model that can classify emails as spam or non-spam. Spam detection is an important real-world application of machine learning because unwanted emails can contain harmful links, advertisements, or phishing attempts.

In this project, the UCI Spambase dataset was used to train and evaluate a Naive Bayes classifier. The model was implemented in Python using the scikit-learn library.

## Dataset

The dataset used in this project is the Spambase dataset from the UCI Machine Learning Repository.

Dataset link:
https://archive.ics.uci.edu/ml/datasets/spambase

The dataset contains:

4,601 email samples

57 numerical features extracted from email content

1 binary label (1 = spam, 0 = not spam)

Since the dataset already provides numerical features, no additional text preprocessing was required.

## Methodology

The following steps were performed:

The dataset was loaded using the ucimlrepo package.

The data was split into training (80%) and testing (20%) sets.

A Gaussian Naive Bayes classifier was trained on the training data.

The model was evaluated using:

-Accuracy
-Precision
-Recall
-F1-score

A confusion matrix was generated to visualize the classification results.

## Results

The model achieved approximately 80–85% accuracy, depending on the train-test split.

The results show that Naive Bayes is an effective baseline model for spam detection. While the model performs well, there is still room for improvement by testing other algorithms such as Support Vector Machines or ensemble models.

Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- UCI ML Repository API
