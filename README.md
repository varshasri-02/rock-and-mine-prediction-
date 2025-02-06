# Rock-Mine Prediction Using Logistic Regression

This project implements a machine learning model to predict whether an object is a rock or a mine based on certain features. We use Logistic Regression for classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Libraries Used](#libraries-used)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [How to Run](#how-to-run)
8. [Requirements](#requirements)

---

## Project Overview

In this project, the goal is to classify objects into two categories: rock or mine. The data consists of features that represent various characteristics of the objects. Logistic Regression is used to train the model and predict the type of object based on these features.

## Dataset

The dataset contains 208 instances and 61 features (60 input features and 1 output label). The labels indicate whether the object is a rock or a mine:
- **M**: Mine
- **R**: Rock

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For visualizations.
- `seaborn`: For advanced visualizations.
- `sklearn`: For implementing machine learning models (Logistic Regression) and evaluating the model.

## Data Preprocessing

The data preprocessing steps include:
1. **Loading the dataset**.
2. **Encoding the target labels** ("M" for mine and "R" for rock) using label encoding.
3. **Splitting the dataset** into training and testing sets.
4. **Feature scaling** to standardize the input features using `StandardScaler`.

## Model Training

The model is trained using Logistic Regression for binary classification. The classifier is trained on the training data and evaluated on the test data.

## Model Evaluation

The model is evaluated using:
1. **Accuracy**: Measures how well the model performs overall.
2. **Confusion Matrix**: Shows the true positive, true negative, false positive, and false negative results.
3. **Classification Report**: Provides precision, recall, and F1-score for a detailed evaluation of the classifier's performance.

## How to Run

1. Clone or download the repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
