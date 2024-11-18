# Machine Learning Project

Welcome to the **Machine Learning** project! This repository implements machine learning models for classification tasks using Python and popular libraries like Scikit-learn and Pandas. It covers data preprocessing, model training, evaluation, and visualization.

---

## Features

- **Data Preprocessing**: Data cleaning and transformation for effective training.
- **Model Training**: Implementation of Decision Tree and Random Forest classifiers.
- **Evaluation**: Metrics such as accuracy, F1-score, confusion matrix, and MSE.
- **Visualization**: Decision tree plotting and data analysis using Seaborn and Matplotlib.

---

## Libraries Used

This project utilizes the following Python libraries:
- **import**
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    mean_squared_error, 
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings


# How to run
1. git clone https://github.com/huuhoang231103/Machine_Learning.git
cd Machine_Learning
2. pip install pandas numpy scikit-learn matplotlib seaborn
3. python DecisionTree.py