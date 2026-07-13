# Credit Card Fraud Detection using Machine Learning

An end-to-end Machine Learning project to detect fraudulent credit card transactions using **Logistic Regression**. Since the original dataset is highly imbalanced, this project utilizes **Under-Sampling** techniques to balance the distribution of normal and fraudulent transactions to build an accurate and robust classification model.

## 📌 Features
- **Data Preprocessing & Analysis:** Detailed statistical analysis of transaction times, amounts, and features.
- **Handling Imbalance:** Implements Under-Sampling to balance the class distribution (492 normal vs. 492 fraudulent transactions).
- **Machine Learning Pipeline:** Trains a **Logistic Regression** classifier using `scikit-learn`.
- **Performance Evaluation:** Evaluates model performance using Accuracy Score.

---

## 📊 Dataset Overview
The dataset contains transactions made by credit cards in September 2013 by European cardholders. 
* **Total Transactions:** 284,807
* **Legitimate Transactions (Class 0):** 284,315 (99.82%)
* **Fraudulent Transactions (Class 1):** 492 (0.17%)
* **Features:** 30 numerical features (`V1` to `V28` resulting from PCA transformation, `Time`, and `Amount`).

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone (https://github.com/Ibragoran10/credit-card-fraud-detection.git)
   cd credit-card-fraud-detection

Install dependencies:
Make sure you have Python installed, then run:

Bash
pip install numpy pandas scikit-learn
Dataset Setup:
Download the creditcard.csv dataset and place it in the root directory of the project.

---

## 🚀 Workflow & Implementation
The implementation follows these steps in the Jupyter Notebook:

Load the Data: Load and inspect the dataset using Pandas.

Handle Missing Values: Check and clean any missing data.

Data Analysis: Compare mean statistics of both classes (Legitimate vs. Fraudulent).

Under-Sampling: Extract a random sample of 492 legitimate transactions to combine with the 492 fraudulent transactions.

Split Data: Divide the balanced dataset into Features (X) and Targets (Y), then perform a Train-Test split.

Model Training: Train the LogisticRegression model.

Evaluation: Measure the accuracy of training and testing data.

---

## ⚙️ Technologies Used
Python

Pandas & NumPy (Data Manipulation)

Scikit-learn (Machine Learning & Evaluation)
