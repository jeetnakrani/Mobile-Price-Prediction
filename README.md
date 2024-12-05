# Mobile Price Prediction

This project uses machine learning to predict the price range of mobile phones based on various features such as RAM, battery power, screen size, and camera resolution.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

In this project, we aim to predict the price range of mobile phones (4 categories: 0, 1, 2, and 3) using a dataset that contains various features related to mobile phones. The task involves cleaning and preprocessing the data, performing exploratory data analysis (EDA), applying feature engineering, and training multiple machine learning models.

The models are then evaluated and compared based on performance metrics like accuracy, precision, recall, and F1-score. This analysis demonstrates which machine learning algorithm is best suited for this classification task.

## Installation

To run this project locally, you will need to have Python 3.x installed along with the necessary libraries.

### Requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Dataset

The dataset used in this project is `train.csv`, which contains information about various mobile phone features. Key features include:
- `ram`: RAM size in MB
- `battery_power`: Battery power in mAh
- `px_height`, `px_width`: Physical dimensions of the phone screen
- `screen_height`, `screen_width`: Screen dimensions
- `price_range`: Target variable representing the price range (0-3)

### Dataset Overview:
- **Shape**: (2000, 21)
- **No missing values** found in the dataset.
- No duplicate records.

## Exploratory Data Analysis

We performed an extensive EDA to understand the distributions and relationships in the data:
1. **Distribution of Features**: Count plots were used to visualize the distribution of categorical features (e.g., `blue`, `dual_sim`, `wifi`).
2. **Feature Correlations**: A heatmap of feature correlations was created to identify the relationships between various features and the target variable (`price_range`).
3. **Outlier Detection**: Box plots were used to detect outliers, and the Interquartile Range (IQR) method was applied to handle outliers in the `px_height` feature.

## Feature Engineering

New features were created to enhance the predictive power of the model:
- **`px_area`**: Area of the mobile phone screen (calculated from `px_height` and `px_width`).
- **`screen_area`**: Screen area (calculated from `screen_height` and `screen_width`).

## Modeling

Several machine learning models were trained and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **AdaBoost Classifier**

We performed **GridSearchCV** for hyperparameter tuning to optimize model performance. The data was split into training and test sets, and the models were evaluated based on accuracy, precision, recall, and F1-score.

### Key steps in modeling:
1. **Data Scaling**: Numerical features were standardized using `StandardScaler`.
2. **Model Evaluation**: Performance metrics were computed for each model using cross-validation.

## Results

The performance of each model was evaluated, and the following results were obtained:

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| **XGBClassifier**      | 0.83     | 0.83      | 0.83   | 0.83     |
| **RandomForestClassifier** | 0.82     | 0.83      | 0.82   | 0.83     |
| **DecisionTreeClassifier** | 0.80     | 0.82      | 0.80   | 0.81     |
| **LogisticRegression** | 0.78     | 0.78      | 0.78   | 0.78     |
| **AdaBoostClassifier** | 0.64     | 0.68      | 0.64   | 0.64     |

- **XGBClassifier** performed the best, achieving an accuracy of **0.83** and balanced metrics across precision, recall, and F1-score.
- **RandomForestClassifier** was a close second, also performing well in all metrics.
- **DecisionTreeClassifier** and **LogisticRegression** showed reasonable performance but were outperformed by ensemble methods.
- **AdaBoostClassifier** had the lowest performance, indicating that it is not well-suited for this dataset.

## Conclusion

- **XGBClassifier** is the best model for this task, followed by **RandomForestClassifier**.
- **DecisionTreeClassifier** and **LogisticRegression** provide reasonable performance but are not as effective as the ensemble models.
- **AdaBoostClassifier** underperformed compared to other models and may not be suitable for this problem.

Further improvements can be made by tuning hyperparameters, applying different feature engineering techniques, and exploring other ensemble methods.

## License

This project is licensed under the MIT License.

---

