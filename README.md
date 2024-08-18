# Bank Churn Prediction with LightGBM

This project is focused on predicting customer churn for a bank using the LightGBM machine learning algorithm. The notebook provides a step-by-step guide to data preprocessing, feature engineering, model training, and evaluation.

## Dataset

The dataset used in this project is sourced from the [Kaggle Playground Series Season 4, Episode 1 competition](https://www.kaggle.com/competitions/playground-series-s4e1). It includes customer data such as demographics, transaction history, and account information. The target variable is whether a customer has exited (churned) or not.

### Key Features:
- **CreditScore:** The credit score of the customer.
- **Geography:** The location of the customer.
- **Gender:** The gender of the customer.
- **Age:** The age of the customer.
- **Balance:** The account balance.
- **NumOfProducts:** Number of products held by the customer.
- **HasCrCard:** Whether the customer has a credit card.
- **IsActiveMember:** Whether the customer is an active member.
- **EstimatedSalary:** The estimated salary of the customer.

## Steps Included in the Notebook

### 1. **Data Loading and Exploration**
   - The dataset is loaded and basic exploration is performed to understand the distribution of features and identify any missing values.

### 2. **Data Preprocessing**
   - **Missing Values:** Imputation of missing values for columns such as `Geography`, `Age`, and `HasCrCard`.
   - **Outlier Handling:** Outliers in numerical columns are adjusted to prevent skewed model performance.
   - **Skewness Fixing:** Skewed distributions are transformed for more effective modeling.

### 3. **Feature Engineering**
   - **One-Hot Encoding:** Categorical variables like `Geography` and `Gender` are encoded.
   - **Feature Selection:** Forward feature selection is performed to identify the most relevant features for the model.

### 4. **Model Training and Hyperparameter Tuning**
   - **LightGBM Model:** The LightGBM algorithm is used to build a predictive model.
   - **Hyperparameter Tuning:** Bayesian optimization is employed to fine-tune model parameters for optimal performance.

### 5. **Model Evaluation**
   - **Classification Report:** The performance of the model is evaluated using metrics like precision, recall, and F1-score.
   - **Confusion Matrix:** A confusion matrix is plotted to visualize the performance on the validation set.
   - **ROC-AUC Curve:** The ROC-AUC curve is plotted to assess the model's ability to distinguish between classes.

## Results

The model's performance on the training set is summarized below:

- **Accuracy:** The overall accuracy of the model is 75%, meaning that 75% of all predictions made by the model are correct.
- **Precision:** 
  - For class 0 (non-churned customers), the precision is 96%, indicating that when the model predicts a customer will not churn, it is correct 96% of the time.
  - For class 1 (churned customers), the precision is 46%, meaning that when the model predicts a customer will churn, it is correct only 46% of the time.
- **Recall:** 
  - For class 0, the recall is 71%, showing that the model correctly identifies 71% of the non-churned customers.
  - For class 1, the recall is 90%, indicating that the model correctly identifies 90% of the churned customers.
- **F1-Score:** 
  - For class 0, the F1-score is 82%, which balances precision and recall to give a single performance metric.
  - For class 1, the F1-score is 60%, indicating that there is room for improvement in the model's ability to accurately predict churned customers.
- **Macro Average:** The macro average F1-score is 71%, which is the average of the F1-scores for both classes, treating them equally regardless of their support.
- **Weighted Average:** The weighted average F1-score is 77%, which takes into account the support (number of true instances for each class) when averaging.

Overall, while the model is highly precise in predicting non-churned customers, it tends to misclassify many of them as churned (low recall for class 0). However, it performs well in identifying actual churned customers (high recall for class 1), though with lower precision. The model achieves a balanced performance but could benefit from further optimization, especially in improving the precision for predicting churned customers.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `lightgbm`

## How to Run

1. Clone the repository.
2. Install the required packages.
3. Run the Jupyter notebook to follow the step-by-step implementation.

## Data Source

The data used in this project can be found on [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e1) as part of the Playground Series Season 4, Episode 1 competition.

## Conclusion

This notebook provides a comprehensive approach to predicting customer churn using LightGBM, from data preprocessing to model evaluation. It serves as a practical guide for building machine learning models in a business context.
