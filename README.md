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

The final model achieved a strong performance on the validation set, with the ROC-AUC score indicating robust predictive power. Further improvements could be explored through advanced feature engineering and additional hyperparameter tuning.

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
