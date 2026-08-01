# Personal Loan Campaign Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-f7931e.svg)](https://scikit-learn.org/)

## Project Overview

AllLife Bank wants to grow its personal-loan portfolio by converting existing liability customers—customers who currently hold deposits but have not taken a personal loan—into loan customers.

This project builds an interpretable **Decision Tree classification model** to:

- Predict whether an existing customer is likely to accept a personal loan.
- Identify the customer attributes that most influence loan acceptance.
- Recommend high-potential customer segments for targeted marketing campaigns.
- Reduce model overfitting through pre-pruning and cost-complexity post-pruning.

## Business Problem

A blanket marketing campaign can be expensive and inefficient. The bank needs a data-driven approach to identify customers with a higher probability of accepting a personal-loan offer.

The analysis answers three main questions:

1. Which customers are most likely to purchase a personal loan?
2. Which demographic and financial variables drive loan acceptance?
3. How can the bank use these insights to improve campaign effectiveness?

## Repository Contents

| File | Description |
|---|---|
| [`PersonalLoanCampaign.ipynb`](PersonalLoanCampaign.ipynb) | Complete Python notebook containing data analysis, preprocessing, model development, pruning, evaluation, and recommendations. |
| [`Personal_Loan_Campaign.html`](Personal_Loan_Campaign.html) | Executed HTML version of the notebook with outputs and visualizations. |
| `README.md` | Project documentation. |

## Dataset

The dataset contains **5,000 customer records** and **14 original columns**, covering customer demographics, financial behaviour, product ownership, and loan-response information.

### Target Variable

- `Personal_Loan = 1`: Customer accepted the personal-loan offer.
- `Personal_Loan = 0`: Customer did not accept the offer.

Only about **9.6%** of customers accepted a personal loan, making the target variable imbalanced.

### Main Features

| Feature | Description |
|---|---|
| `Age` | Customer age. |
| `Experience` | Number of years of professional experience. |
| `Income` | Customer's annual income. |
| `Family` | Customer's family size. |
| `CCAvg` | Average monthly credit-card spending. |
| `Education` | Education category. |
| `Mortgage` | Home-mortgage value. |
| `Securities_Account` | Whether the customer holds a securities account. |
| `CD_Account` | Whether the customer holds a certificate-of-deposit account. |
| `Online` | Whether the customer uses online banking. |
| `CreditCard` | Whether the customer holds a bank-issued credit card. |

> The dataset file is loaded in the notebook as `Loan_Modelling.csv` from Google Drive and is not currently included in this repository.

## Project Workflow

### 1. Data Inspection

- Reviewed the dataset shape, sample records, column data types, and class distribution.
- Confirmed that all 5,000 records contain values for the original 14 columns.
- Identified the low personal-loan acceptance rate and resulting class imbalance.

### 2. Exploratory Data Analysis

The notebook performs univariate and bivariate analysis using numerical distributions, categorical count plots, and loan-response comparisons.

Key observations include:

- Loan customers generally have higher income.
- Higher credit-card spending is associated with loan acceptance.
- Customers with advanced or professional education are more receptive.
- Customers with family sizes of three or four show slightly higher acceptance.
- CD-account holders have a considerably stronger likelihood of accepting a loan.

### 3. Data Preprocessing

- Checked for missing values.
- Replaced negative `Experience` values using the logical approximation `Age - 22`.
- Retained outliers while applying log transformations to skewed variables:
  - `CCAvg_log = log(1 + CCAvg)`
  - `Income_log = log(1 + Income)`
- Removed `ID` and `ZIPCode` because they were not considered useful predictive variables.
- Used a **70:30 stratified train-test split** with `random_state=42`.

### 4. Model Development

A Decision Tree classifier was selected because it provides both predictive performance and interpretable decision rules.

The notebook compares:

1. An unrestricted base Decision Tree.
2. A pre-pruned Decision Tree using depth and leaf-size constraints.
3. A post-pruned Decision Tree using cost-complexity pruning.

### 5. Model Evaluation

Models are evaluated using:

- Precision
- Recall
- F1-score
- Accuracy
- ROC-AUC
- Confusion matrix
- Decision-tree visualization

## Model Results

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Base Decision Tree | 0.98 | 0.90 | 0.92 | 0.91 | 0.9532 |
| Pre-pruned Decision Tree | 0.98 | 0.90 | 0.85 | 0.87 | 0.9940 |
| Post-pruned Decision Tree | **0.99** | **0.94** | **0.94** | **0.94** | **0.9968** |

The post-pruned model produced the strongest overall result. Its selected cost-complexity parameter was approximately:

```text
ccp_alpha = 0.0009328067
```

## Key Predictors

The most influential variables identified in the analysis are:

1. **Income**
2. **Average credit-card spending (`CCAvg`)**
3. **CD-account ownership**
4. **Education level**

These variables provide clear and actionable customer-selection criteria for the bank's campaign team.

## Business Recommendations

The bank should prioritise personalised personal-loan campaigns for:

- High-income customers with high monthly credit-card spending.
- Customers with advanced or professional education.
- Existing CD-account holders who may respond well to cross-selling.
- Families with three or four members as a secondary target segment.

Instead of running blanket campaigns, the bank can integrate the model's decision rules into its CRM or campaign-management system to flag customers with a high probability of conversion. This can improve response rates while reducing unnecessary marketing cost.

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Running the Project

### Option 1: Google Colab

1. Open [`PersonalLoanCampaign.ipynb`](PersonalLoanCampaign.ipynb).
2. Select **Open in Colab**.
3. Upload `Loan_Modelling.csv` to Google Drive.
4. Update the dataset path in the notebook when required:

```python
df = pd.read_csv("/content/drive/MyDrive/MLProject/Loan_Modelling.csv")
```

5. Run all cells in sequence.

### Option 2: Local Jupyter Environment

Clone the repository:

```bash
git clone https://github.com/PrasadSomya/ML.git
cd ML
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Before running locally, replace the Google Drive dataset path with a local path, for example:

```python
df = pd.read_csv("Loan_Modelling.csv")
```

## Future Improvements

- Compare the Decision Tree with Random Forest, XGBoost, and Logistic Regression.
- Use cross-validation for more robust model selection.
- Evaluate class-weighting or resampling techniques for the imbalanced target.
- Tune the classification threshold based on campaign cost and expected revenue.
- Add SHAP-based explanations for individual customer predictions.
- Deploy the final model as a simple campaign-scoring application or API.

## Author

**Somya Prasad**

GitHub: [PrasadSomya](https://github.com/PrasadSomya)

---

This project demonstrates how interpretable machine learning can support targeted banking campaigns and convert model insights into practical business actions.
