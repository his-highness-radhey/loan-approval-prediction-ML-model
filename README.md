# Loan Approval Prediction

A machine learning classification workflow developed as part of a course in Jupyter Notebook (`code.ipynb`), predicting whether a loan application will be approved (`Loan_Approved`) based on applicant financials, demographics, credit profile, and loan terms.

## Dataset Overview
- **Source:** `loan_approval_data.csv` (1,000 rows, 20 columns initially, with 50 missing values across columns).
- **Target:** `Loan_Approved` (label-encoded binary target).
- **Key Features:** `Applicant_Income`, `Coapplicant_Income`, `Credit_Score`, `DTI_Ratio`, `Existing_Loans`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Employment_Status`, `Property_Area`, `Education_Level`, and `Employer_Category`.

## Workflow & Pipeline
1. **Missing Value Imputation:**
   - Numerical features imputed using `SimpleImputer(strategy="mean")`.
   - Categorical features imputed using `SimpleImputer(strategy="most_frequent")`.
2. **Exploratory Data Analysis (EDA):**
   - Class balance check on `Loan_Approved`, category counts (`Gender`), `Credit_Score` distribution by approval status, `Applicant_Income` boxplots, and correlation analysis (`Credit_Score` positively correlated; `DTI_Ratio` and `Loan_Amount` negatively correlated).
3. **Encoding & Scaling:**
   - Dropped `Applicant_ID`.
   - Applied `LabelEncoder` to `Education_Level` and `Loan_Approved`, and `OneHotEncoder(drop="first", handle_unknown="ignore")` to remaining categorical columns.
   - Split data (`80/20` train-test split, `random_state=42`) and scaled features with `StandardScaler`.
4. **Baseline Models (Before Feature Engineering):**
   - **Logistic Regression:** Accuracy `0.865`, Precision `0.783`, Recall `0.770`, F1 `0.777`
   - **K-Nearest Neighbors (`n_neighbors=5`):** Accuracy `0.760`, Precision `0.627`, Recall `0.525`, F1 `0.571`
   - **Gaussian Naive Bayes:** Accuracy `0.865`, Precision `0.804`, Recall `0.738`, F1 `0.769`
5. **Feature Engineering & Re-evaluation:**
   - Added `DTI_Ratio_sq`, `Credit_Score_sq`, and `Applicant_Income_log` (`np.log1p`), dropping the raw `DTI_Ratio`, `Credit_Score`, and `Applicant_Income` columns.
   - **Logistic Regression (Post-FE):** Accuracy `0.870`, Precision `0.778`, Recall `0.803`, F1 `0.790`
   - **Gaussian Naive Bayes (Post-FE):** Accuracy `0.865`, Precision `0.804`, Recall `0.738`, F1 `0.769`

## Tech Stack
- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (`SimpleImputer`, `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `train_test_split`, `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`)

## How to Run
1. Place `loan_approval_data.csv` in the same directory as `code.ipynb`.
2. Launch Jupyter Notebook or JupyterLab and run all cells in `code.ipynb`.

## Course Context
Created as a course assignment/project to practice end-to-end tabular data preprocessing, exploratory analysis, feature engineering, and baseline classification model evaluation.
