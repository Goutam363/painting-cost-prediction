# Building Painting Cost Estimation Using Machine Learning

A machine-learning project for estimating building painting costs using historical Bill of Quantities (BOQ) records. The project covers data preprocessing, project-aware dataset splitting, regression model development, performance evaluation, energy-consumption estimation, visualization, and deployment through a Streamlit web application.

## Live Demo

**Try the deployed application:** https://painting-cost-prediction.streamlit.app/

The application allows users to select painting items and project details, enter the required quantity and project start date, and obtain an estimated painting cost in Indian Rupees (INR).

**Support:** The application integrates the Support Vector Machine (SVM) model, which achieved the highest R² score among the evaluated models.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Repository Structure](#repository-structure)
- [Dataset and Features](#dataset-and-features)
- [Workflow](#workflow)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Energy Measurement](#energy-measurement)
- [Streamlit Application](#streamlit-application)
- [Requirements](#requirements)
- [How to Run the Project](#how-to-run-the-project)
- [Figures and Outputs](#figures-and-outputs)
- [Reproducibility and Notes](#reproducibility-and-notes)

## Project Overview

Estimating painting costs is an important part of building construction estimation. This project uses historical BOQ line-item records and project-level information to predict the painting cost recorded in `amount_inr`.

The project compares multiple machine-learning regression models to identify an effective approach for painting cost estimation.

The workflow includes:

1. Merging BOQ line-item data with project-level information.
2. Filtering records belonging to the `Painting` sub-category.
3. Cleaning the dataset and preparing input features and the target variable.
4. Splitting the dataset by project to reduce data leakage between training and testing.
5. Training and evaluating Linear Regression, Decision Tree, and Support Vector Machine (SVM) models.
6. Comparing model performance using MAE, RMSE, and R².
7. Measuring execution time and estimating energy consumption for selected computational stages.
8. Generating figures for dataset analysis, prediction errors, model comparison, and energy consumption.
9. Integrating the best-performing model into a Streamlit web application.

**Key result:** SVM achieved the highest R² score among the evaluated models and was selected for integration into the interactive cost prediction application.

## Objectives

- Prepare a cleaned dataset of building-painting BOQ records.
- Predict painting costs using item-level and project-level features.
- Compare regression models using MAE, RMSE, and R².
- Use project-aware train/test splitting and cross-validation.
- Evaluate computational runtime and estimated energy consumption.
- Generate figures for model comparison and performance analysis.
- Deploy an interactive application that provides painting cost estimates.

## Repository Structure

The repository contains the source datasets, preprocessing and modeling notebooks, generated CSV files, trained models, figures, and Streamlit application.

```text
Final_ML_Project
    │
    ├──CompleteCodeBase_with_Readme/
    |    │
    |    ├── Dataset/
    |    │
    |    ├── figures/
    |    │
    |    ├── painting-cost-app/
    |    │   ├── app.py
    |    │   ├── svm_model.joblib
    |    │   ├── painting_dataset_cleaned.csv
    |    │   └── requirements.txt
    |    │
    |    ├── .gitignore
    |    │
    |    ├── MainCode.ipynb
    |    ├── Preprocess.ipynb
    |    │
    |    ├── painting_dataset.csv
    |    ├── painting_dataset_cleaned.csv
    |    │
    |    ├── X_train.csv
    |    ├── X_test.csv
    |    ├── y_train.csv
    |    ├── y_test.csv
    |    ├── groups_train.csv
    |    ├── groups_test.csv
    |    │
    |    ├── linear_regression_model.joblib
    |    ├── linear_regression_metrics.csv
    |    ├── linear_regression_test_predictions.csv
    |    ├── linear_regression_energy_summary.csv
    |    │
    |    ├── decision_tree_model.joblib
    |    ├── decision_tree_metrics.csv
    |    ├── decision_tree_test_predictions.csv
    |    ├── decision_tree_energy_summary.csv
    |    │
    |    ├── svm_model.joblib
    |    ├── svm_metrics.csv
    |    ├── svm_test_predictions.csv
    |    ├── svm_energy_summary.csv
    |    │
    |    ├── energy_measurements.csv
    |    │
    |    └── README.md
    ├── ML_Project_9_Group_3_Presentation.ppt
    |
    └── ML_Project_9_Group_3_Report.pdf
```

*Note: The structure above lists the principal files. The `Dataset/` and `figures/` directories contain their respective source data and generated figures. The app directory should contain the SVM model artifact referenced by `app.py`.*

## Dataset and Features

### Source data

The `Dataset/` directory contains the source data used in the project.

`Preprocess.ipynb` merges the BOQ line-item records with project information using `project_id`, filters records whose `sub_category` is `Painting`, and produces the processed datasets.

### Target variable

- `amount_inr`: The painting line-item cost in Indian Rupees (INR). This is the regression target.

### Input features

| Feature | Description |
|---|---|
| `item_name` | Name or description of the painting item |
| `quantity` | Quantity recorded for the BOQ item |
| `building_type` | Type of building |
| `city` | Project city |
| `state` | Project state |
| `project_year` | Year extracted from the project start date |
| `project_month` | Month extracted from the project start date |

The modeling workflow extracts `project_year` and `project_month` from `project_start_date`. The original date column is then removed from the model input.

The `project_id` is used to group the training and testing sets and cross-validation folds. It is not used as a predictive feature.

## Workflow

### 1. Data preprocessing

Open and execute `Preprocess.ipynb` from top to bottom.

The notebook performs the following operations:

1. Loads the BOQ line-item and project-level CSV files.
2. Merges the records using `project_id`.
3. Filters the records to the `Painting` sub-category.
4. Removes duplicate records.
5. Converts selected numeric columns to appropriate numeric types.
6. Removes records with missing essential values.
7. Removes records with negative quantities or painting costs.
8. Cleans text fields and exports the processed datasets.

Main outputs:

- `painting_dataset.csv`
- `painting_dataset_cleaned.csv`

### 2. Project-aware train/test splitting

The modeling workflow in `MainCode.ipynb` uses `GroupShuffleSplit` with:

- Test size: `0.30`
- Random state: `42`
- Grouping variable: `project_id`

Grouping by project ensures that records from the same project are not distributed between the training and testing sets.

The resulting files are:

- `X_train.csv` and `X_test.csv`: Input features.
- `y_train.csv` and `y_test.csv`: Target values.
- `groups_train.csv` and `groups_test.csv`: Project IDs corresponding to the respective feature records.

The recorded experiment contained 846 training samples and 353 testing samples, with 521 training projects and 224 testing projects. The exact sample proportion can differ slightly from 70:30 because the split is performed by project.

### 3. Model training and evaluation

Three regression approaches are evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Support Vector Machine (SVM)

Each model is evaluated using the same project-aware evaluation methodology to support a fair comparison.

The repository contains the trained model artifacts, metrics, test predictions, and energy summaries for these approaches.

### 4. Model selection

The models are compared using MAE, RMSE, and R².

**SVM achieved the highest R² score among the evaluated models.** Therefore, SVM was selected as the final model for the Streamlit application.

## Machine Learning Methodology

### Preprocessing pipeline

The machine-learning pipelines use a `ColumnTransformer` to handle numerical and categorical features.

- Numerical features are imputed using the median.
- Categorical features are imputed using the most frequent value.
- Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")`.

Keeping preprocessing within the model pipeline ensures that imputers and encoders are fitted on the appropriate training data during cross-validation.

### Project-aware cross-validation

The workflow uses `GroupKFold` with up to five folds.

Each fold groups observations by `project_id`, reducing the risk of data leakage between training and validation records from the same project.

The final test set is held out by project and used to report final test performance after model selection.

### Regression models

**Linear Regression**

A linear baseline model that estimates the relationship between the input features and painting cost.

**Decision Tree Regressor**

A tree-based model that captures nonlinear relationships using feature-based decision rules.

**Support Vector Machine (SVM)**

A support-vector-based regression approach used to estimate painting costs. Its performance is evaluated against the other models using the same regression metrics.

The SVM model achieved the highest R² score among the evaluated models and is used in the deployed prediction application.

## Evaluation Metrics

The project evaluates regression performance using three standard metrics.

### Mean Absolute Error (MAE)

MAE is the average absolute difference between actual and predicted costs.

A lower MAE indicates smaller average absolute prediction errors.

### Root Mean Squared Error (RMSE)

RMSE is the square root of the average squared prediction error. It penalizes large errors more heavily than MAE.

A lower RMSE generally indicates better predictive performance.

### Coefficient of Determination (R²)

R² measures how well the model explains the variation in the target values.

- An R² value closer to 1 generally indicates better predictive performance.
- An R² value of 0 indicates performance comparable to predicting the test-set mean, under the standard R² definition.
- A negative R² indicates performance worse than predicting the test-set mean.

The models should be compared on the same held-out test set for a meaningful comparison.

MAE and RMSE are reported in INR.

For plots expressed in lakh INR:

`1 lakh INR = 100,000 INR`

## Energy Measurement

CodeCarbon is used to estimate energy consumption for selected computational stages.

The estimated energy in kWh is converted into joules.

The energy summary CSV files include stage-level information such as execution time and estimated energy consumption.

Depending on the model implementation, the measured stages may include:

- Model training
- Cross-validation
- Hyperparameter tuning
- Test-set prediction
- Other selected computational operations

The relevant output files are:

- `linear_regression_energy_summary.csv`
- `decision_tree_energy_summary.csv`
- `svm_energy_summary.csv`
- `energy_measurements.csv`

**Important:** CodeCarbon provides software-based energy estimates rather than direct measurements from a physical power meter. Results can vary with hardware, background processes, monitoring overhead, and execution duration. Energy comparisons are most meaningful when the measurement conditions and computational stages are comparable.

## Streamlit Application

### Live deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** https://painting-cost-prediction.streamlit.app/

### Application features

The application allows users to provide:

- Painting item name
- Building type
- Quantity
- State
- City
- Project start date

The application processes the input features and uses the saved SVM model to estimate the painting cost.

The project start date is transformed into `project_year` and `project_month` to match the features expected by the trained model.

### Application files

The `painting-cost-app/` directory contains the Streamlit interface, the SVM model artifact, the dataset used to populate input options, and the application dependencies.

- `app.py`: Streamlit interface and prediction logic.
- `svm_model.joblib`: Saved SVM model pipeline used for predictions.
- `painting_dataset_cleaned.csv`: Cleaned dataset used to populate application options.
- `requirements.txt`: Python dependencies for the application.

### Run the application locally

Make sure Python is installed. Open a terminal from the repository root and run:

```bash
cd painting-cost-app
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### Deployment and updates

The application is hosted on Streamlit Community Cloud and connected to a GitHub repository.

To update the deployed application:

1. Modify the required application code or model.
2. Ensure the app references the correct SVM model and feature pipeline.
3. Commit and push the changes to the connected GitHub branch.

For example, from the repository root:

```bash
git add painting-cost-app/
git commit -m "Integrate SVM cost prediction model"
git push
```

Streamlit Community Cloud can detect changes to the connected repository and redeploy the application.

**Note:** The deployed application must contain every required file, including the compatible saved model artifact and dataset. The scikit-learn version used to load a saved model should be compatible with the version used to train it.

The predictions are estimates based on the trained model and supplied features. They should not be treated as guaranteed project quotations.

## Requirements

For the notebooks and model experiments, Python 3.10 or a compatible newer version is recommended.

The principal dependencies include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `joblib`
- `codecarbon`
- `jupyter`

Install the notebook dependencies using:

```bash
python -m pip install pandas numpy scikit-learn matplotlib scipy joblib codecarbon jupyter
```

The Streamlit application has a separate `requirements.txt` file in `painting-cost-app/`. Use that file to install the dependencies required by the deployed interface.

## How to Run the Project

Run the commands and notebooks from the repository root so that the relative CSV paths resolve correctly.

### Step 1: Prepare the dataset

- Confirm that the source CSV files are present in `Dataset/`.
- Open `Preprocess.ipynb`.
- Execute all cells.
- Verify that `painting_dataset.csv` and `painting_dataset_cleaned.csv` are generated.

### Step 2: Generate the train/test split

- Open `MainCode.ipynb`.
- Execute the initial data preparation and splitting cells.
- Verify that the feature, target, and group CSV files are available.

### Step 3: Train and evaluate the models

- Execute the Linear Regression, Decision Tree, and SVM sections in `MainCode.ipynb`.
- Verify that the model, metrics, prediction, and energy-summary files are generated.
- Compare the models using MAE, RMSE, and R².

### Step 4: Generate the figures

- Run the plotting cells in `MainCode.ipynb`, where applicable.
- Save the generated figures in the `figures/` directory.
- Ensure that the required model output CSV files exist before generating plots that depend on them.

### Step 5: Run the Streamlit application

Follow the [local application instructions](#run-the-application-locally), or open the [deployed application](https://painting-cost-prediction.streamlit.app/).

### Step 6: Reproduce the experiment after changing the data or code

- Rerun preprocessing.
- Recreate the project-aware train/test split.
- Retrain and evaluate the models.
- Regenerate the figures.
- Update the application model and associated files if the selected model changes.
- Commit and push the updated application files if the deployed version should be updated.

## Figures and Outputs

The `figures/` directory contains the generated figures used for analysis, model comparison, and reporting.

### Figure categories

Depending on the plotting cells executed, the figures can include:

- Painting-cost target distribution.
- MAE and RMSE comparison across models.
- R² comparison across models.
- Energy-consumption comparison.
- Actual versus predicted painting costs.
- Residuals versus predicted costs.

The figures are maintained in one directory rather than separate MATLAB and ordinary Python plotting directories.

### Dataset and split files

| File | Purpose |
|---|---|
| `painting_dataset.csv` | Merged painting-category dataset before final cleaning |
| `painting_dataset_cleaned.csv` | Cleaned dataset used in the modeling workflow |
| `X_train.csv`, `X_test.csv` | Training and testing features |
| `y_train.csv`, `y_test.csv` | Training and testing targets |
| `groups_train.csv`, `groups_test.csv` | Project IDs aligned with each split |

### Model artifacts and results

| File | Purpose |
|---|---|
| `linear_regression_model.joblib` | Saved Linear Regression model pipeline |
| `linear_regression_metrics.csv` | Linear Regression evaluation metrics |
| `linear_regression_test_predictions.csv` | Linear Regression test predictions |
| `linear_regression_energy_summary.csv` | Linear Regression energy and runtime summary |
| `decision_tree_model.joblib` | Saved Decision Tree model pipeline |
| `decision_tree_metrics.csv` | Decision Tree evaluation metrics |
| `decision_tree_test_predictions.csv` | Decision Tree test predictions |
| `decision_tree_energy_summary.csv` | Decision Tree energy and runtime summary |
| `svm_model.joblib` | Saved SVM model pipeline |
| `svm_metrics.csv` | SVM evaluation metrics |
| `svm_test_predictions.csv` | SVM test predictions |
| `svm_energy_summary.csv` | SVM energy and runtime summary |
| `energy_measurements.csv` | CodeCarbon measurement output |

### Interpreting the figures

**Actual versus predicted costs**

Each point represents a held-out test observation. Points closer to the diagonal reference line indicate predictions closer to the actual costs.

**Residuals versus predicted costs**

Residuals are calculated as:

```text
Residual = Actual cost - Predicted cost
```

Positive residuals indicate underprediction, while negative residuals indicate overprediction. Systematic patterns may indicate prediction errors or limitations in the model.

**Model comparison**

- MAE: Lower values indicate smaller average absolute errors.
- RMSE: Lower values indicate smaller errors with less impact from large deviations.
- R²: Higher values generally indicate better performance on the same test set.
- Energy consumption: Lower values indicate lower estimated energy use when the measured stages and conditions are comparable.