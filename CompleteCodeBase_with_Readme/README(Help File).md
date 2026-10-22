# Building Painting Cost Estimation Using Machine Learning

Implementation of a Machine Learning Project to Estimate Building Painting Costs from BOQ Records Research Paper
A description of a machine-learning project that predicts building painting costs, based on Bill of Quantities records. The work covered such aspects as data preprocessing, project-aware train/test splitting, regression modeling, metric evaluation, energy-consumption estimation, plotting, and a Streamlit-based web application to demonstrate the model predictions.

## Live Demo

**Try the deployed application:** https://painting-cost-prediction.streamlit.app/

The Streamlit application allows the user to select a painting item and project details, input a quantity and project start date, and receive an estimated cost of painting.

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
- [Plots and Outputs](#plots-and-outputs)
- [Reproducibility and Notes](#reproducibility-and-notes)

## Project Overview

Estimating painting costs is one part of building construction estimation. This project uses historical BOQ line items and project-level information to predict the painting cost recorded in `amount_inr`.

The workflow includes:

1. Merging BOQ line-item data with project information.
2. Filtering records to the `Painting` sub-category.
3. Cleaning the data and preparing the target and input features.
4. Splitting data by project so the same project does not appear in both training and test sets.
5. Training regression models and evaluating their predictions.
6. Estimating energy consumption for selected model operations with CodeCarbon.
7. Creating plots for data distributions, prediction errors, model comparisons, and energy use.
8. Providing a Streamlit web app for interactive painting-cost prediction.

## Objectives

- Prepare a usable dataset of building-painting BOQ records.
- Predict `amount_inr` from item and project attributes.
- Compare regression models using MAE, RMSE, and R².
- Use project-aware splitting and cross-validation to reduce leakage between records belonging to the same project.
- Record execution time and estimated energy consumption for selected stages.
- Produce figures for analysis and reporting.
- Deploy an interactive application that makes cost predictions from user-provided project details.

## Repository Structure

The repository contains the analysis files, generated outputs, plots, and the Streamlit application. The exact files may vary depending on which notebook cells and scripts have been run.

```text
ML_Project/
├── Dataset/
│   ├── boq_line_items.csv
│   ├── boq_projects.csv
│   ├── data_dictionary_line_items.csv
│   └── data_dictionary_projects.csv
├── MatlabPlotsAndScripts/
│   ├── ActualVsPredictedCost.m
│   ├── ResidualsVsPredictedPaintingCosts.m
│   ├── PaintingCostDistribution.m
│   ├── TestErrorComparisonOfRegressionModels.m
│   ├── ComparisonOfR2_Score.m
│   ├── EnergyConsumptionComparison.m
│   └── associated .fig and .eps files
├── NormalPythonPlots/
│   ├── target_distribution.png
│   ├── test_error_comparison.png
│   ├── test_r2_comparison.png
│   ├── energy_comparison.png
│   ├── linear_regression_actual_vs_predicted.png
│   └── linear_regression_residuals.png
├── painting-cost-app/
│   ├── app.py
│   ├── linear_regression_model.joblib
│   ├── painting_dataset_cleaned.csv
│   └── requirements.txt
├── MainCode.ipynb
├── Preprocess.ipynb
├── painting_dataset.csv
├── painting_dataset_cleaned.csv
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
├── groups_train.csv
├── groups_test.csv
├── linear_regression_model.joblib
├── linear_regression_metrics.csv
├── linear_regression_test_predictions.csv
├── linear_regression_energy_summary.csv
├── decision_tree_model.joblib
├── decision_tree_metrics.csv
├── decision_tree_test_predictions.csv
├── decision_tree_energy_summary.csv
├── energy_measurements.csv
└── README.md
```

The `painting-cost-app/` directory contains the files used by the deployed Streamlit application. The report, presentation, notebooks, source datasets, model evaluation files, and plots can remain in the same GitHub repository. Streamlit runs only the app file configured as the deployment entry point.

## Dataset and Features

### Source files

- `Dataset/boq_line_items.csv`: BOQ line-item records.
- `Dataset/boq_projects.csv`: project-level information.
- `Dataset/data_dictionary_line_items.csv`: data dictionary for line-item fields.
- `Dataset/data_dictionary_projects.csv`: data dictionary for project fields.

`Preprocess.ipynb` merges line items and project information using `project_id`, filters records whose `sub_category` is `Painting`, and saves `painting_dataset.csv`. It then cleans the data and saves `painting_dataset_cleaned.csv`.

### Target variable

- `amount_inr`: painting line-item cost in Indian rupees (INR); the regression target.

### Input features

| Feature | Description |
|---|---|
| `item_name` | Name or description of the BOQ item |
| `quantity` | Quantity recorded for the item |
| `building_type` | Type of building |
| `city` | Project city |
| `state` | Project state |
| `project_year` | Year extracted from `project_start_date` |
| `project_month` | Month extracted from `project_start_date` |

The modelling workflow derives `project_year` and `project_month` from `project_start_date`, then drops the original date column. The project identifier is used to group the split and validation folds; it is not used as a predictive feature.

## Workflow

### 1. Data preprocessing

Open and run `Preprocess.ipynb` from top to bottom. The notebook:

1. Loads the BOQ line-item and project CSV files.
2. Merges them using `project_id`.
3. Keeps records whose sub-category is `Painting`.
4. Removes duplicate records.
5. Converts selected numeric fields to numeric types.
6. Removes records missing essential fields such as project ID, item name, quantity, unit, or target amount.
7. Removes records with negative quantity or painting cost.
8. Trims text fields and writes the outputs.

Main outputs:

- `painting_dataset.csv`
- `painting_dataset_cleaned.csv`

### 2. Train/test split

The notebook runs data preparation and split cell at the beginning of the `MainCode.ipynb`. The notebook uses `GroupShuffleSplit` with `test_size=0.30` and `random_state=42`. The split is grouped by `project_id` to ensure that projects are held out as complete groups, not line items from the same project being distributed across both sets.

The resulting files are:

- `X_train.csv` and `X_test.csv`: input features.
- `y_train.csv` and `y_test.csv`: target values.
- `groups_train.csv` and `groups_test.csv`: project IDs corresponding row-by-row to the feature files.

The recorded run contained 846 training samples and 353 test samples. Since splitting is group-based, the exact percentage of samples can differ slightly from 70:30.

### 3. Model training and evaluation

The modeling cells in `MainCode.ipynb` consist of Linear Regression and Decision Tree regression. Each model was implemented using a sklearn pipeline, which performs preprocessing steps before fitting the regressor.

Moreover, the repo contains model, metrics, prediction, and energy-summary files for both Linear Regression and Decision Tree. We explored a Random Forest approach on a separate script; include its modeling cells and outputs if you would like to report it as a fully implemented alternative.

## Machine Learning Methodology

### Preprocessing pipeline

The model pipelines use a ColumnTransformer:

- The numeric features are imputed using the median.

- The categorical features are imputed using the most frequent value.

- The categorical values are then encoded using OneHotEncoder(handle_unknown="ignore").

The preprocessing is done inside the pipeline, so that the imputers and encoders are fit on each fold of the training data, and not on the whole training set at once.

### Project-aware validation
The workflow uses `GroupKFold` with up to five folds to evaluate the ability of the model to generalize to projects not seen during the fitting of that particular fold. The groups are defined by the project ID. The final test set is separated by project and should not be touched during model selection, but only used for reporting the final test metric.

### Regression models

- **Linear Regression:** a linear baseline model.
- **Decision Tree Regressor:** models nonlinear relationships through decision rules.

## Evaluation Metrics

The project evaluates regression performance using:

### Mean Absolute Error (MAE)

It is the average of the absolute differences between actual and predicted values. It represents the average absolute difference between the actual and predicted values. Thus, a lower value of the MAE indicates that the model has a lower average absolute difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

It is the square root of the average squared difference between actual and predicted values. It represents the average magnitude of the error. However, it penalizes the large errors more. Therefore, a lower RMSE suggests a better model.

### Coefficient of Determination (R2)

It measures the performance of the model based on how well it predicts the variance of the target variable. A higher value of this metric between 0 and 1 indicates a better fit. A negative value of this metric indicates that the model performs worse than predicting the mean value of the target variable on the test set.

The value of MAE and RMSE is expressed in INR, in case a plot represents the conversion in lakh INR.

`1 lakh INR = 100,000 INR`

## Energy Measurement

The CodeCarbon is applied to estimate the energy consumption for the selected stages of operations. The code evaluates the run time of the execution and reports CodeCarbon's energy estimate that is further converted from kWh to Joules using the following equation

Energy (J) = Energy (kWh) × 3,600,000

The energy summary CSV contains such fields as Stage, Time_seconds, and Energy_joules.

Depending on the code of the model, the measured stages can include cross-validation predictions, final model training, test-set prediction, hyperparameter tuning, and ensemble training.

Interpretation: CodeCarbon is used to provide software-based energy estimates, which may not always correlate with the actual energy usage of a physical computer due to various factors. In particular, the energy consumption depends on background applications, additional hardware-specific software, computational features of the computer processors, and energy overhead due to the computational intensity of monitored operations and short run-time of the processes.

## Streamlit Application

### Live deployment

The application is deployed at:

**https://painting-cost-prediction.streamlit.app/**

### App files

The `painting-cost-app/` folder contains:

- `app.py`: Streamlit user interface and prediction code.
- `linear_regression_model.joblib`: saved Linear Regression pipeline used for predictions.
- `painting_dataset_cleaned.csv`: dataset used to populate the app's dropdown options and quantity range.
- `requirements.txt`: Python packages required by the app.

### Run the app locally

Make sure Python is installed. Open a terminal and change to the app directory:

```bash
cd painting-cost-app
```

Install the app dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

The terminal will show a local URL, usually `http://localhost:8501`. Open it in a browser.

### Deployment and updates

The app is hosted with Streamlit Community Cloud and connected to the GitHub repository. After editing tracked app files, commit and push the changes to the connected deployment branch. Streamlit Community Cloud will detect the commit and update the deployment.

Example commands, run from the repository root:

```bash
git add painting-cost-app/app.py
git commit -m "Update Streamlit application"
git push
```

If you change `requirements.txt`, the deployment may reinstall dependencies. If you update the model or dataset, commit and push those files too.

### Prediction inputs

The app accepts the painting item, building type, quantity, state, city, and project start date. The date is transformed into `project_year` and `project_month` to match the features expected by the saved model pipeline.

The prediction is an estimate based on the saved model and the selected inputs; it should not be treated as a guaranteed quotation.

## Requirements

For the analysis notebooks, use Python 3.10 or newer where possible, with compatible versions of:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `joblib`
- `codecarbon`
- `jupyter`

Install the notebook dependencies with:

```bash
python -m pip install pandas numpy scikit-learn matplotlib scipy joblib codecarbon jupyter
```

The Streamlit application has its own `requirements.txt` inside `painting-cost-app/`. Keep the app dependencies listed there so the hosting service can install them during deployment. Use a scikit-learn version compatible with the version used to save the `.joblib` model.

## How to Run the Project

Run commands from the `ML_Project/` root directory so that relative CSV paths used in the notebooks resolve correctly.

1. **Prepare the data**
   - Confirm the four source CSV files are in `Dataset/`.
   - Open `Preprocess.ipynb`.
   - Run all cells from top to bottom.
   - Confirm `painting_dataset.csv` and `painting_dataset_cleaned.csv` are generated in the project root.

2. **Create the train/test split**
   - Open `MainCode.ipynb`.
   - Run the initial data preparation and split cell.
   - Confirm all six split files are created.

3. **Train and evaluate models**
   - Run the Linear Regression and Decision Tree sections in `MainCode.ipynb`.
   - Confirm model, metric, prediction, and energy-summary files are written.

4. **Generate plots**
   - Run the plotting cells in `MainCode.ipynb`, or open the MATLAB scripts in `MatlabPlotsAndScripts/`.
   - Ensure the relevant prediction and energy CSVs exist before running plots that load them.

5. **Run the web app**
   - Follow the [local Streamlit instructions](#run-the-app-locally), or open the [deployed app](https://painting-cost-prediction.streamlit.app/).

6. **Re-run after changing the dataset or code**
   - Re-run preprocessing first.
   - Recreate the split files.
   - Re-run model training and evaluation.
   - Regenerate figures so plots correspond to the current result files.
   - If the deployed app's model or dataset changes, update the corresponding files in `painting-cost-app/` and push the changes to GitHub.

## Plots and Outputs

### Dataset and split files

| File | Purpose |
|---|---|
| `painting_dataset.csv` | Merged Painting-category data before final cleaning |
| `painting_dataset_cleaned.csv` | Cleaned dataset used by the modelling workflow |
| `X_train.csv`, `X_test.csv` | Training and test features |
| `y_train.csv`, `y_test.csv` | Training and test targets |
| `groups_train.csv`, `groups_test.csv` | Project IDs aligned with each split |

### Model artifacts and results

| File | Purpose |
|---|---|
| `linear_regression_model.joblib` | Saved fitted Linear Regression pipeline |
| `linear_regression_metrics.csv` | Cross-validation and/or test metrics, depending on the run |
| `linear_regression_test_predictions.csv` | Actual and predicted test costs |
| `linear_regression_energy_summary.csv` | Stage-level energy and runtime summary |
| `decision_tree_model.joblib` | Saved fitted Decision Tree pipeline |
| `decision_tree_metrics.csv` | Decision Tree metrics |
| `decision_tree_test_predictions.csv` | Actual and predicted test costs |
| `decision_tree_energy_summary.csv` | Decision Tree stage-level energy and runtime |
| `energy_measurements.csv` | CodeCarbon measurement output |

### Figures

The `NormalPythonPlots/` folder contains PNG figures generated for analysis and reporting, such as:

- Painting-cost target distribution.
- Test MAE and RMSE comparison.
- Test R² comparison.
- Energy-consumption comparison.
- Actual versus predicted costs for Linear Regression.
- Residuals versus predicted costs for Linear Regression.

The `MatlabPlotsAndScripts/` folder contains MATLAB scripts and saved `.fig` and `.eps` figures for plotting and further editing.

### Interpreting plots

**Actual versus predicted costs:** Each point represents a held-out test observation. The diagonal line represents perfect predictions. Points closer to the line indicate predictions closer to actual costs.

**Residuals versus predicted costs:** Residuals are calculated as:

```text
Residual = Actual cost − Predicted cost
```

Positive residuals indicate underprediction; negative residuals indicate overprediction. A pattern in the residuals may indicate systematic prediction errors.

**Model comparison figures:**

- **MAE/RMSE:** lower values indicate smaller errors.
- **R²:** values closer to 1 generally indicate better fit on the same test set.
- **Energy:** lower energy indicates lower estimated energy use only when the measured stages and measurement conditions are comparable.