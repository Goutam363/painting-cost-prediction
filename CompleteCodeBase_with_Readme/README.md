# Building Painting Cost Estimation Using Machine Learning

A machine-learning project for estimating building painting costs from
Bill of Quantities (BOQ) records. The project combines data merging and
cleaning, project-aware train/test splitting, regression models,
evaluation metrics, energy-consumption estimates, and plots for
analysis.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Repository Structure](#repository-structure)
-   [Dataset and Features](#dataset-and-features)
-   [Workflow](#workflow)
-   [Machine Learning Methodology](#machine-learning-methodology)
-   [Evaluation Metrics](#evaluation-metrics)
-   [Energy Measurement](#energy-measurement)
-   [Generated Outputs](#generated-outputs)
-   [Requirements](#requirements)
-   [How to Run](#how-to-run)
-   [Plots](#plots)
-   [Reproducibility and Notes](#reproducibility-and-notes)

## Project Overview

Estimating painting costs is one part of building construction
estimation. This project uses historical BOQ line items and
project-level information to predict the painting cost recorded in
`amount_inr`.

The current workflow includes:

1.  Merging BOQ line-item data with project information.
2.  Filtering records to the `Painting` sub-category.
3.  Cleaning the data and preparing the target and input features.
4.  Splitting the data by project, so the same project does not appear
    in both the training and test sets.
5.  Training regression models and evaluating predictions.
6.  Estimating energy consumption for selected model operations with
    CodeCarbon.
7.  Creating plots for the dataset distribution, prediction errors,
    model comparisons, and energy use.

## Objectives

-   Prepare a usable dataset of building painting BOQ records.
-   Predict `amount_inr` from item and project attributes.
-   Compare regression models using MAE, RMSE, and R².
-   Use project-aware splitting and cross-validation to reduce leakage
    between records belonging to the same project.
-   Record execution time and estimated energy consumption for selected
    stages.
-   Produce figures suitable for analysis and reporting.

## Repository Structure

The project folder is organised approximately as follows:

``` text
ML_Project/
├── Dataset/
│   ├── boq_line_items.csv
│   ├── boq_projects.csv
│   ├── data_dictionary_line_items.csv
│   └── data_dictionary_projects.csv
│
├── MatlabPlotsAndScripts/
│   ├── ActualVsPredictedCost.m
│   ├── ResidualsVsPredictedPaintingCosts.m
│   ├── PaintingCostDistribution.m
│   ├── TestErrorComparisonOfRegressionModels.m
│   ├── ComparisonOfR2_Score.m
│   ├── EnergyConsumptionComparison.m
│   └── [associated .fig and .eps files]
│
├── NormalPythonPlots/
│   ├── target_distribution.png
│   ├── test_error_comparison.png
│   ├── test_r2_comparison.png
│   ├── energy_comparison.png
│   ├── linear_regression_actual_vs_predicted.png
│   └── linear_regression_residuals.png
│
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

The exact files in the working directory may vary depending on which
notebook cells and model scripts have been run. The `Dataset/` folder
contains the source CSV files; generated datasets, model files, metrics,
and plots are saved separately.

## Dataset and Features

### Source files

-   `Dataset/boq_line_items.csv`: BOQ line-item records.
-   `Dataset/boq_projects.csv`: project-level information.
-   `Dataset/data_dictionary_line_items.csv`: data dictionary for
    line-item fields.
-   `Dataset/data_dictionary_projects.csv`: data dictionary for project
    fields.

`Preprocess.ipynb` merges the line items and project information using
`project_id`, filters records whose `sub_category` is `Painting`, and
saves `painting_dataset.csv`. It then cleans the data and saves
`painting_dataset_cleaned.csv`.

### Target variable

  -----------------------------------------------------------------------
  Variable                            Meaning
  ----------------------------------- -----------------------------------
  `amount_inr`                        Painting line-item cost in Indian
                                      rupees (INR); the regression target

  -----------------------------------------------------------------------

### Input features

The modelling workflow uses these features:

  Feature           Description
  ----------------- -------------------------------------------
  `item_name`       Name or description of the BOQ item
  `quantity`        Quantity recorded for the item
  `building_type`   Type of building
  `city`            Project city
  `state`           Project state
  `project_year`    Year extracted from `project_start_date`
  `project_month`   Month extracted from `project_start_date`

The modelling notebook derives `project_year` and `project_month` from
`project_start_date`, then drops the original date column. The project
identifier is used to group the split and validation folds; it is not
used as a predictive feature.

## Workflow

### 1. Data preprocessing

Open and run `Preprocess.ipynb` from top to bottom.

The notebook:

1.  Loads the BOQ line-item and project CSV files.
2.  Merges them using `project_id`.
3.  Keeps records whose sub-category is `Painting`.
4.  Removes duplicate records.
5.  Converts selected numeric fields to numeric types.
6.  Removes records missing essential fields such as project ID, item
    name, quantity, unit, or target amount.
7.  Removes records with negative quantity or painting cost.
8.  Trims text fields and writes the outputs.

Main outputs:

-   `painting_dataset.csv`
-   `painting_dataset_cleaned.csv`

### 2. Train/test split

Run the data preparation and split cell at the start of
`MainCode.ipynb`.

The notebook uses `GroupShuffleSplit` with `test_size=0.30` and
`random_state=42`. The split is grouped by `project_id`, so projects are
held out as complete groups rather than distributing line items from the
same project across both sets.

The resulting files are:

-   `X_train.csv` and `X_test.csv`: input features.
-   `y_train.csv` and `y_test.csv`: target values.
-   `groups_train.csv` and `groups_test.csv`: project IDs corresponding
    row-by-row to the feature files.

The observed split in the recorded run was 846 training samples and 353
test samples. Since splitting is group-based, the exact percentage of
samples can differ slightly from 70:30.

### 3. Model training and evaluation

The modelling cells in `MainCode.ipynb` include Linear Regression and
Decision Tree regression. Each model uses a scikit-learn pipeline that
performs preprocessing before fitting the regressor.

The repository also contains saved model, metrics, prediction, and
energy-summary files for Linear Regression and Decision Tree. A Random
Forest workflow has been explored separately; make sure its current
script and outputs are included in the repository if you want the README
and comparison figures to describe all three models as fully implemented
in the submitted project.

## Machine Learning Methodology

### Preprocessing pipeline

The model pipelines use a `ColumnTransformer`:

-   Numeric features are imputed using the median.
-   Categorical features are imputed using the most frequent value.
-   Categorical values are encoded with
    `OneHotEncoder(handle_unknown="ignore")`.

Putting preprocessing inside the pipeline ensures that imputers and
encoders are fitted on each training fold rather than on the complete
dataset before validation.

### Project-aware validation

The workflow uses `GroupKFold` with up to five folds, grouped by project
ID. This helps evaluate how well the model generalises to projects not
used to fit that fold.

The final test set is separated by project and should remain untouched
during model selection. Use it for the final reported test metrics only.

### Regression models

-   **Linear Regression** --- a linear baseline model.
-   **Decision Tree Regressor** --- models nonlinear relationships
    through decision rules.
-   **Random Forest Regressor** --- an ensemble of decision trees; use
    the separately maintained Random Forest script and outputs when
    including this model in a comparison.

## Evaluation Metrics

The project evaluates regression performance using:

### Mean Absolute Error (MAE)

The average absolute difference between actual and predicted costs.
Lower MAE indicates smaller average absolute prediction errors.

### Root Mean Squared Error (RMSE)

The square root of the average squared prediction error. RMSE penalises
large errors more heavily than MAE. Lower RMSE is better.

### Coefficient of Determination (R²)

Measures how much variation in the target is explained by the
predictions relative to a constant-mean baseline. Values closer to 1
generally indicate better fit; a negative value can occur if predictions
perform worse than that baseline on the evaluated data.

MAE and RMSE are reported in INR unless a plot explicitly converts them
to lakh INR. The conversion is:

`1 lakh INR = 100,000 INR`


## Energy Measurement

CodeCarbon is used to estimate energy consumption for selected
operations. The code measures execution time and obtains CodeCarbon's
energy estimate, then converts the reported kWh value to joules:

`Energy (J) = Energy (kWh) × 3,600,000`

The energy summary CSVs use columns such as `Stage`, `Time_seconds`, and
`Energy_joules`.

Depending on the model script, measured stages can include:

-   Cross-validation predictions.
-   Final model training.
-   Test-set prediction.
-   For Random Forest, hyperparameter tuning and ensemble training may
    also be measured.

**Interpretation:** CodeCarbon provides software-based estimates, not
necessarily a direct measurement of the whole computer's wall power.
Background applications, hardware support, monitoring overhead, and
short execution times can slightly affect the estimates.

## Generated Outputs

### Dataset and split files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `painting_dataset.csv`              Merged Painting-category data
                                      before final cleaning

  `painting_dataset_cleaned.csv`      Cleaned dataset used by the
                                      modelling workflow

  `X_train.csv`, `X_test.csv`         Training and test features

  `y_train.csv`, `y_test.csv`         Training and test targets

  `groups_train.csv`,                 Project IDs aligned with each split
  `groups_test.csv`                   
  -----------------------------------------------------------------------

### Model artifacts and results

  ------------------------------------------------------------------------------
  File                                       Purpose
  ------------------------------------------ -----------------------------------
  `linear_regression_model.joblib`           Saved fitted Linear Regression
                                             pipeline

  `linear_regression_metrics.csv`            Cross-validation and/or test
                                             metrics, depending on the run

  `linear_regression_test_predictions.csv`   Actual and predicted test costs

  `linear_regression_energy_summary.csv`     Stage-level energy and runtime
                                             summary

  `decision_tree_model.joblib`               Saved fitted Decision Tree pipeline

  `decision_tree_metrics.csv`                Decision Tree metrics

  `decision_tree_test_predictions.csv`       Actual and predicted test costs

  `decision_tree_energy_summary.csv`         Decision Tree stage-level energy
                                             and runtime

  `energy_measurements.csv`                  CodeCarbon measurement output
  ------------------------------------------------------------------------------

### Figures

The `NormalPythonPlots/` folder contains PNG figures generated for use
in the report, such as:

-   Painting-cost target distribution.
-   Test MAE and RMSE comparison.
-   Test R² comparison.
-   Energy-consumption comparison.
-   Actual versus predicted costs for Linear Regression.
-   Residuals versus predicted costs for Linear Regression.

The `MatlabPlotsAndScripts/` folder contains MATLAB scripts and saved
`.fig` and `.eps` figures for plotting and further editing.

## Requirements

Use Python 3.10 or newer where possible, with compatible versions of the
following libraries:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `scipy`
-   `joblib`
-   `codecarbon`
-   `jupyter`

Install the dependencies in the Python environment used for the
notebooks:

``` bash
python -m pip install pandas numpy scikit-learn matplotlib scipy joblib codecarbon jupyter
```

If you use a virtual environment, activate it before installing packages
and running the notebooks.

## How to Run

Run commands from the `ML_Project/` root directory so that the relative
CSV paths used in the notebooks resolve correctly.

1.  **Prepare the data**
    -   Confirm the four source CSV files are in `Dataset/`.
    -   Open `Preprocess.ipynb`.
    -   Run all cells from top to bottom.
    -   Confirm `painting_dataset.csv` and
        `painting_dataset_cleaned.csv` are generated in the project
        root.
2.  **Create the train/test split**
    -   Open `MainCode.ipynb`.
    -   Run the initial data preparation and split cell.
    -   Confirm all six split files are created.
3.  **Train and evaluate models**
    -   Run the Linear Regression and Decision Tree sections in
        `MainCode.ipynb`.
    -   Confirm model, metric, prediction, and energy-summary files are
        written.
4.  **Generate plots**
    -   Run the plotting cells in `MainCode.ipynb`, or open the MATLAB
        scripts in `MatlabPlotsAndScripts/`.
    -   Ensure the relevant prediction and energy CSVs exist before
        running plots that load them.
5.  **Re-run after changing the dataset or code**
    -   Re-run preprocessing first.
    -   Recreate the split files.
    -   Re-run model training and evaluation.
    -   Regenerate figures so that plots correspond to the current
        result files.

## Plots

### Target distribution

The target-distribution plot shows how painting costs are spread across
the observations. If a logarithmic transformation is used, the axis must
be labelled accordingly; a plot of raw costs should instead use a
raw-cost label.

### Actual versus predicted costs

Each point represents a held-out test observation. The diagonal line
represents perfect predictions. Points closer to the line indicate
predictions closer to actual costs.

### Residuals versus predicted costs

Residuals are calculated as:

`Residual = Actual cost − Predicted cost`

Positive residuals indicate underprediction; negative residuals indicate
overprediction. A pattern in the residuals may indicate systematic
prediction errors.

### Model comparison figures

-   **MAE/RMSE:** lower values indicate smaller errors.
-   **R²:** values closer to 1 generally indicate better fit on the same
    test set.
-   **Energy:** lower energy indicates lower estimated energy use only
    when the measured stages and measurement conditions are comparable.