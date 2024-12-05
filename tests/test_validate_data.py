import pytest
import os
import numpy as np
import pandas as pd
import pandera as pa
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.validate_data import validate_data


# Test data setup
valid_data = pd.DataFrame({
    "class" : ["Benign", "Malignant", "Benign"],
    "mean_radius": [np.nan, 6, 44.9],
    "mean_texture": [6.0, np.nan, 49.9],
    "mean_perimeter": [40.1, np.nan, 259.9], 
    "mean_area": [140.1, np.nan, 4299.9],
    "mean_smoothness": [0.1, np.nan, 0.9],
    "mean_compactness": [0.1, np.nan, 1.9],
    "mean_concavity": [0.1, np.nan, 1.9],
    "mean_concave_points": [0.1, np.nan, 0.9],
    "mean_symmetry": [0.1, np.nan, 0.9],
    "mean_fractal_dimension": [0.1, np.nan, 0.9],
    "se_radius": [0.1, np.nan, 2.9],
    "se_texture": [0.1, np.nan, 4.9],
    "se_perimeter": [0.1, np.nan, 21.9], 
    "se_area": [6.1, np.nan, 549.9],
    "se_smoothness": [0.1, np.nan, 0.9],
    "se_compactness": [0.1, np.nan, 0.9],
    "se_concavity": [0.1, np.nan, 0.9],
    "se_concave_points": [0.1, np.nan, 0.9],
    "se_symmetry": [0.1, np.nan, 0.9,],
    "se_fractal_dimension": [0.1, np.nan, 0.9,],
    "max_radius": [5.9, np.nan, 39.9],
    "max_texture": [5.9, np.nan, 49.9],
    "max_perimeter": [40.1, np.nan, 259.9],
    "max_area": [140.1, np.nan, 4299.9],
    "max_smoothness": [0.1, np.nan, 0.9],
    "max_compactness": [0.1, np.nan, 1.9],
    "max_concavity": [0.1, np.nan, 1.9],
    "max_concave_points": [0.1, np.nan, 0.9],
    "max_symmetry": [0.1, np.nan, 0.9],
    "max_fractal_dimension": [0.1, np.nan, 0.9],
})

# Case: wrong type passed to function
valid_data_as_np = valid_data.copy().to_numpy()
def test_valid_data_type():
    with pytest.raises(TypeError):
        validate_data(valid_data_as_np)

# Case: empty data frame
case_empty_data_frame = valid_data.copy().iloc[0:0]
def test_valid_data_empty_data_frame():
    with pytest.raises(ValueError):
        validate_data(case_empty_data_frame)

# Setup list of invalid data cases 
invalid_data_cases = []

# Case: missing value in "class" column
case_missing_class = valid_data.copy()
case_missing_class.loc[0, "class"] = None
invalid_data_cases.append((case_missing_class, "Check absent or incorrect for missing/null 'class' value"))

# Case: label in "class" column encoded as 0 and 1, instead of Benign and Malignant
case_wrong_label_type = valid_data.copy()
case_wrong_label_type["class"] = case_wrong_label_type["class"].map({'Benign': 0, 'Malignant': 1})
invalid_data_cases.append((case_missing_class, "Check incorrect type for'class' values is missing or incorrect"))

# Case: missing "class" column
case_missing_class_col = valid_data.copy()
case_missing_class_col = case_missing_class_col.drop("class", axis=1)  # drop class column
invalid_data_cases.append((case_missing_class_col, "`class` from DataFrameSchema"))

# Case: missing numeric columns (one for each numeric column) where column is missing
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_missing_col = valid_data.copy()
    case_missing_col = case_missing_col.drop(col, axis=1)  # drop column
    invalid_data_cases.append((case_missing_col, f"'{col}' is missing from DataFrameSchema"))
    
# Generate 30 cases (one for each numeric column) where data is out of range (too large)
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_too_big = valid_data.copy()
    case_too_big[col] = case_too_big[col] + 10  # Adding an arbitrary value to make it out of range
    invalid_data_cases.append((case_too_big, f"Check absent or incorrect for numeric values in '{col}' being too large"))

# Generate 30 cases (one for each numeric column) where data is out of range (too small)
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_too_small = valid_data.copy()
    case_too_small[col] = case_too_small[col] - 10  # Adding an arbitrary value to make it out of range
    invalid_data_cases.append((case_too_small, f"Check absent or incorrect for numeric values in '{col}' being too small"))

# Generate 30 cases (one for each numeric column) where data is the wrong type
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_wrong_type = valid_data.copy()
    case_wrong_type[col] = case_wrong_type[col].fillna(0.0).astype(int) # convert from float to int
    invalid_data_cases.append((case_wrong_type, f"Check incorrect type for float values in '{col}' is missing or incorrect"))

# Case: duplicate observations
case_duplicate = valid_data.copy()
case_duplicate = pd.concat([case_duplicate, case_duplicate.iloc[[0], :]], ignore_index=True)
invalid_data_cases.append((case_duplicate, f"Check absent or incorrect for duplicate rows"))

# Case: entire missing observation
case_missing_obs = valid_data.copy()
nan_row = pd.DataFrame([[np.nan] * (case_missing_obs.shape[1] - 1) + [np.nan]], columns=case_missing_obs.columns)
case_missing_obs = pd.concat([case_missing_obs, nan_row], ignore_index=True)
invalid_data_cases.append((case_missing_obs, f"Check absent or incorrect for missing observations (e.g., a row of all missing values)"))

# Parameterize invalid data test cases
@pytest.mark.parametrize("invalid_data, description", invalid_data_cases)
def test_valid_w_invalid_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(invalid_data)