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

# Setup list of invalid data cases 
invalid_data_cases = []

# Case: missing value in "class" column
case_missing_class = valid_data.copy()
case_missing_class["class"].iloc[0] = None
invalid_data_cases.append((case_missing_class, "Check absent or incorrect for missing/null 'class' value"))
    
# Generate 30 cases Case (one for each numeric column) where data is out of range (too large)
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_too_big = valid_data.copy()
    case_too_big[col] = case_too_big[col] + 10  # Adding an arbitrary value to make it out of range
    invalid_data_cases.append((case_too_big, f"Check absent or incorrect for numeric values in '{col}' being too large"))

# Generate 30 cases Case (one for each numeric column) where data is out of range (too small)
numeric_columns = valid_data.select_dtypes(include=np.number).columns
for col in numeric_columns:
    case_too_small = valid_data.copy()
    case_too_small[col] = case_too_small[col] - 10  # Adding an arbitrary value to make it out of range
    invalid_data_cases.append((case_too_small, f"Check absent or incorrect for numeric values in '{col}' being too small"))

# Parameterize test cases
@pytest.mark.parametrize("invalid_data, description", invalid_data_cases)
def test_valid_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(invalid_data)
