import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "class": ["Benign", "Malignant"],
        "mean_radius": [6.1, 7.8],
        "mean_area": [145.2, 156.1],
    })

# 
@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path

def test_write_csv_success(sample_dataframe, temp_directory):
    filename = "test_file.csv"
    write_csv(sample_dataframe, temp_directory, filename, index=False)
    
    # Check that the file exists
    file_path = os.path.join(temp_directory, filename)
    assert os.path.isfile(file_path)

    # Validate the contents of the CSV file
    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(sample_dataframe.reset_index(drop=True), loaded_df)

def test_write_csv_with_index(sample_dataframe, temp_directory):
    filename = "test_file_with_index.csv"
    write_csv(sample_dataframe, temp_directory, filename, index=True)
    
    # Check that the file exists
    file_path = os.path.join(temp_directory, filename)
    assert os.path.isfile(file_path)

    # Validate the contents of the CSV file, including the index
    loaded_df = pd.read_csv(file_path, index_col=0)
    pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

def test_write_csv_invalid_filename(sample_dataframe, temp_directory):
    invalid_filename = "test_file.txt"
    
    with pytest.raises(ValueError, match="Filename must end with '.csv'"):
        write_csv(sample_dataframe, temp_directory, invalid_filename, index=False)

def test_write_csv_nonexistent_directory(sample_dataframe):
    non_existent_directory = "/nonexistent_directory"
    filename = "test_file.csv"
    
    with pytest.raises(FileNotFoundError, match="Directory /nonexistent_directory does not exist."):
        write_csv(sample_dataframe, non_existent_directory, filename, index=False)

def test_write_csv_invalid_dataframe_type(temp_directory):
    invalid_dataframe = {}  # Not a pandas DataFrame
    
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        write_csv(invalid_dataframe, temp_directory, "test_file.csv", index=False)

def test_write_csv_empty_dataframe(sample_dataframe, temp_directory):
    empty_df = pd.DataFrame()  # Empty DataFrame
    
    with pytest.raises(ValueError, match="DataFrame must contain observations."):
        write_csv(empty_df, temp_directory, "test_file.csv", index=False)