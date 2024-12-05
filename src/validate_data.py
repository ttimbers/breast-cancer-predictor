import pandas as pd
import pandera as pa


def validate_data(cancer_dataframe):
    """
    Validates the input cancer data in the form of a pandas DataFrame against a predefined schema,
    and returns the validated DataFrame.

    This function checks that the columns in the input DataFrame conform to the expected types and value ranges.
    It also ensures there are no duplicate rows and no entirely empty rows.

    Parameters
    ----------
    cancer_dataframe : pandas.DataFrame
        The DataFrame containing cancer-related data, which includes columns such as 'class', 'mean_radius', 
        'mean_texture', and other related measurements. The data is validated based on specific criteria for 
        each column.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame that conforms to the specified schema.

    Raises
    ------
    pandera.errors.SchemaError
        If the DataFrame does not conform to the specified schema (e.g., incorrect data types, out-of-range values,
        duplicate rows, or empty rows).
    
    Notes
    -----
    The following columns are validated:
        - 'class': Values must be either 'Benign' or 'Malignant'.
        - Measurement columns (e.g., 'mean_radius', 'mean_texture', etc.) must fall within specific ranges.
        - Additional checks ensure there are no duplicate or completely empty rows in the DataFrame.
    """
    if not isinstance(cancer_dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if cancer_dataframe.empty:
        raise ValueError("Dataframe must contain observations.")
    
    schema = pa.DataFrameSchema(
        {
            "class": pa.Column(str, pa.Check.isin(["Benign", "Malignant"]), nullable=False),
            "mean_radius": pa.Column(float, pa.Check.between(5, 45), nullable=True),
            "mean_texture": pa.Column(float, pa.Check.between(5, 50), nullable=True),
            "mean_perimeter": pa.Column(float, pa.Check.between(40, 260), nullable=True), 
            "mean_area": pa.Column(float, pa.Check.between(140, 4300), nullable=True),
            "mean_smoothness": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "mean_compactness": pa.Column(float, pa.Check.between(0, 2), nullable=True),
            "mean_concavity": pa.Column(float, pa.Check.between(0, 2), nullable=True),
            "mean_concave_points": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "mean_symmetry": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "mean_fractal_dimension": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_radius": pa.Column(float, pa.Check.between(0, 3), nullable=True),
            "se_texture": pa.Column(float, pa.Check.between(0, 5), nullable=True),
            "se_perimeter": pa.Column(float, pa.Check.between(0, 22), nullable=True), 
            "se_area": pa.Column(float, pa.Check.between(6, 550), nullable=True),
            "se_smoothness": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_compactness": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_concavity": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_concave_points": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_symmetry": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "se_fractal_dimension": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "max_radius": pa.Column(float, pa.Check.between(5, 40), nullable=True),
            "max_texture": pa.Column(float, pa.Check.between(5, 50), nullable=True),
            "max_perimeter": pa.Column(float, pa.Check.between(40, 260), nullable=True), 
            "max_area": pa.Column(float, pa.Check.between(140, 4300), nullable=True),
            "max_smoothness": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "max_compactness": pa.Column(float, pa.Check.between(0, 2), nullable=True),
            "max_concavity": pa.Column(float, pa.Check.between(0, 2), nullable=True),
            "max_concave_points": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "max_symmetry": pa.Column(float, pa.Check.between(0, 1), nullable=True),
            "max_fractal_dimension": pa.Column(float, pa.Check.between(0, 1), nullable=True)
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )

    schema.validate(cancer_dataframe, lazy=True)