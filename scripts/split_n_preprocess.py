# split_n_preprocess.py
# author: Tiffany Timbers
# date: 2023-11-27

import click
import os
import numpy as np
import pandas as pd
import pandera as pa
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector


@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(raw_data, data_to, preprocessor_to, seed):
    '''This script splits the raw data into train and test sets, 
    and then preprocesses the data to be used in exploratory data analysis.
    It also saves the preprocessor to be used in the model training script.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    colnames = [
        "id",
        "class",
        "mean_radius",
        "mean_texture",
        "mean_perimeter", 
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",
        "se_radius",
        "se_texture",
        "se_perimeter", 
        "se_area",
        "se_smoothness",
        "se_compactness",
        "se_concavity",
        "se_concave_points",
        "se_symmetry",
        "se_fractal_dimension",
        "max_radius",
        "max_texture",
        "max_perimeter", 
        "max_area",
        "max_smoothness",
        "max_compactness",
        "max_concavity",
        "max_concave_points",
        "max_symmetry",
        "max_fractal_dimension"
    ]

    cancer = pd.read_csv(raw_data, names=colnames, header=None).drop(columns=['id'])
    # re-label Class 'M' as 'Malignant', and Class 'B' as 'Benign'
    cancer['class'] = cancer['class'].replace({
        'M' : 'Malignant',
        'B' : 'Benign'
    })

    # validate data
    schema = pa.DataFrameSchema(
        {
            "class": pa.Column(str, pa.Check.isin(["Benign", "Malignant"])),
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
    schema.validate(cancer, lazy=True)
    
    # create the split
    cancer_train, cancer_test = train_test_split(
        cancer, train_size=0.70, stratify=cancer["class"]
    )

    cancer_train.to_csv(os.path.join(data_to, "cancer_train.csv"), index=False)
    cancer_test.to_csv(os.path.join(data_to, "cancer_test.csv"), index=False)

    cancer_preprocessor = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include='number')),
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    pickle.dump(cancer_preprocessor, open(os.path.join(preprocessor_to, "cancer_preprocessor.pickle"), "wb"))

    cancer_preprocessor.fit(cancer_train)
    scaled_cancer_train = cancer_preprocessor.transform(cancer_train)
    scaled_cancer_test = cancer_preprocessor.transform(cancer_test)

    scaled_cancer_train.to_csv(os.path.join(data_to, "scaled_cancer_train.csv"), index=False)
    scaled_cancer_test.to_csv(os.path.join(data_to, "scaled_cancer_test.csv"), index=False)

if __name__ == '__main__':
    main()
