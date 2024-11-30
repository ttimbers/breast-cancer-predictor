# eda.py
# author: Tiffany Timbers
# date: 2023-11-27

import click
import os
import altair as alt
import numpy as np
import pandas as pd

@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(processed_training_data, plot_to):
    '''Plots the densities of each feature in the processed training data
        by class and displays them as a grid of plots. Also saves the plot.'''

    scaled_cancer_train = pd.read_csv(processed_training_data)
    
    # melt for plotting correlation heat map
    correlation_matrix = scaled_cancer_train.drop(columns=['class']).corr()
    correlation_long = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # exploratory data analysis - visualize correlation heat map
    corr_plot = alt.Chart(correlation_long).mark_rect().encode(
        x='Feature 1:O',
        y='Feature 2:O',
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Feature 1', 'Feature 2', 'Correlation']
    ).properties(
        width=600,
        height=600,
        title="Correlation Heatmap"
    )

    corr_plot.save(os.path.join(plot_to, "correlation_heat_map.png"),
              scale_factor=2.0)
    
    # melt for plotting predictor distributions across classes by facets
    cancer_train_melted = scaled_cancer_train.melt(
        id_vars=['class'],
        var_name='predictor',
        value_name='value'
    )

    # make columns names nicer for plotting
    cancer_train_melted['predictor'] = cancer_train_melted['predictor'].str.replace('_',' ')

    # exploratory data analysis - visualize predictor distributions across classes
    dist_plot = alt.Chart(cancer_train_melted, width=150, height=100).transform_density(
        'value',
        groupby=['class', 'predictor']
    ).mark_area(opacity=0.7).encode(
        x="value:Q",
        y=alt.Y('density:Q').stack(False),
        color='class:N'
    ).facet(
        'predictor:N',
        columns=3
    ).resolve_scale(
        y='independent'
    )

    dist_plot.save(os.path.join(plot_to, "feature_densities_by_class.png"),
              scale_factor=2.0)

if __name__ == '__main__':
    main()