# Breast Cancer Predictor

  - author: Tiffany Timbers, Melissa Lee & Joel Ostblom

Demo of a data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

## About

Here we attempt to build a classification model using the k-nearest neighbours algorithm which can use breast cancer tumour image measurements to predict whether a newly discovered breast cancer tumour is benign (i.e., is not harmful and does not require treatment) or malignant (i.e., is harmful and requires treatment intervention). Our final classifier performed well on an unseen test data set, with the F2 score, where beta = 2, of 0.97 and an overall accuracy calculated to be 0.87. On the 171 test data cases, it correctly predicted 157. Nine mistakes were predicting a benign tumour as malignant, while 4 mistakes where predicting a malignant tumour as benign. This is somewhat promising for implementing this in the clinic as false positives are less harmful than false negatives. Although they could theoretically cause the patient to undergo unnecessary treatment if the model is used as a decision tool, it is likely that the model is used for initial screening and that there will be a follow up appointment and further testing until treatment commences. However, the observation of even 4 mistakes predicting a malignant tumour as benign is concerning. As such, we believe further development of this model is needed for it to have clinical utility. Research to improve the model performance and understand the characteristics of incorrectly predicted patients is recommended.

The data set that was used in this project is of digitized breast cancer
image features created by Dr. William H. Wolberg, W. Nick Street, and
Olvi L. Mangasarian at the University of Wisconsin, Madison (Street,
Wolberg, and Mangasarian 1993). It was sourced from the UCI Machine
Learning Repository (Dua and Graff 2017) and can be found
[here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\(Diagnostic\)),
specifically [this
file](http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data).
Each row in the data set represents summary statistics from measurements
of an image of a tumour sample, including the diagnosis (benign or
malignant) and several other measurements (e.g., nucleus texture,
perimeter, area, etc.). Diagnosis for each image was conducted by
physicians.

## Report

The final report can be found
[here](https://ttimbers.github.io/breast-cancer-predictor/notebooks/breast_cancer_predictor_report.html).

## Dependencies
- [Docker](https://www.docker.com/) 
- [VS Code](https://code.visualstudio.com/download)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage

### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.

### Running the analysis

1. Navigate to the root of this project on your computer using the
   command line and enter the following command:

``` 
docker compose up
```

2. In the terminal, look for a URL that starts with 
`http://127.0.0.1:8888/lab?token=` 
(for an example, see the highlighted text in the terminal below). 
Copy and paste that URL into your browser.

<img src="img/jupyter-container-web-app-launch-url.png" width=400>

3. To run the analysis,
open a terminal and run the following commands:

```
python scripts/download_data.py \
    --url="https://archive.ics.uci.edu/static/public/15/breast+cancer+wisconsin+original.zip" \
    --write-to=data/raw

python scripts/split_n_preprocess.py \
    --raw-data=data/raw/wdbc.data \
    --data-to=data/processed \
    --preprocessor-to=results/models \
    --seed=522

python scripts/eda.py \
    --processed-training-data=data/processed/scaled_cancer_train.csv \
    --plot-to=results/figures

python scripts/fit_breast_cancer_classifier.py \
    --training-data=data/processed/cancer_train.csv \
    --preprocessor=results/models/cancer_preprocessor.pickle \
    --columns-to-drop=data/processed/columns_to_drop.csv \
    --pipeline-to=results/models \
    --plot-to=results/figures \
    --seed=523


python scripts/evaluate_breast_cancer_predictor.py \
	--scaled-test-data=data/processed/cancer_test.csv \
	--pipeline-from=results/models/cancer_pipeline.pickle \
	--results-to=results/tables \
	--seed=524

quarto render docs/breast_cancer_predictor_report.qmd --to html
quarto render docs/breast_cancer_predictor_report.qmd --to pdf
```

### Clean up

1. To shut down the container and clean up the resources, 
type `Cntrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

### Adding a new dependency

1. Add the dependency to the `environment.yml` file on a new branch.

2. Run `conda-lock -k explicit --file environment.yml -p linux-64` to update the `conda-linux-64.lock` file.

2. Re-build the Docker image locally to ensure it builds and runs properly.

3. Push the changes to GitHub. A new Docker
   image will be built and pushed to Docker Hub automatically.
   It will be tagged with the SHA for the commit that changed the file.

4. Update the `docker-compose.yml` file on your branch to use the new
   container image (make sure to update the tag specifically).

5. Send a pull request to merge the changes into the `main` branch. 

## License

The Breast Cancer Predictor report contained herein are licensed under the
[Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
See [the license file](LICENSE.md) for more information. . If
re-using/re-mixing please provide attribution and link to this webpage.
The software code contained within this repository is licensed under the
MIT license. See [the license file](LICENSE.md) for more information.

## References

Dua, Dheeru, and Casey Graff. 2017. “UCI Machine Learning Repository.”
University of California, Irvine, School of Information; Computer
Sciences. <http://archive.ics.uci.edu/ml>.

Street, W. Nick, W. H. Wolberg, and O. L. Mangasarian. 1993. “Nuclear
feature extraction for breast tumor diagnosis.” In *Biomedical Image
Processing and Biomedical Visualization*, edited by Raj S. Acharya and
Dmitry B. Goldgof, 1905:861–70. International Society for Optics;
Photonics; SPIE. <https://doi.org/10.1117/12.148698>.
