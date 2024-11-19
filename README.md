# Breast Cancer Predictor

  - author: Tiffany Timbers, Melissa Lee & Joel Ostblom

Demo of a data analysis project for DSCI 522 (Data Science workflows); a
course in the Master of Data Science program at the University of
British Columbia.

## About

Here we attempt to build a classification model using the k-nearest 
neighbours algorithm which can use breast cancer tumour image 
measurements to predict whether a newly discovered breast cancer tumour 
is benign (i.e., is not harmful and does not require treatment) or 
malignant (i.e., is harmful and requires treatment intervention). 
Our final classifier performed fairly well on an unseen test data set, 
with Fbeta score, where beta = 2, of 0.98 
and an overall accuracy calculated to be 0.96. On the 171 test data cases, 
it correctly predicted 168. 
It incorrectly predicted 3 cases, 
however these were false positives - predicting that a tumour is malignant 
when in fact it is benign. 
These kind of incorrect predictions could cause the patient 
to undergo unnecessary treatment, 
and as such we recommend further research to improve the model 
before it is ready to be put into production in the clinic.


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
- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)
- [Docker](https://www.docker.com/) 
- [VS Code](https://code.visualstudio.com/download)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage

#### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.

#### Running the analysis

1. Open VS Code to the root of this project.

2. Open the `notebooks/breast_cancer_predict_report.ipynb` file.

3. Open a new terminal inside VS Code and type: 

```
docker compose up
```

4. In the terminal, look for a URL that starts with 
`http://127.0.0.1:8888/lab?token=` 
(for an example, see the highlighted text in the terminal below). 
Copy that URL to your clipboard.

5. Click on the kernel selector in the top right corner of the notebook, 
and then click "Select Another Kernel" > "Existing Jupyter Server". 
Paste the copied URL into the text box, press Enter twice,
and select "Python 3" as the kernel.

6. To run the analysis, click "Restart" and then "Run All" (at the top of the notebook).

#### Clean up

1. To shut down the container and clean up the resources, 
type `Cntrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

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
