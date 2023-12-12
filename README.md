# EuroSciPy 2022 - Evaluating your ML models tutorial

Follow the intro slides [here](https://github.com/ArturoAmorQ/euroscipy_2022_evaluation/blob/main/slides.pdf).

## Follow the tutorial online

Launch an online notebook environment using [![Binder](https://mybinder.org/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/ArturoAmorQ/euroscipy_2022_evaluation/HEAD)

- [1_evaluation_tutorial.ipynb](https://notebooks.gesis.org/binder/jupyter/user/arturoamorq-eur-2022_evaluation-0kay4vsg/lab/tree/notebooks/1_evaluation_tutorial.ipynb)
- [2_roc_pr_curves_tutorial.ipynb](https://notebooks.gesis.org/binder/jupyter/user/arturoamorq-eur-2022_evaluation-0kay4vsg/lab/tree/notebooks/2_roc_pr_curves_tutorial.ipynb)
- [3_uncertainty_in_metrics_tutorial.ipynb ](https://notebooks.gesis.org/binder/jupyter/user/arturoamorq-eur-2022_evaluation-0kay4vsg/lab/tree/notebooks/3_uncertainty_in_metrics_tutorial.ipynb)

You need an internet connection but you will not have to install any package
locally.

## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python
* jupyter
* pandas
* matplotlib
* seaborn
* scikit-learn >= 1.2.0

### Local install

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

You can create an `evaluation-tutorial` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate evaluation-tutorial
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```
