# EuroSciPy 2022 - Evaluating your ML models tutorial

Some intro [slides](https://www.slideshare.net/GaelVaroquaux/evaluating-machine-learning-models-and-their-diagnostic-value)

## Follow the tutorial online

- Launch an online notebook environment using: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ArturoAmorQ/euscipy_2022/HEAD).

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
* scikit-learn ! **Dev version**

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
