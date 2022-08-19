# EuroSciPy 2022 - Evaluating your ML models tutorial

Some intro [slides](https://docs.google.com/presentation/d/16AEtPVY4MVnBUsZ8CVPsnpvK5uo9HP3oJZ1Ds3AUBoA/edit?usp=sharing)

## Follow the tutorial online

- Launch an online notebook environment using: Binder.

You need an internet connection but you will not have to install any package
locally.

## Running the tutorial locally

### Dependencies

The tutorials will require the following packages:

* python
* jupyter
* pandas
* matplotlib
* scikit-learn ! Dev version

### Local install

We provide both `requirements.txt` and `environment.yml` to install packages.

You can install the packages using `pip`:

```
$ pip install -r requirements.txt
```

You can create an `sklearn-tutorial` conda environment executing:

```
$ conda env create -f environment.yml
```

and later activate the environment:

```
$ conda activate sklearn-tutorial
```

You might also only update your current environment using:

```
$ conda env update --prefix ./env --file environment.yml  --prune
```
