# %% [markdown]
#
# Uncertainty in evaluation metrics for classification
# ====================================================
#
# Has it ever happen to you that one of your colleagues claim their model with
# test score of 0.8001 is better than your model with test score of 0.7998?
# Maybe they are not aware that model-evaluation procedures should gauge not
# only the expected generalization performance, but also its variations. As
# usual, let's build a toy dataset to illustrate this.

# %%
from sklearn.datasets import make_classification

common_params = {
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_classes": 2,  # binary classification
    "random_state": 0,
    "weights": [0.55, 0.45],
}
X, y = make_classification(**common_params, n_samples=400)

prevalence = y.mean()
print(f"Percentage of samples in the positive class: {100*prevalence:.2f}%")

# %% [markdown]
# We are already familiar with using a a train-test split to estimate the
# generalization performance of a model. By default the `train_test_split` uses
# `shuffle=True`. Let's see what happens if we set a particular seed.

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
classifier = LogisticRegression().fit(X_train, y_train)
classifier.score(X_test, y_test)

# %% [markdown]
# Now let's see what happens when shuffling with a different seed:

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
classifier = LogisticRegression().fit(X_train, y_train)
classifier.score(X_test, y_test)

# %% [markdown]
# It seems that 42 is indeed the Ultimate answer to the Question of Life, the
# Universe, and Everything! Or maybe the score of a model depends on the split:
#  - the train-test proportion;
#  - the representativeness of the elements in each set.
#
# A more systematic way of evaluating the generalization performance of a model
# is through cross-validation, which consists of repeating the split such that
# the training and testing sets are different for each evaluation.

# %%
from sklearn.model_selection import cross_val_score

classifier = LogisticRegression()
scores = cross_val_score(classifier, X, y, cv=5)
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.2f} ± {scores.std():.2f}."
)

# %% [markdown]
# Scores have a variability. A sample probabilistic model gives the distribution
# of observed error: if the classification rate is p, the observed distribution
# of correct classifications on a set of size follows a binomial distribution.
# Let's create a function to easily visualize this:

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


def plot_error_distrib(classifier, X, y, cv=5):

    n = len(X)

    scores = cross_val_score(classifier, X, y, cv=cv)
    distrib = stats.binom(n=n, p=scores.mean())

    plt.plot(
        np.linspace(0, 1, n),
        n * distrib.pmf(np.arange(0, n)),
        linewidth=2,
        color="black",
        label="binomial distribution",
    )
    sns.histplot(scores, stat="density", label="empirical distribution")
    plt.title("Accuracy: " f"{scores.mean():.2f} ± {scores.std():.2f}.")
    plt.legend()
    plt.show()


plot_error_distrib(classifier, X, y, cv=5)

# %% [markdown]
# This still does not seem to be a binomial distribution. There are not enough
# evaluations, but the `KFold` cross-validation is intrinsically finite. Let's
# try another method for cross-validation:

# %%
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=200, test_size=0.2)
plot_error_distrib(classifier, X, y, cv=cv)

# %% [markdown]
# The empirical distribution is still broader than the theoretical one. This can
# be explained by the fact that as we are retraining the model on each fold, it
# actually fluctuates due the sampling noise in the training data, while the
# model above only accounts for sampling noise in the test data.
#
# The situation does get better with more data:

# %%
X, y = make_classification(**common_params, n_samples=1_000)
plot_error_distrib(classifier, X, y, cv=cv)

# %% [markdown]
# Importantly, the standard error of the mean (SEM) across folds is not a good
# measure of this error, as the different data folds are not independent. For
# instance, doing many random splits reduces the variance arbitrarily, but does
# not provide actually new data points.

# %%
cv = ShuffleSplit(n_splits=10, test_size=0.2)
X, y = make_classification(**common_params, n_samples=400)
scores = cross_val_score(classifier, X, y, cv=cv)

print(
    f"Mean accuracy ± SEM with n_split={cv.get_n_splits()}: "
    f"{scores.mean():.3f} ± {stats.sem(scores):.3f}."
)

cv = ShuffleSplit(n_splits=100, test_size=0.2)
scores = cross_val_score(classifier, X, y, cv=cv)

print(
    f"Mean accuracy ± SEM with n_split={cv.get_n_splits()}: "
    f"{scores.mean():.3f} ± {stats.sem(scores):.3f}."
)

cv = ShuffleSplit(n_splits=500, test_size=0.2)
scores = cross_val_score(classifier, X, y, cv=cv)

print(
    f"Mean accuracy ± SEM with n_split={cv.get_n_splits()}: "
    f"{scores.mean():.3f} ± {stats.sem(scores):.3f}."
)

# %% [markdown]
# Indeed, the SEM goes to zero as 1/sqrt{n_splits}. Wraping-up:
# - the more data the better;
# - the more splits, the more descriptive of the variance is the binomial
#   distribution, but keep in mind that more splits consume more computing
#   power;
# - use std instead of SEM to present your results.
#
# Now that we have an intuition on the variability of an evaluation metric, we
# are ready to apply it to our original Diabetes problem:

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

common_params = {
    "n_samples": 10_000,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_classes": 2,  # binary classification
    "shift": [4, 6],
    "scale": [10, 25],
    "random_state": 0,
}
X, y = make_classification(**common_params, weights=[0.55, 0.45])

X_train, X_plot, y_train, y_plot = train_test_split(
    X, y, stratify=y, test_size=0.1, random_state=0
)

estimator = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    estimator,
    X_plot,
    response_method="predict",
    alpha=0.5,
    xlabel="age (years)",
    ylabel="blood sugar level (mg/dL)",
    ax=ax,
)
scatter = disp.ax_.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolor="k")
disp.ax_.set_title(f"Diabetes test with prevalence = {y.mean():.2f}")
_ = disp.ax_.legend(*scatter.legend_elements())

# %% [markdown]
# Notice that the decision boundary changed with respect to the first notebook
# we explored. Let's make a remark: models depend on the prevalence of
# the data they were trained on. Therefore, all metrics (including likelihood ratios)
# depend on prevalence as much as the model depends on it. The difference is that
# likelihood ratios extrapolate through populations of different prevalence for
# a **fixed model**.
#
# Let's compute all the metrics and assez their variability in this case

# %%
from collections import defaultdict
from sklearn import metrics
import pandas as pd

cv = ShuffleSplit(n_splits=50, test_size=0.2)

evaluation = defaultdict(list)
scoring_strategies = [
    "accuracy",
    "balanced_accuracy",
    "recall",
    "precision",
    "matthews_corrcoef",
    # "positive_likelihood_ratio",
    # "neg_negative_likelihood_ratio",
]

for score_name in scoring_strategies:
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=score_name)
    evaluation[score_name] = scores

evaluation = pd.DataFrame(evaluation).aggregate(["mean", "std"]).T
evaluation["mean"].plot.barh(xerr=evaluation["std"]).set_xlabel("score")
plt.show()

# %% [markdown]
# Notice that `"positive_likelihood_ratio"` is not bounded from above and
# therefore it can't be directly compared with the other metrics on a single
# plot. Similarly, the `"neg_negative_likelihood_ratio"` has a reversed sign (is
# negative) to follow the scikit-learn convention for metrics for which a lower
# score is better.
#
# In this case we trained the model on nearly balanced classes. Try changing the
# prevalence and see how the variance of the metrics depend on data imbalance.
