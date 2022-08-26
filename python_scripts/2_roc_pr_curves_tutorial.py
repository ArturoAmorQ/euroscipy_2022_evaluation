# %% [markdown]
# Evaluation of non-thresholded prediction
# ========================================
#
# All statistics that we presented up to now rely on `.predict` which outputs
# the most likely label. We haven’t made use of the probability associated with
# this prediction, which gives the confidence of the classifier in this
# prediction. By default, the prediction of a classifier corresponds to a
# threshold of 0.5 probability in a binary classification problem. Let's build a
# toy dataset to illustrate this.

# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split

common_params = {
    "n_samples": 10_000,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_classes": 2,  # binary classification
    "class_sep": 0.5,
    "random_state": 0,
    }
X, y = datasets.make_classification(**common_params, weights=[0.6, 0.4])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, test_size=0.02
)

# %% [markdown]
# We can quickly check the predicted probabilities to belong to either class
# using a `LogisticRegression`. To ease the visualization we select a subset
# of `n_plot` samples.

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression

n_plot = 10
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

proba_predicted = pd.DataFrame(
    classifier.predict_proba(X_test), columns=classifier.classes_
).round(decimals=2)
proba_predicted[:n_plot]

# %% [markdown]
# Probabilites sum to 1. In the binary case it suffices to retain the
# probability of belonging to the positive class, here shown as an annotation in
# the `DecisionBoundaryDisplay`. Notice that setting
# `response_method="predict_proba"` shows the level curves of the 2D sigmoid
# (logistic curve).

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    classifier,
    X_test,
    response_method="predict_proba",
    alpha=0.5,
    ax=ax,
)
scatter = disp.ax_.scatter(
    X_test[:n_plot, 0], X_test[:n_plot, 1], c=y_test[:n_plot], edgecolor="k"
)
disp.ax_.legend(*scatter.legend_elements(), title="True class", loc="lower right")
for i, proba in enumerate(proba_predicted[:n_plot][1]):
    disp.ax_.annotate(proba, (X_test[i, 0], X_test[i, 1]), fontsize="large")
plt.xlim(-2.0, 2.0)
plt.ylim(-4.0, 4.0)
plt.title(
    "Probability of belonging to the positive class\n(default decision threshold)"
)
plt.show()

# %% [markdown]
# Evaluation of different probability thresholds
# ==============================================
#
# The default decision threshold (0.5) might not be the best threshold that
# leads to optimal generalization performance of our classifier. One can vary
# the decision threshold (and therefore the underlying prediction) and compute
# some evaluation metrics as presented earlier.
#
# Receiver Operating Characteristic curve
# ---------------------------------------
#
# One could be interested in the compromise between accurately discriminating
# both the positive class and the negative classes. The statistics used for this
# are sensitivity and specificity, which measure the proportion of correctly
# classified samples per class.
#
# Sensitivity and specificity are generally plotted as a curve called the
# Receiver Operating Characteristic (ROC) curve. Each point on the graph
# corresponds to a specific decision threshold. Below is such a curve:

# %%
from sklearn.metrics import RocCurveDisplay
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(X_train, y_train)

disp = RocCurveDisplay.from_estimator(
    classifier, X_test, y_test, name="LogisticRegression", color="tab:green"
)
disp = RocCurveDisplay.from_estimator(
    dummy_classifier,
    X_test,
    y_test,
    name="chance level",
    color="tab:red",
    ax=disp.ax_,
)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.title("ROC curve for LogisticRegression")
plt.show()

# %% [markdown]
# ROC curves typically feature true positive rate on the Y axis, and false
# positive rate on the X axis. This means that the top left corner of the plot
# is the "ideal" point - a false positive rate of zero, and a true positive rate
# of one. This is not very realistic, but it does mean that a larger area under
# the curve (AUC) is usually better.
#
# We can compute the area under the ROC curve (using `roc_auc_score`) to
# summarize the generalization performance of a model with a single number, or
# to compare several models across thresholds.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


classifiers = {
    "Hist Gradient Boosting": HistGradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=1),
    "Logistic Regression": LogisticRegression(),
    "Chance": DummyClassifier(strategy="most_frequent"),
}

fig = plt.figure()
ax = plt.axes([0.18, 0.15, 0.78, 0.78])

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    disp = RocCurveDisplay.from_estimator(
        clf, X_test, y_test, name=name, ax=ax
    )
plt.xlabel("False positive rate")
plt.ylabel("True positive rate                           ")
plt.text(
    0.098,
    0.575,
    "= sensitivity or recall",
    transform=fig.transFigure,
    size=7,
    rotation="vertical",
)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.title("ROC curves for several models")
plt.show()

# %% [markdown]
# It is important to notice that the lower bound of the ROC-AUC is 0.5,
# corresponding to chance level. Indeed, we show the generalization performance
# of a dummy classifier (the red line) to show that even the worst
# generalization performance obtained will be above this line.

# %% [markdown]
# Precision-Recall curves
# -----------------------
#
# As mentioned above, maximizing the ROC curve helps finding a compromise
# between accurately discriminating both the positive class and the negative
# classes. If the interest is to focus mainly on the positive class, the
# precision and recall metrics are more appropriated. Similarly to the ROC
# curve, each point in the Precision-Recall curve corresponds to a level of
# probability which we used as a decision threshold.

# %%
from sklearn.metrics import PrecisionRecallDisplay

fig = plt.figure()
ax = plt.axes([0.18, 0.15, 0.78, 0.78])

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    disp = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, name=name, ax=ax
    )
plt.xlabel("Recall                  ")
plt.text(0.56, 0.0485, "= TPR or sensitivity", transform=fig.transFigure, size=7)
plt.ylabel("Precision         ")
plt.text(0.1, 0.6, "= PPV", transform=fig.transFigure, size=7, rotation="vertical")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.title("Precision-recall curve for LogisticRegression")
plt.show()

# %% [markdown]
# A classifier with no false positives would have a precision of 1 for all
# recall values. In like manner to the ROC-AUC, the area under the curve can be
# used to characterize the curve in a single number and is named average
# precision (AP). With an ideal classifier, the average precision would be 1.
#
# In this case, notice that the AP of a `DummyClassifier`, used as baseline to
# define the chance level, coincides with the prevalence of the positive class.
# This is analogous to the downside of the accuracy score as shown in the first
# notebook.

# %%
prevalence = y.mean()
print(f"Prevalence of the positive class: {prevalence:.3f}")

# %% [markdown]
# Let's see the effect of adding umbalance between classes in our set of models:

# %%
X, y = datasets.make_classification(**common_params, weights=[0.83, 0.17])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, test_size=0.02
)

fig = plt.figure()
ax = plt.axes([0.18, 0.15, 0.78, 0.78])

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    disp = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, name=name, ax=ax
    )
plt.xlabel("Recall                  ")
plt.text(0.56, 0.0485, "= TPR or sensitivity", transform=fig.transFigure, size=7)
plt.ylabel("Precision         ")
plt.text(0.1, 0.6, "= PPV", transform=fig.transFigure, size=7, rotation="vertical")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc="upper right")
plt.title("Precision-recall curve for LogisticRegression")
plt.show()

# %% [markdown]
# The AP of all models decreased, including the baseline defined by the dummy
# classifier. Indeed, we confirm that AP does not account for prevalence.
#
# Conclusions
# ===========
#
# - Consider the prevalence in your target population. It may be that the
#   prevalence in your testing sample is not representative of that of the
#   target population. In that case, aside from LR+ and LR-, performance metrics
#   computed from the testing sample will not be representative of those in the
#   target population.
#
# - Never trust a single summary metric (accuracy, balanced accuracy, ROC-AUC,
#   etc.), but rather look at all the individual metrics. Understand the
#   implication of your choices to known the right tradeoff.
