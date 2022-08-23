# %% [markdown]
# Evaluation of different probability thresholds
# ----------------------------------------------
#
# All statistics that we presented up to now rely on `.predict` which outputs
# the most likely label. We haven’t made use of the probability associated with
# this prediction, which gives the confidence of the classifier in this
# prediction. By default, the prediction of a classifier corresponds to a
# threshold of 0.5 probability in a binary classification problem. Let's build a
# toy dataset to illustrate this.

# %%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

common_params = {
    "n_samples": 10_000,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_classes": 2,  # binary classification
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

n_plot = 10
classifier = make_pipeline(StandardScaler(), LogisticRegression())
classifier.fit(X_train, y_train)

proba_predicted = pd.DataFrame(
    100 * classifier.predict_proba(X_test), columns=classifier.classes_
).round(decimals=1)
proba_predicted[:n_plot]

# %% [markdown]
# Probabilites sum to 100%. In the binary case it suffices to retain the
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
disp.ax_.set_title("This is a plot title")
disp.ax_.legend(*scatter.legend_elements())
for i, proba in enumerate(proba_predicted[:n_plot][1]):
    disp.ax_.annotate(proba, (X_test[i, 0], X_test[i, 1]), fontsize="large")

# %% [markdown]
# The default decision threshold (0.5) might not be the best threshold that
# leads to optimal generalization performance of our classifier. One can vary
# the decision threshold, and therefore the underlying prediction, and compute
# the same evaluation metrics presented earlier. Usually, the recall and
# precision are computed and plotted on a graph. Each point on the graph
# corresponds to a specific decision threshold. Let’s start by computing the
# precision-recall curve.

# %%
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier

dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(X_train, y_train)

disp = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test, name="LogisticRegression", marker="+"
)
disp = PrecisionRecallDisplay.from_estimator(
    dummy_classifier,
    X_test,
    y_test,
    name="chance level",
    color="tab:orange",
    linestyle="--",
    ax=disp.ax_,
)

plt.legend(loc="lower left")
_ = disp.ax_.set_title("Precision-recall curve")

# %% [markdown]
# Notice that the `DummyClassifier` is a way to define chance level, and
# its average precision coincides with the prevalence of the positive class:

# %%
print(f"Prevalence of the positive class: {y.mean():.2f}")

# %% [markdown]
# On this curve, each blue cross corresponds to a level of probability which we
# used as a decision threshold. We can see that, by varying this decision
# threshold, we get different precision vs. recall values.
#
# A perfect classifier would have a precision of 1 for all recall values. A
# metric characterizing the curve is linked to the area under the curve (AUC)
# and is named average precision (AP). With an ideal classifier, the average
# precision would be 1.
#
# The precision and recall metric focuses on the positive class. However, one
# might be interested in the compromise between accurately discriminating the
# positive class and accurately discriminating the negative classes. The
# statistics used for this are sensitivity and specificity. On the one hand,
# sensitivity is just another name for recall. On the other hand specificity
# measures the proportion of correctly classified samples in the negative class
# defined as: TN / (TN + FP). Similar to the precision-recall curve, sensitivity
# and specificity are generally plotted as a curve called the Receiver Operating
# Characteristic (ROC) curve. Below is such a curve:

# %%
from sklearn.metrics import RocCurveDisplay


disp = RocCurveDisplay.from_estimator(classifier, X_test, y_test, name="LogisticRegression", marker="+")
disp = RocCurveDisplay.from_estimator(
    dummy_classifier, X_test, y_test, name="chance level", color="tab:orange", linestyle="--", ax=disp.ax_
)
plt.legend()
_ = disp.ax_.set_title("ROC AUC curve for the ")

# %% [markdown]
# This curve was built using the same principle as the precision-recall curve:
# we vary the probability threshold for determining "hard" prediction and
# compute the metrics. As with the precision-recall curve, we can compute the
# area under the ROC (ROC-AUC) to characterize the generalization performance of
# our classifier. However, it is important to observe that the lower bound of
# the ROC-AUC is 0.5. Indeed, we show the generalization performance of a dummy
# classifier (the orange dashed line) to show that even the worst generalization
# performance obtained will be above this line.
#
# Let's do something more realistic: We load the [Forest covertypes
# dataset](https://scikit-learn.org/stable/datasets/real_world.html#covtype-dataset).
# The samples in this dataset correspond to 30×30m patches of forest in the US,
# collected for the task of predicting each patch’s cover type, i.e. the
# dominant species of tree. There are seven covertypes, making this a multiclass
# classification problem, but in this case we will only retain two of such
# classes.

# %%
X, y = datasets.fetch_covtype(return_X_y=True, as_frame=True)
mask = np.isin(y, [1, 2])  # Select two classes
X = X[mask]
y = y[mask]

# Downsample, to see sampling effects in curves
X_reserve, X, y_reserve, y = train_test_split(
    X, y, stratify=y, random_state=0, test_size=10_000
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, test_size=0.02
)

print(
    f"prevalence of the majority class: {max(y.value_counts())/y.value_counts().sum():.2f}"
)

# %% [markdown]
# In this case we also explore some other models to give a visual intuition of what
# is a good PR-AUC and ROC-AUC.

 # %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = {
    "Excellent": RandomForestClassifier(n_jobs=-1, random_state=1),
    "Good": make_pipeline(StandardScaler(), LogisticRegression()),
    "Poor": GaussianNB(var_smoothing=0.3),
    "Chance": DummyClassifier(strategy="most_frequent"),
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)


# %% [markdown]
# The precision recall curve can be plotted using the `.predict_proba` for linear
# models or the `.decision_function` in other cases.

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score

fig = plt.figure()
ax = plt.axes([0.18, 0.15, 0.78, 0.78])

for name, clf in classifiers.items():
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
    auc = average_precision_score(y_test, y_score, pos_label=clf.classes_[1])
    if name == "Chance":
        # Use a much bigger test set: we do not want to illustrate
        # sampling noise
        y_score = clf.predict_proba(X_reserve)[:, 1]
        y_score += 0.25 * np.random.random(size=y_score.shape)
        prec, recall, _ = precision_recall_curve(
            y_reserve, y_score, pos_label=clf.classes_[1]
        )
        auc = average_precision_score(y_reserve, y_score, pos_label=clf.classes_[1])
    plt.plot(recall, prec, label=f"{name}, AUC={auc:.2f}")

plt.xlabel("Recall                  ")
plt.text(0.56, 0.0485, "= TPR or sensitivity", transform=fig.transFigure, size=7)
plt.ylabel("Precision         ")
plt.text(0.1, 0.6, "= PPV", transform=fig.transFigure, size=7, rotation="vertical")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Precision-recall curves for Forest Covertypes dataset")
plt.legend(loc="best")
plt.show()

# %%
print(f"Notice that the class imbalance {np.mean(y_reserve - 1):.4f}\n"
      f"equals the chance AUC {auc:.4f}")

# %% [markdown]
# We can evaluate the ROC curves of the same set of models using the Forest
# Covertypes dataset

# %%
from sklearn.metrics import roc_curve, roc_auc_score

fig = plt.figure()
ax = plt.axes([0.18, 0.15, 0.78, 0.78])

for name, clf in classifiers.items():
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    auc = roc_auc_score(y_test, y_score)
    if name == "Chance":
        fpr = [0, 1]
        tpr = [0, 1]
        auc = 0.5
    plt.plot(fpr, tpr, label=f"{name}, AUC={auc:.2f}")


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
plt.title("ROC curves for Forest Covertypes dataset")
plt.legend(loc="best")
plt.show()
