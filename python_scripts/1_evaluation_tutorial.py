# %% [markdown]
#
# Accounting for imbalance in evaluation metrics for classification 
# =================================================================
#
# Suppose we have a population of subjects with features `X` that can hopefully
# serve as indicators of a binary class `y` (known ground truth). Additionally,
# suppose the class prevalence (the number of samples in the positive class
# divided by the total number of samples) is very low.
#
# To fix ideas, let's use a medical analogy and think about diabetes. We only
# use two features -age and blood sugar level-, to keep the example as simple as
# possible. We use `make_classification` to simulate the distribution of the
# disease and to ensure **the data-generating process is always the same**. We
# set the `weights=[0.99, 0.01]` to obtain a prevalence of around 1% which,
# according to [The World
# Bank](https://data.worldbank.org/indicator/SH.STA.DIAB.ZS?most_recent_value_desc=false),
# is the case for the country with the lowest diabetes prevalence in 2022
# (Benin).
# 
# In practice, the ideas presented here can be applied in settings where the
# data available to learn and evaluate a classifier has nearly balanced classes,
# such as a case-control study, while the target application, i.e. the general
# population, has very low prevalence.

# %%
from sklearn.datasets import make_classification

common_params = {
    "n_samples": 10_000,
    "n_features": 2,
    "n_informative": 2,
    "n_redundant": 0,
    "n_classes": 2, # binary classification
    "shift": [4, 6],
    "scale": [10, 25],
    "random_state": 0,
}
X, y = make_classification(**common_params, weights=[0.99, 0.01])
prevalence = y.mean()
print(f"Percentage of people carrying the disease: {100*prevalence:.2f}%")

# %% [markdown]
# A simple model is trained to diagnose if a person is likely to have diabetes.
# To estimate the generalization performance of such model, we do a train-test
# split.

# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

estimator = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

# %% [markdown]
# We now show the decision boundary learned by the estimator. Notice that we
# only plot an stratified subset of the original data.

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    estimator,
    X_test,
    response_method="predict",
    alpha=0.5,
    xlabel="age (years)",
    ylabel="blood sugar level (mg/dL)",
    ax=ax,
)
scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
disp.ax_.set_title(f"Hypothetical diabetes test with prevalence = {y.mean():.2f}")
_ = disp.ax_.legend(*scatter.legend_elements())

# %% [markdown]
# The most widely used summary metric is arguably accuracy. Its main advantage
# is a natural interpretation: the proportion of correctly classified samples.

# %%
from sklearn import metrics

y_pred = estimator.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.3f}")

# %% [markdown]
# However, it is misleading when the data is imbalanced. Our model performs
# as well as a trivial majority classifier.

# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_dummy = estimator.predict(X_test)
accuracy_dummy = metrics.accuracy_score(y_test, y_dummy)
print(f"Accuracy if Diabetes did not exist: {accuracy_dummy:.3f}")

# %% [markdown]
# Some of the other metrics are better at describing the flaws of our model:

# %%
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
matthews = metrics.matthews_corrcoef(y_test, y_pred)
PPV = metrics.precision_score(y_test, y_pred)

print(f"Sensitivity on the test set: {sensitivity:.2f}")
print(f"Specificity on the test set: {specificity:.2f}")
print(f"Balanced accuracy on the test set: {balanced_acc:.2f}")
print(f"Matthews correlation coeff on the test set: {matthews:.2f}")
print()
print(f"Probability to have the disease given a positive test: {100*PPV:.2f}%")

# %% [markdown]
# Our classifier is not informative enough on the general population. The PPV
# and NPV give the information of interest: P(D+ | T+) and P(D− | T−). However,
# they are not intrinsic to the medical test (in other words the trained ML
# model) but also depend on the prevalence and thus on the target population.
#
# The class likelihood ratios (LR±) depend only on sensitivity and specificity
# of the classifier, and not on the prevalence of the study population. For the
# moment it suffice to recall that LR± is defined as
#
#     LR± = P(D± | T+) / P(D± | T−)

# %%
pos_LR, neg_LR = metrics.class_likelihood_ratios(y_test, y_pred)
print(f"LR+ on the test set: {pos_LR:.3f}") # higher is better
print(f"LR- on the test set: {neg_LR:.3f}") #  lower is better

# %% [markdown]
# <div class="admonition note alert alert-info">
# <p class="first admonition-title" style="font-weight: bold;">Caution</p>
# <p class="last">Please notice that if you want to use the
# `metrics.class_likelihood_ratios`, you require scikit-learn > v.1.2.0.
# </p>
# </div>
#
# Extrapolating between populations
# ---------------------------------
#
# The prevalence can be variable (for instance the prevalence of an infectious
# disease will be variable across time) and a given classifier may be intended
# to be applied in various situations.
#
# According to the World Bank, the diabetes prevalence in the French Polynesia
# in 2022 is above 25%. Let's now evaluate our previously trained model on a
# **different population** with such prevalence and **the same data-generating
# process**.

X, y = make_classification(**common_params, weights=[0.75, 0.25])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    estimator,
    X_test,
    response_method="predict",
    alpha=0.5,
    xlabel="age (years)",
    ylabel="blood sugar level (mg/dL)",
    ax=ax,
)
scatter = disp.ax_.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k")
disp.ax_.set_title(f"Hypothetical diabetes test with prevalence = {y.mean():.2f}")
_ = disp.ax_.legend(*scatter.legend_elements())

# %% 
# We then compute the same metrics using a test set with the new
# prevalence:

y_pred = estimator.predict(X_test)
prevalence = y.mean()
accuracy = metrics.accuracy_score(y_test, y_pred)
sensitivity = metrics.recall_score(y_test, y_pred)
specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
matthews = metrics.matthews_corrcoef(y_test, y_pred)
PPV = metrics.precision_score(y_test, y_pred)

print(f"Accuracy on the test set: {accuracy:.2f}")
print(f"Sensitivity on the test set: {sensitivity:.2f}")
print(f"Specificity on the test set: {specificity:.2f}")
print(f"Balanced accuracy on the test set: {balanced_acc:.2f}")
print(f"Matthews correlation coeff on the test set: {matthews:.2f}")
print()
print(f"Probability to have the disease given a positive test: {100*PPV:.2f}%")

# %% [markdown]
# The same model seems to perform better on this new dataset. Notice in
# particular that the probability to have the disease given a positive test
# increased. The same blood sugar test is less predictive in Benin than in
# the French Polynesia!
#
# If we really want to score the test and not the dataset, we need a metric that
# does not depend on the prevalence of the study population.

# %%
pos_LR, neg_LR = metrics.class_likelihood_ratios(y_test, y_pred)

print(f"LR+ on the test set: {pos_LR:.3f}")
print(f"LR- on the test set: {neg_LR:.3f}")

# %% [markdown]
# Despite some variations due to residual dataset dependence, the class
# likelihood ratios are mathematically invariant with respect to prevalence. See
# [this example from the User
# Guide](https://scikit-learn.org/dev/auto_examples/model_selection/plot_likelihood_ratios.html#invariance-with-respect-to-prevalence)
# for a demo regarding such property.
#
# Pre-test vs. post-test odds
# ---------------------------
#
# Both class likelihood ratios are interpretable in terms of odds:
#
#     post-test odds = Likelihood ratio * pre-test odds
#
# The interpretation of LR+ in this case reads:

# %%
print("The post-test odds that the condition is truly present given a positive "
     f"test result are: {pos_LR:.3f} times larger than the pre-test odds.")

# %% [markdown]
# We found that diagnosis tool is useful: the post-test odds are larger than the
# pre-test odds. We now choose the pre-test probability to be the prevalence of
# the disease in the held-out testing set.

# %%
pretest_odds = y_test.mean() / (1 - y_test.mean())
posttest_odds = pretest_odds * pos_LR

print(f"Observed pre-test odds: {pretest_odds:.3f}")
print(f"Estimated post-test odds using LR+: {posttest_odds:.3f}")

# %% [markdown]
# The post-test probability is the probability of an individual to truly have
# the condition given a positive test result, i.e. the number of true positives
# divided by the total number of samples. In real life applications this is
# unknown.

# %%
posttest_prob = posttest_odds / (1 + posttest_odds)

print(f"Estimated post-test probability using LR+: {posttest_prob:.3f}")

# %% [markdown]
# We can verify that if we had had access to the true labels, we would have
# obatined the same probabilities:

# %%
posttest_prob = y_test[y_pred == 1].mean()

print(f"Observed post-test probability: {posttest_prob:.3f}")

# %% [markdown]
# Conclusion: If a Benin salesperson was to sell the model to the French Polynesia
# by showing them the 59.84% probability to have the disease given a positive test,
# the French Polynesia would have never bought it, even though it would be quite
# predictive for their own population. The right thing to report are the LR±.
#
# Can you imagine what would happen if the model is trained on nearly balanced classes
# and then extrapolated to other scenarios?
