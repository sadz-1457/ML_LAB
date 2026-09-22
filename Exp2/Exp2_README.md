# Experiment 2 — Naive Bayes and K-Nearest Neighbors Classification

## Objective
Understand and compare two fundamentally different approaches to classification:
a probabilistic model built on Bayes' theorem, and a non-parametric, distance-based
method with no explicit training phase.

## Theoretical Background

### Bayes' theorem and Naive Bayes
Bayes' theorem relates the probability of a class given the observed features to
the probability of the features given the class:
`P(class | features) ∝ P(features | class) × P(class)`.
Computing `P(features | class)` directly is intractable for more than a few
features, because it would require modeling every possible interaction between
them. Naive Bayes makes this tractable with a simplifying assumption: every
feature is conditionally independent of every other feature, given the class.
This assumption is almost never exactly true in real data, which is why the
method is called "naive" — yet it often performs well regardless, because
classification only requires the *ranking* of class probabilities to be correct,
not their exact values.

Different Naive Bayes variants correspond to different assumptions about how
features are distributed within each class: **Gaussian** assumes continuous,
normally-distributed features; **Multinomial** assumes discrete counts (such as
word frequencies); **Bernoulli** assumes binary presence/absence features.

### K-Nearest Neighbors
KNN makes no assumption about the underlying data distribution at all. To
classify a new point, it finds the *k* closest training points (by a distance
metric, typically Euclidean or Manhattan) and assigns the majority class among
them. With distance-weighted voting, closer neighbors contribute more to the
decision than farther ones.

The choice of *k* controls a direct bias-variance trade-off: a small *k* (such as
1) fits very closely to the training data and is sensitive to noise (high
variance, low bias); a large *k* smooths the decision boundary but risks blending
across true class boundaries (higher bias, lower variance).

Because KNN's entire decision rule is distance-based, **feature scaling is
essential** — a feature measured on a larger numeric scale would dominate the
distance calculation regardless of its actual relevance, unless every feature is
first standardized to a comparable range.

### Efficient neighbor search
Finding the *k* nearest points by brute-force comparison is expensive at scale.
KD-Trees and Ball Trees are spatial data structures that partition the training
data so that most points can be ruled out without being individually compared —
KD-Trees use axis-aligned splits and work best in lower dimensions; Ball Trees use
hyperspherical partitions and remain effective in higher-dimensional or
non-uniformly distributed data. All three search strategies (brute force, KD-Tree,
Ball Tree) produce mathematically identical predictions — they differ only in
computational efficiency.

### Model evaluation beyond accuracy
A confusion matrix breaks predictions into true/false positives and negatives,
revealing *what kind* of errors a model makes rather than just how often it is
wrong. The ROC curve plots the true positive rate against the false positive rate
across all possible classification thresholds, and the area under it (AUC)
summarizes ranking quality as a single number. On imbalanced datasets, a
Precision-Recall curve is often more informative than ROC, since it focuses
specifically on performance on the minority (often more important) class.

### Hyperparameter search strategies
GridSearchCV exhaustively evaluates every combination in a defined hyperparameter
space using cross-validation, guaranteeing the best combination within that grid
at the cost of computation time. RandomizedSearchCV instead samples a fixed number
of combinations at random, trading a small chance of missing the true optimum for
substantially reduced computation on large search spaces.

## Dataset
A classification dataset with numeric and/or categorical features suitable for
comparing generative (Naive Bayes) and instance-based (KNN) learning approaches.
