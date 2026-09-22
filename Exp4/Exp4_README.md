# Experiment 4 — Logistic Regression and Support Vector Machines

## Objective
Study two linear-and-kernel-based approaches to binary classification, and
understand how each defines and optimizes its decision boundary.

## Theoretical Background

### Logistic Regression
Despite the name, Logistic Regression is a classification algorithm. It computes
a linear combination of the input features, then passes the result through the
sigmoid function to produce a value between 0 and 1, interpreted as the
probability of belonging to the positive class. A threshold, usually 0.5,
converts this probability into a discrete class label.

The model is trained by minimizing log-loss (cross-entropy), rather than squared
error. Log-loss heavily penalizes confident but wrong predictions, and it keeps
the optimization problem convex when combined with the sigmoid function — squared
error does not have either property in this setting.

**Regularization** in Logistic Regression works the same way as in linear
regression: an L2 penalty (Ridge-like) shrinks coefficients while retaining every
feature; an L1 penalty (Lasso-like) can drive some coefficients to exactly zero,
performing feature selection. The hyperparameter `C` controls regularization
strength, but inversely — a small `C` means strong regularization (a simpler
model), while a large `C` means weak regularization (a more complex, closely
fit model), because `C` multiplies the data-fitting term rather than the penalty.

### Support Vector Machines
An SVM finds the hyperplane that separates two classes while maximizing the
margin — the distance between the hyperplane and the nearest point of each
class. Only these nearest points, called support vectors, determine the
decision boundary; every other point could move freely without changing it.

The parameter `C` again controls a trade-off, this time between margin width and
misclassification tolerance: a small `C` favors a wider margin even if it
misclassifies some points; a large `C` prioritizes fitting the training data
closely, potentially at the cost of a narrower margin.

### The kernel trick
Real data is often not linearly separable in its original feature space. Rather
than explicitly transforming data into a higher-dimensional space where it might
become separable — which can be computationally expensive or even infinite-
dimensional — the kernel trick computes the equivalent of a dot product in that
higher-dimensional space directly, without ever constructing the transformed
coordinates. This makes non-linear decision boundaries computationally
practical. Common kernels include the linear kernel (for already-separable data),
polynomial kernel (captures interactions up to a chosen degree), RBF kernel (the
most flexible, based on distance-based similarity), and sigmoid kernel
(resembling a neural network activation function).

The `γ` (gamma) parameter, used in RBF, polynomial, and sigmoid kernels, controls
how far a single training point's influence extends: high `γ` means highly
localized influence (risking overfitting to individual points), low `γ` means
broader, smoother influence.

### Choosing between the two
Logistic Regression offers a simpler, faster-to-train, more directly
interpretable model — its coefficients can be read as the influence of each
feature on the log-odds of the outcome. SVMs, particularly with non-linear
kernels, can model more complex decision boundaries and often achieve higher
accuracy on data that is not linearly separable, at the cost of interpretability
and, for some kernel/parameter combinations, significantly higher computational
expense.

## Dataset
A binary classification dataset (spam/ham email classification) with numerical
features extracted from the underlying content.
