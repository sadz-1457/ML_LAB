# Experiment 3 — Linear, Ridge, Lasso, and Elastic Net Regression

## Objective
Understand ordinary least squares regression and the theoretical motivation for
regularization, and compare how different penalty structures affect coefficient
behavior and generalization.

## Theoretical Background

### Ordinary Least Squares
Linear Regression finds the coefficients that minimize the sum of squared
residuals between predicted and actual values. Geometrically, this is the
hyperplane that is closest, in the least-squares sense, to every data point.
This solution has a closed form and is unbiased, but it can become highly
unstable when predictors are correlated with one another (multicollinearity) or
when the number of features is large relative to the number of samples — in
these situations, small changes in the data can cause large, erratic swings in
the estimated coefficients.

### Why regularization is needed
Regularization adds a penalty term to the loss function that discourages large
coefficient values. This introduces a small amount of bias into the model in
exchange for a often much larger reduction in variance — a direct application of
the bias-variance trade-off — typically improving how well the model generalizes
to unseen data, even though it no longer perfectly minimizes training error.

### Ridge Regression (L2 penalty)
Ridge adds the sum of squared coefficients to the loss function. Because this
penalty is smooth and has no corners, its effect is to shrink every coefficient
toward zero proportionally, but essentially never exactly to zero. Ridge is
particularly effective when many features are correlated, since it distributes
influence across them rather than favoring one arbitrarily.

### Lasso Regression (L1 penalty)
Lasso adds the sum of the *absolute values* of the coefficients. Geometrically,
this penalty region has corners aligned with the coordinate axes, and the
optimal solution frequently lands exactly on one of those corners — meaning some
coefficients are driven to exactly zero. This gives Lasso an automatic feature
selection property that Ridge lacks: it does not just shrink irrelevant features,
it can remove them from the model entirely.

### Elastic Net
Elastic Net combines both penalties, controlled by a mixing parameter
(`l1_ratio`). This addresses a specific weakness of pure Lasso: when several
features are highly correlated with each other, Lasso tends to arbitrarily select
only one of them and zero out the rest, which can be unstable. Elastic Net's
Ridge component encourages correlated features to be shrunk together rather than
one being arbitrarily favored.

### Regression evaluation metrics
Mean Absolute Error and Mean Squared Error both measure prediction error in the
target's own units, but MSE penalizes large errors disproportionately more due to
squaring. Root Mean Squared Error returns MSE to the original units while
retaining that same sensitivity to large errors. All three are *scale-dependent*
— their magnitude only means something relative to the scale of the target being
predicted. R², by contrast, is scale-free: it measures the proportion of the
target's variance explained by the model, making it the appropriate metric for
judging model quality independent of units.

### Diagnosing overfitting versus underfitting
Comparing training performance to cross-validation or test performance reveals
which regime a model is in: a large gap between the two (high training
performance, much lower validation performance) indicates overfitting — the model
has captured noise specific to the training set. Comparable training and
validation performance, even if both are only moderate, points instead toward
underfitting or an appropriately-fit model, not overfitting.

## Dataset
A regression dataset with a continuous numeric target and a mix of numeric and
categorical predictors.
