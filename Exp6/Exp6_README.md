# Experiment 6 — Bagging, Boosting, and Stacked Ensembles

## Objective
Study three distinct ensemble learning strategies, understand which aspect of
model error each is designed to reduce, and analyze how combining models differs
fundamentally from combining independent copies of the same model.

## Theoretical Background

### Bagging (Bootstrap Aggregation)
Bagging trains multiple independent copies of the same base model, each on a
different bootstrap sample of the training data, and combines their predictions
by averaging or majority vote. Because the models are trained independently and
in parallel, their individual errors — fit to the noise or quirks of their
specific sample — tend to be only weakly correlated, so averaging cancels out
much of that noise. Bagging is therefore a variance-reduction technique, and it
is most beneficial when applied to base models that are already high-variance,
such as unconstrained decision trees. It does relatively little for high-bias
models, since averaging several similarly wrong predictions still yields a wrong
answer.

### Boosting
Boosting trains models **sequentially** rather than independently: each new
model is specifically trained to correct the errors made by the ensemble so far.
**AdaBoost** does this by increasing the weight of misclassified samples after
each round, forcing the next weak learner to focus on the harder cases.
**Gradient Boosting** generalizes this idea by having each new model fit the
residual errors (the gradient of the loss function) of the current ensemble
directly, which works with any differentiable loss function rather than only
reweighting. Boosting is a bias-reduction technique — it can turn a sequence of
weak learners (each barely better than random) into a strong overall model. The
`learning_rate` parameter scales down each model's individual contribution;
smaller values require more boosting rounds but typically generalize better than
a few large, aggressive correction steps. Pushed too far — too many rounds, too
high a learning rate — boosting can begin fitting noise in later rounds,
overfitting in a way bagging does not.

### Stacking
Stacking combines predictions from multiple **heterogeneous** models — different
model types, such as an SVM, a Naive Bayes classifier, and a Decision Tree —
rather than many copies of one type. Instead of a simple vote or average, a
separate meta-learner is trained on the base models' predictions, learning the
optimal way to combine them. This approach benefits specifically from diversity:
different model types make different kinds of errors based on their different
assumptions, so if their errors are largely uncorrelated, the meta-learner can
learn to rely on whichever base model tends to be correct in different regions
of the feature space. Using several models of the same type would give the
meta-learner redundant, correlated inputs with little extra information to
combine. To prevent the meta-learner from training on overly optimistic,
leaked predictions, the base models' predictions used for meta-training are
generated via internal cross-validation, so no base model ever predicts on data
it was trained on.

### Bagging versus Boosting versus Stacking, conceptually
Bagging reduces variance by averaging independent models. Boosting reduces bias
by sequentially and deliberately correcting mistakes. Stacking does neither
directly — it instead tries to extract complementary strengths from structurally
different models via a learned combination rule. None of the three guarantees
improvement over a single well-tuned model: bagging adds little if the base
model is already low-variance, boosting can overfit if pushed too far, and
stacking gains nothing if its base learners are not making meaningfully
different errors.

## Dataset
The same binary classification dataset used for the Decision Tree and Random
Forest experiment, allowing direct comparison of ensemble strategies against a
single tree and against each other.
