# Experiment 5 — Decision Tree and Random Forest

## Objective
Understand how a Decision Tree partitions feature space, why it is prone to
overfitting, and how ensembling many trees into a Random Forest addresses that
weakness.

## Theoretical Background

### How a Decision Tree splits
A Decision Tree builds a set of nested if-then rules by recursively splitting the
data. At each node, it evaluates every possible split across every feature and
selects the one that most reduces impurity between the parent node and the
resulting children. Two common impurity measures are the **Gini Index**, the
probability of misclassifying a randomly chosen element if labeled according to
the node's class distribution, and **Entropy**, a measure of information-theoretic
disorder. Both typically produce similar trees in practice; Gini is marginally
faster to compute since it avoids a logarithm.

### Why trees overfit
An unconstrained tree can keep splitting until every leaf is pure, effectively
memorizing the training data, including its noise. This makes a single deep tree
a **high-variance** model: a small change in the training data can produce a
substantially different tree structure. Constraining `max_depth`,
`min_samples_split`, and `min_samples_leaf` limits how finely the tree can
subdivide the data, preventing it from creating leaves that represent only a
handful of specific training points.

### Bagging and Random Forest
Random Forest builds many decision trees, each trained on a different bootstrap
sample — a random sample of the same size as the original dataset, drawn with
replacement. This technique, called bagging (bootstrap aggregation), reduces
variance: because each tree sees a slightly different version of the data, their
individual errors are only weakly correlated, and averaging their predictions
(majority vote, for classification) cancels out much of that noise.

Random Forest adds a second source of randomness beyond bagging: at each split,
only a random subset of features is considered, rather than all of them. This
decorrelates the trees further — without it, a single very strong feature would
dominate the first split of nearly every tree regardless of the bootstrap sample,
undermining the variance-reduction benefit of averaging.

### Why averaging reduces variance
If individual models' errors are independent, or only weakly correlated,
averaging their predictions causes much of that error to cancel out
statistically, so the ensemble's prediction varies less across different
possible training sets than any single model's would — even though each
individual tree can remain deep and low-bias.

### Practical implications
Increasing `n_estimators` (the number of trees) generally stabilizes predictions
further, with diminishing returns rather than a risk of overfitting — unlike tree
depth, more trees essentially never make a Random Forest worse, only slower to
train. Random Forest's advantage over a single tree is most pronounced when the
single tree is genuinely high-variance; on very simple or very small, clean
datasets, a well-tuned single tree can sometimes match an ensemble's performance.

## Dataset
A binary classification dataset (Wisconsin Diagnostic Breast Cancer) with a
modest number of samples relative to features, a setting where a single
unconstrained tree is especially prone to overfitting.
