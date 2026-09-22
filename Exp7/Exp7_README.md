# Experiment 7 — Effect of Principal Component Analysis on Classifier Performance

## Objective
Understand the mathematical basis of Principal Component Analysis as a
dimensionality reduction technique, and study how reducing feature space affects
different families of classifiers differently.

## Theoretical Background

### What PCA actually does
PCA re-expresses a dataset in a new coordinate system, where each new axis (a
**principal component**) is a linear combination of the original features,
chosen so that the first component captures the maximum possible variance in the
data, the second captures the maximum remaining variance subject to being
orthogonal (uncorrelated) to the first, and so on. Because real-world features
are often correlated with one another, much of a dataset's total variance can
typically be captured by far fewer components than the original number of
features — this is the basis for using PCA to compress a high-dimensional
dataset with limited loss of information.

### Eigenvalues and explained variance
Each principal component has an associated eigenvalue, representing the amount
of variance it captures. The proportion of total variance explained by a
component is its eigenvalue divided by the sum of all eigenvalues. Components
are chosen either by a fixed count, or, more commonly, by a cumulative variance
target (for example, retaining however many components are needed to preserve
95% of the total variance). The Kaiser criterion offers an alternative rule of
thumb — retain any component whose eigenvalue exceeds 1, meaning it explains at
least as much variance as a single original standardized variable would.

### Statistical justification for applying PCA
PCA is only worth applying if the original features are meaningfully correlated
with each other — if every feature were already independent, there would be no
redundancy to compress. This can be checked statistically before committing to
PCA: **Bartlett's Test of Sphericity** tests whether the correlation matrix
differs significantly from an identity matrix (uncorrelated variables); a
significant result supports using PCA. The **Kaiser-Meyer-Olkin (KMO)** measure
quantifies how much shared variance exists among variables relative to noise, on
a 0-to-1 scale, offering a second, complementary check of suitability.

### Why standardization must precede PCA
PCA identifies directions of maximum variance, which means features measured on
larger numeric scales would dominate the principal components purely due to
scale, not genuine importance, unless every feature is first standardized to
comparable variance.

### Reconstruction and information loss
Because the transformation to principal components is linear and invertible
(within the retained components), a reduced dataset can be projected back into
the original feature space to check how much information was discarded by
comparing the reconstruction to the original data — this offers a direct,
quantifiable measure of the trade-off between compression and fidelity.

### Why PCA affects different models unevenly
PCA's effect on downstream classifier performance is not uniform, because
different models are differently sensitive to feature correlation and
dimensionality. Distance-based and margin-based methods (KNN, SVM) are often
more sensitive to redundant or correlated dimensions diluting their distance or
margin calculations, and may benefit noticeably from PCA. Tree-based methods
split on one feature at a time and are comparatively robust to correlated
features already, so PCA may change their performance only marginally, or not
consistently in one direction. This is why comparing "with PCA" against "without
PCA" per model, rather than assuming a single universal effect, is the correct
experimental design.

### Testing whether an observed difference is statistically real
Because cross-validation only produces a handful of fold-level scores (for
example, 5 for 5-fold CV), a **paired t-test** or the non-parametric **Wilcoxon
signed-rank test** can determine whether an observed difference in average
accuracy between two settings is likely to be a real effect rather than random
fold-to-fold variation. These tests only address differences in the *mean*;
checking whether variance/stability across folds differs meaningfully requires a
separate test, such as **Levene's test**, since a model can have an unchanged
mean while becoming substantially more or less consistent across folds.

## Dataset
A moderate-dimensionality classification dataset, evaluated across a wide range
of classifier families (linear, distance-based, kernel-based, tree-based, and
ensemble models) to observe PCA's differing effects across model types.
