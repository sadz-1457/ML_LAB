# Experiment 1 — Exploratory Data Analysis (EDA)

## Objective
Build a systematic, reusable pipeline for understanding a dataset's structure,
quality, and relationships before any model is trained on it.

## Theoretical Background

### Why EDA comes first
Every downstream modeling decision — which features to use, which algorithm is
appropriate, how to handle missing data, whether a linear model's assumptions
hold — depends on properties of the data that are invisible until you actually
look. EDA is the disciplined process of surfacing those properties: shape of
distributions, presence of outliers, relationships between variables, and data
quality issues, all before committing to a model.

### Descriptive statistics
Mean, median, and mode describe central tendency, but disagreement between them
is itself informative: a large gap between mean and median signals skewness. The
standard deviation, along with the interquartile range (Q1 to Q3), describes
spread. **Skewness** measures asymmetry — a positive skew indicates a long right
tail (common in income or price data). **Kurtosis** measures tail-heaviness
relative to a normal distribution, independent of skew — high kurtosis means more
extreme outliers than a normal distribution would predict.

### Outlier detection via the IQR method
Tukey's rule defines outliers as any point below `Q1 - 1.5×IQR` or above
`Q3 + 1.5×IQR`, where IQR = Q3 - Q1. This method is distribution-agnostic — it
does not assume normality, unlike a z-score-based rule — which is why it is the
standard default for exploratory outlier flagging. Its main weakness is that it
breaks down on binary or near-constant columns, where Q1 and Q3 can both be zero,
causing any nonzero value to be flagged as an "outlier" even when it is a
perfectly normal, meaningful value.

### Correlation and multicollinearity
The Pearson correlation coefficient measures linear association between two
numeric variables, ranging from -1 to +1. Identifying pairs of highly correlated
features matters because multicollinearity — redundant, overlapping information
between predictors — destabilizes linear models' coefficient estimates, even
though it may not hurt the model's raw predictive accuracy.

### Mutual information and feature relevance
Mutual information measures how much knowing one variable reduces uncertainty
about another, without assuming any particular functional form (unlike
correlation, which only captures linear relationships). This makes it a more
general tool for ranking which features are likely to matter for predicting a
target, especially when the true relationship is nonlinear.

### Normality testing
The Shapiro-Wilk test formally checks whether a sample plausibly comes from a
normal distribution, producing a p-value: a low p-value (typically below 0.05)
means the data significantly deviates from normality. This matters because many
classical statistical methods and some models carry a normality assumption.

## Dataset
Applied generically across multiple datasets throughout the course (Iris, loan
approval data, breast cancer data, etc.) as a first step before every subsequent
experiment.
