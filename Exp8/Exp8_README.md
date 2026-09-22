# Experiment 8 — Clustering: K-Means, DBSCAN, and Hierarchical Clustering

## Objective
Understand three structurally different approaches to unsupervised clustering —
centroid-based, density-based, and hierarchical — and how to evaluate clustering
quality without relying on labeled data.

## Theoretical Background

### K-Means
K-Means partitions data into *k* clusters by minimizing the within-cluster sum
of squared distances to each cluster's centroid (Within-Cluster Sum of Squares,
or WCSS/inertia). It proceeds iteratively: initialize *k* centroids, assign each
point to its nearest centroid, recompute each centroid as the mean of its
assigned points, and repeat until assignments stop changing. K-Means implicitly
assumes clusters are roughly spherical and similarly sized, since it partitions
space based purely on distance to a single central point per cluster.

Choosing *k* is not determined by the algorithm itself. The **elbow method**
plots WCSS against *k*: WCSS always decreases as *k* increases, but the *rate* of
decrease typically slows sharply at some point — the "elbow" — beyond which
additional clusters yield diminishing returns. The **silhouette score**
complements this by measuring, for each point, how much closer it is to its own
cluster than to the next-nearest cluster, averaged across all points; higher
values indicate more clearly separated clusters.

### DBSCAN
DBSCAN defines clusters based on density rather than distance to a central
point. Given a neighborhood radius `ε` (epsilon) and a minimum point count
`minPts`, it classifies points as **core points** (with at least `minPts`
neighbors within `ε`), **border points** (within `ε` of a core point but not
core themselves), or **noise** (neither). Clusters are formed by connecting core
points that are density-reachable from one another. This gives DBSCAN two
properties K-Means lacks: it can discover clusters of arbitrary, non-spherical
shape, and it naturally identifies noise points rather than forcing every point
into some cluster.

DBSCAN's behavior is highly sensitive to `ε` and, importantly, to
dimensionality: in high-dimensional spaces, the distances between points tend to
concentrate — points that are genuinely close together can still appear
numerically far apart — a manifestation of the curse of dimensionality. A
principled way to choose `ε` is the **k-distance plot**: sorting every point's
distance to its k-th nearest neighbor and looking for the point where this
distance suddenly rises sharply.

### Hierarchical Agglomerative Clustering
Agglomerative clustering builds a hierarchy bottom-up: it starts with every point
as its own cluster and repeatedly merges the two closest clusters until only one
remains, recording the merge order and distances. This hierarchy is represented
as a **dendrogram**, a tree diagram where the height of each merge reflects the
distance between the clusters being joined; cutting the dendrogram at a chosen
height yields a specific number of clusters. The **linkage criterion** defines
how the distance between two clusters (as opposed to two points) is measured:
**single linkage** uses the closest pair of points between clusters (prone to
forming elongated, chained clusters); **complete linkage** uses the farthest
pair (tends toward compact, evenly-sized clusters); **average linkage** uses the
mean pairwise distance; **Ward's method** minimizes the increase in
within-cluster variance caused by each merge, tending to produce clusters
similar in spirit to K-Means.

### Internal versus external evaluation
Because clustering is unsupervised, its "correctness" cannot be measured against
ground truth in the way classification accuracy can. **Internal metrics**
evaluate cluster quality using only the data and the cluster assignments
themselves: the silhouette score (separation and cohesion), the Davies-Bouldin
Index (average similarity between each cluster and its most similar other
cluster, where lower is better), and the Calinski-Harabasz Index (ratio of
between-cluster to within-cluster dispersion, where higher is better). When
ground-truth labels happen to be available for validation purposes, **external
metrics** such as the Adjusted Rand Index and Normalized Mutual Information
measure how well the discovered clusters align with the true groupings, without
requiring cluster numbers to literally match label numbers.

### Why dimensionality reduction matters for cluster visualization
Clustering is often performed in a high-dimensional feature space, but clusters
cannot be visually inspected in more than two or three dimensions. PCA or t-SNE
are used purely to *project* the already-computed cluster assignments into two
dimensions for visualization — the projection does not change the clusters
themselves, it only provides a way to look at them.

## Dataset
A high-dimensional sensor-derived dataset (accelerometer and gyroscope features)
with a known ground-truth activity label per sample, used to evaluate clustering
quality both internally and against the true activity groupings.
