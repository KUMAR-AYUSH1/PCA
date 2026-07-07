# Dimensionality and Dataset Reduction using PCA and K-Means

This repository demonstrates techniques for optimizing datasets by reducing both features (columns) and samples (rows) to improve machine learning workflows.

## 🛠️ Feature Reduction: PCA & KNN Evaluation
We applied **Principal Component Analysis (PCA)** for dimensionality reduction (column reduction) and evaluated its impact using the **K-Nearest Neighbors (KNN)** algorithm. 

* Trained a KNN classifier on the original dataset.
* Trained a KNN classifier on the PCA-reduced dataset.
* Compared model performance metrics before and after feature reduction.

## 📉 Row Reduction: Representative Sampling via K-Means
Just like PCA is used for dimensionality reduction (reducing columns), you can use clustering to reduce the number of rows when dealing with massive datasets. This method compresses the dataset while preserving its overall distribution by selecting rows closest to the cluster centroids.

### Implementation

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Define the number of rows you want to keep
N_ROWS_TO_KEEP = 100 

# Fit K-Means to the dataframe
kmeans = KMeans(n_clusters=N_ROWS_TO_KEEP, random_state=42, n_init=10)
kmeans.fit(df)

# Extract the coordinates of the mathematical center of each cluster
centroids = kmeans.cluster_centers_

# Find the index of the real row in 'df' closest to each centroid
closest_row_indices, _ = pairwise_distances_argmin_min(centroids, df)

# Ensure unique rows are selected and create the reduced subset
unique_selected_indices = np.unique(closest_row_indices)
df_subset = df.iloc[unique_selected_indices].copy()
```
