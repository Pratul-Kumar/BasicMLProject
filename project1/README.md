# K-Means Clustering Project

## Overview
This project demonstrates the implementation of K-Means clustering algorithm using Python and scikit-learn. The project creates synthetic data and applies K-Means clustering to group data points into distinct clusters, visualizing the results with matplotlib.

## Features
- **Synthetic Data Generation**: Creates 300 sample data points using `make_blobs`
- **K-Means Clustering**: Implements K-Means algorithm with 3 clusters
- **Data Visualization**: Plots clustered data points with centroids
- **Clear Visualization**: Uses different colors for clusters and highlights centroids

## Dependencies
Make sure you have the following packages installed:

```bash
pip install numpy scikit-learn matplotlib
```

Or install them in your notebook environment:
- `numpy`: For numerical operations and array handling
- `scikit-learn`: For machine learning algorithms including K-Means
- `matplotlib`: For data visualization and plotting

## Project Structure
```
project1/
├── kmeans.ipynb    # Main Jupyter notebook with K-Means implementation
└── README.md       # This file
```

## Usage

1. **Open the Notebook**:
   - Launch Jupyter Notebook or VS Code
   - Open `kmeans.ipynb`

2. **Run the Code**:
   - Execute the first cell to import libraries and run the K-Means clustering
   - The code will automatically generate synthetic data and create clusters

3. **View Results**:
   - The output displays a scatter plot showing:
     - Data points colored by their assigned cluster
     - Red 'X' markers indicating cluster centroids
     - Clear separation between the three clusters

## Code Explanation

### Data Generation
```python
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
```
- Creates 300 synthetic data points
- Generates 3 natural clusters
- Uses standard deviation of 0.60 for cluster spread
- Fixed random state for reproducible results

### K-Means Implementation
```python
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
```
- Initializes K-Means with 3 clusters
- Fits the model to the generated data
- Finds optimal cluster centroids

### Visualization
- **Data Points**: Colored by cluster assignment using 'viridis' colormap
- **Centroids**: Marked with red 'X' symbols for easy identification
- **Labels**: Clear axis labels and title for better understanding

## Output
The visualization shows three distinct clusters:
- **Purple Cluster**: Bottom-left region
- **Yellow Cluster**: Top-right region  
- **Teal Cluster**: Bottom-right region

Each cluster is clearly separated with its centroid marked by a red 'X'.

## Parameters
- **Number of Clusters**: 3 (can be modified in the `n_clusters` parameter)
- **Data Points**: 300 (adjustable via `n_samples`)
- **Cluster Standard Deviation**: 0.60 (controls cluster tightness)
- **Random State**: 0 (ensures reproducible results)

## Customization
You can modify the following parameters to experiment with different scenarios:

1. **Change number of clusters**:
   ```python
   kmeans = KMeans(n_clusters=4, random_state=0)  # Try 4 clusters
   ```

2. **Adjust data points**:
   ```python
   X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=0)
   ```

3. **Modify cluster spread**:
   ```python
   X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
   ```

## Learning Objectives
This project helps understand:
- K-Means clustering algorithm fundamentals
- Data preprocessing and synthetic data generation
- Cluster visualization techniques
- Machine learning workflow with scikit-learn
- Data analysis and interpretation

## Next Steps
Consider exploring:
- Real-world datasets for clustering
- Elbow method for optimal cluster selection
- Different clustering algorithms (DBSCAN, Hierarchical)
- Advanced visualization techniques
- Cluster evaluation metrics

## Author
Created as part of Python machine learning exploration.

## License
This project is for educational purposes.
