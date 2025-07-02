# Iris Classification Project

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier to classify iris flowers into their respective species. Using the famous Iris dataset, the project demonstrates fundamental machine learning concepts including data splitting, model training, and accuracy evaluation.

## Dataset
The project uses the built-in Iris dataset from scikit-learn, which contains:
- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width
- **3 target classes**: Setosa, Versicolor, and Virginica

## Features
- **Data Loading**: Uses scikit-learn's built-in Iris dataset
- **Data Splitting**: 80% training, 20% testing split
- **KNN Classification**: Implements K-Nearest Neighbors algorithm
- **Model Evaluation**: Calculates and displays accuracy score

## Dependencies
Make sure you have the following packages installed:

```bash
pip install scikit-learn numpy
```

Required libraries:
- `scikit-learn`: For machine learning algorithms and datasets
- `numpy`: For numerical operations (included with scikit-learn)

## Usage

1. **Open the Notebook**:
   - Launch Jupyter Notebook or VS Code
   - Open `iris.ipynb`

2. **Run the Code**:
   - Execute the cell to train the KNN classifier
   - The model will automatically load data, train, and evaluate

3. **View Results**:
   - The output displays the accuracy score of the classifier
   - Typically achieves 90-100% accuracy on the test set

## Code Explanation

### Data Loading and Preparation
```python
iris = load_iris()
X, y = iris.data, iris.target
```
- Loads the Iris dataset with features (X) and target labels (y)

### Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
- Splits data into training (80%) and testing (20%) sets
- Ensures model evaluation on unseen data

### Model Training
```python
model = KNeighborsClassifier()
model.fit(X_train, y_train)
```
- Creates a KNN classifier with default parameters (k=5)
- Trains the model on the training data

### Prediction and Evaluation
```python
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
```
- Makes predictions on test data
- Calculates and displays accuracy score

## Algorithm Details

### K-Nearest Neighbors (KNN)
- **Classification Method**: Instance-based learning
- **Default K Value**: 5 neighbors
- **Distance Metric**: Euclidean distance
- **Decision Rule**: Majority vote among k nearest neighbors

## Expected Results
- **Typical Accuracy**: 90-100%
- **Performance**: High accuracy due to well-separated classes in Iris dataset
- **Training Time**: Very fast (no complex training required)

## Customization Options

1. **Change K Value**:
   ```python
   model = KNeighborsClassifier(n_neighbors=3)  # Use 3 neighbors
   ```

2. **Different Train/Test Split**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70/30 split
   ```

3. **Add Cross-Validation**:
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5)
   ```

## Learning Objectives
This project demonstrates:
- Basic machine learning workflow
- Classification using KNN algorithm
- Train/test data splitting
- Model evaluation with accuracy metrics
- Working with real-world datasets

## Iris Dataset Classes
1. **Setosa**: Easily distinguishable species
2. **Versicolor**: Medium characteristics
3. **Virginica**: Largest petals and sepals

## Next Steps
Consider exploring:
- Different classification algorithms (SVM, Random Forest)
- Feature visualization and analysis
- Hyperparameter tuning for KNN
- Cross-validation techniques
- Confusion matrix analysis

## Author
**Pratul Kumar**

## License
This project is for educational purposes.
