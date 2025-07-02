# Movie Recommendation System

## Overview
This project implements a collaborative filtering movie recommendation system using cosine similarity. The system analyzes user-movie rating patterns to find similar users and recommend movies based on user preferences and similarities.

## Features
- **User-Based Collaborative Filtering**: Finds similar users based on rating patterns
- **Cosine Similarity**: Calculates similarity between users using cosine distance
- **Rating Matrix**: Uses user-movie rating matrix for analysis
- **Similarity Visualization**: Displays user similarity matrix

## Algorithm
The recommendation system uses **Collaborative Filtering** approach:
- Analyzes user ratings for different movies
- Calculates similarity between users using cosine similarity
- Recommends movies based on preferences of similar users

## Dependencies
Make sure you have the following packages installed:

```bash
pip install pandas scikit-learn numpy
```

Required libraries:
- `pandas`: For data manipulation and analysis
- `scikit-learn`: For cosine similarity calculations
- `numpy`: For numerical operations

## Usage

1. **Open the Notebook**:
   - Launch Jupyter Notebook or VS Code
   - Open `movieRecom.ipynb`

2. **Run the Code**:
   - Execute the cell to calculate user similarities
   - View the similarity matrix output

3. **Interpret Results**:
   - Higher similarity values (closer to 1) indicate similar users
   - Lower values (closer to 0) indicate dissimilar users

## Code Explanation

### Sample Data Creation
```python
data = {
    'Movie1': [5, 4, 0, 0],
    'Movie2': [4, 0, 0, 2],
    'Movie3': [0, 0, 5, 4],
    'Movie4': [0, 3, 4, 5]
}
df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4'])
```
- Creates a user-movie rating matrix
- Ratings from 0-5 (0 indicates no rating/not watched)
- 4 users and 4 movies for demonstration

### Similarity Calculation
```python
similarity = cosine_similarity(df)
```
- Calculates cosine similarity between all user pairs
- Returns a symmetric matrix with similarity scores

### Results Display
```python
print(pd.DataFrame(similarity, index=df.index, columns=df.index))
```
- Displays similarity matrix with user labels
- Values range from 0 (completely different) to 1 (identical preferences)

## Sample Data Structure

| User   | Movie1 | Movie2 | Movie3 | Movie4 |
|--------|--------|--------|--------|--------|
| User1  | 5      | 4      | 0      | 0      |
| User2  | 4      | 0      | 0      | 3      |
| User3  | 0      | 0      | 5      | 4      |
| User4  | 0      | 2      | 4      | 5      |

## Understanding Cosine Similarity
- **Range**: 0 to 1 (for non-negative ratings)
- **Interpretation**:
  - 1.0 = Identical preferences
  - 0.0 = No similarity
  - Higher values = More similar users

## Expected Output
The similarity matrix shows relationships like:
- Users with similar rating patterns have higher similarity scores
- Users who rated different sets of movies may have lower similarity
- Self-similarity (diagonal) is always 1.0

## Recommendation Logic
Based on similarity scores, the system can:
1. **Find Similar Users**: Identify users with highest similarity scores
2. **Recommend Movies**: Suggest movies highly rated by similar users
3. **Filter Recommendations**: Exclude movies already rated by the target user

## Extending the Project

### Add Recommendation Function
```python
def recommend_movies(user_index, similarity_matrix, ratings_df, n_recommendations=2):
    # Find most similar users
    similar_users = similarity_matrix[user_index].argsort()[-3:-1][::-1]
    # Get movies rated highly by similar users
    # Return top recommendations
```

### Real Dataset Integration
- Use MovieLens dataset
- Handle larger rating matrices
- Implement memory-based collaborative filtering

### Advanced Features
- Item-based collaborative filtering
- Matrix factorization techniques
- Hybrid recommendation systems
- Cold start problem solutions

## Limitations
- **Sparsity**: Real rating matrices are often sparse
- **Scalability**: Cosine similarity calculation can be expensive for large datasets
- **Cold Start**: Difficulty recommending for new users/items

## Learning Objectives
This project demonstrates:
- Collaborative filtering concepts
- Cosine similarity calculations
- Pandas DataFrame operations
- Recommendation system fundamentals
- User behavior analysis

## Real-World Applications
- Netflix movie recommendations
- Amazon product suggestions
- Spotify music recommendations
- YouTube video suggestions
- E-commerce personalization

## Next Steps
Consider exploring:
- Item-based collaborative filtering
- Matrix factorization (SVD, NMF)
- Deep learning approaches
- Evaluation metrics (RMSE, MAE)
- A/B testing for recommendations

## Author
**Pratul Kumar**

## License
This project is for educational purposes.
