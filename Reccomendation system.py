import numpy as np
import pandas as pd
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Create a fictional dataset for movies
movies_data = {
    'movieId': [1, 2, 3, 4, 5],
    'title': ['3 Idiots', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy']
}

movies_df = pd.DataFrame(movies_data)

# Create a fictional dataset for user ratings
ratings_data = {
    'userId': [1, 1, 2, 2, 3],
    'movieId': [1, 2, 5, 3, 4],
    'rating': [5, 4, 3, 4, 5]
}

ratings_df = pd.DataFrame(ratings_data)

# Content-Based Recommendation

# Use TF-IDF vectorizer to convert the 'genre' column to numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf_vectorizer.fit_transform(movies_df['genre'])

# Calculate cosine similarity between movies based on genres
content_similarity = cosine_similarity(genre_matrix, genre_matrix)

# Collaborative Filtering

# Merge ratings and movies dataframes
movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')

# Create a user-item matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating', fill_value=0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(user_movie_ratings, test_size=0.2)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(train_data)


# Function to get collaborative filtering recommendations for a user
def get_collaborative_filtering_recommendations(user_id, top_n=2):
    user_ratings = user_movie_ratings.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_ratings, train_data.values)
    similar_users = train_data.index[np.argsort(similarity_scores[0])[::-1]][1:top_n + 1]
    recommendations = user_movie_ratings.loc[similar_users].mean().sort_values(ascending=False).index
    return recommendations


# Example: Get content-based recommendations for a user
user_id = int(input("user id : "))
content_based_recommendations = get_collaborative_filtering_recommendations(user_id)
print(f"Collaborative Filtering Recommendations for user {user_id}:")
print(content_based_recommendations)