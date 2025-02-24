import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, pairwise_distances, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

def get_paired_genre_ratings(ratings, movies, genre_pairs):
    paired_ratings = pd.DataFrame()
    column_names = []
    
    for genre1, genre2 in genre_pairs:
        paired_movies = movies[
            movies['genres'].str.contains(genre1, na=False) & 
            movies['genres'].str.contains(genre2, na=False)
        ]
        avg_paired_votes_per_user = ratings[ratings['movieId'].isin(paired_movies['movieId'])] \
            .groupby('userId')['rating'].mean().round(2)
        paired_ratings = pd.concat([paired_ratings, avg_paired_votes_per_user], axis=1)
        column_names.append(f'avg_{genre1.lower()}_{genre2.lower()}_rating')
    
    paired_ratings.columns = column_names
    return paired_ratings.dropna()

# Define specific genre pairs
specific_genre_pairs = [('Romance', 'Drama'), ('Action', 'Adventure'), ('Sci-Fi', 'Fantasy')]
paired_genre_ratings = get_paired_genre_ratings(ratings, movies, specific_genre_pairs)

# Drop NaN and create a copy to avoid chained assignment warnings
paired_dataset = paired_genre_ratings.copy()
X = paired_dataset.values

# Custom KMeans with Canberra Distance
def custom_kmeans_canberra(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        distances = pairwise_distances(X, centers, metric='canberra')
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(axis=0))
            else:
                new_centers.append(centers[i])  # Keep old center if cluster is empty
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    
    return labels, centers

# Determine optimal number of clusters using Calinski-Harabasz index
def optimal_k_calinski(X, min_k=2, max_k=10):
    if X.shape[0] < min_k:
        return min_k  # Return the minimum cluster count if data is too small
    best_k = min_k
    best_score = -np.inf
    for k in range(min_k, max_k + 1):
        labels, _ = custom_kmeans_canberra(X, n_clusters=k)
        score = calinski_harabasz_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

# Function to process and compare silhouette scores
def process_genre_pair(X, pair_name):
    if X.shape[0] == 0:
        print(f"{pair_name}: No valid data for clustering. Skipping.")
        return
    
    # Original K-Means (Euclidean Distance)
    kmeans_original = KMeans(n_clusters=3, random_state=42)
    predictions_original = kmeans_original.fit_predict(X)
    silhouette_original = silhouette_score(X, predictions_original, metric='euclidean')
    
    # Remove outliers using LOF
    lof = LocalOutlierFactor(metric='canberra')
    outlier_flags = lof.fit_predict(X)
    X_filtered = X[outlier_flags == 1]
    
    if X_filtered.shape[0] == 0:
        print(f"{pair_name}: All data points were marked as outliers. Skipping enhanced K-Means.")
        return
    
    # Determine optimal number of clusters
    optimal_k = optimal_k_calinski(X_filtered, min_k=2, max_k=10)
    
    # Enhanced K-Means (Canberra Distance)
    predictions_enhanced, _ = custom_kmeans_canberra(X_filtered, n_clusters=optimal_k)
    silhouette_enhanced = silhouette_score(X_filtered, predictions_enhanced, metric='canberra')
    
    print(f"{pair_name}:")
    print(f"  Silhouette Score for Original K-Means (Euclidean): {silhouette_original:.4f}")
    print(f"  Silhouette Score for Enhanced K-Means (Canberra, LOF, Calinski-Harabasz): {silhouette_enhanced:.4f}\n")

# Process each genre pair
for (genre1, genre2) in specific_genre_pairs:
    pair_name = f"{genre1} & {genre2}"
    genre_column = f'avg_{genre1.lower()}_{genre2.lower()}_rating'
    X_pair = paired_dataset[[genre_column]].dropna().values
    process_genre_pair(X_pair, pair_name)
