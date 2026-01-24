"""
Assignment 3: Clustering and Collaborative Filtering for Netflix Recommendations
FastAPI Backend
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()


# Global variables to store data
ratings_df = None
user_movie_matrix = None
movie_titles = None


# GitHub raw URL for data
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/netflix_ratings.csv"


def load_data():
    """Load and prepare Netflix ratings data"""
    global ratings_df, user_movie_matrix, movie_titles

    if ratings_df is not None:
        return

    print("Loading Netflix ratings data from GitHub...")
    try:
        ratings_df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"Error loading data from GitHub: {e}")
        raise

    print(f"Loaded {len(ratings_df)} ratings")
    print(f"Columns: {ratings_df.columns.tolist()}")

    # Create user-movie matrix
    # Assuming columns are: userId, movieId, rating, title (or similar)
    if 'title' in ratings_df.columns:
        movie_col = 'title'
    elif 'movieId' in ratings_df.columns:
        movie_col = 'movieId'
    else:
        movie_col = ratings_df.columns[1]

    if 'userId' in ratings_df.columns:
        user_col = 'userId'
    else:
        user_col = ratings_df.columns[0]

    if 'rating' in ratings_df.columns:
        rating_col = 'rating'
    else:
        rating_col = ratings_df.columns[2]

    # Create pivot table
    user_movie_matrix = ratings_df.pivot_table(
        index=user_col,
        columns=movie_col,
        values=rating_col,
        aggfunc='mean'
    ).fillna(0)

    movie_titles = user_movie_matrix.columns.tolist()
    print(f"User-movie matrix shape: {user_movie_matrix.shape}")


class ClusteringRequest(BaseModel):
    k: int


class CollaborativeFilteringRequest(BaseModel):
    n_neighbors: int


@router.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_data()


@router.get("/")
async def root():
    return FileResponse("index.html")


@router.get("/health")
async def health():
    return {"status": "healthy", "data_loaded": ratings_df is not None}


@router.post("/clustering")
async def run_clustering(request: ClusteringRequest):
    """Run K-Means clustering with specified k"""
    if request.k < 2 or request.k > 10:
        raise HTTPException(status_code=400, detail="k must be between 2 and 10")

    load_data()

    # Run K-Means
    kmeans = KMeans(n_clusters=request.k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(user_movie_matrix.values)

    # Calculate metrics
    inertia = float(kmeans.inertia_)
    silhouette = float(silhouette_score(user_movie_matrix.values, cluster_labels))

    # Cluster sizes
    cluster_sizes = {}
    for i in range(request.k):
        cluster_sizes[f"Cluster {i+1}"] = int(np.sum(cluster_labels == i))

    # Cluster profiles - top 5 movies per cluster
    cluster_profiles = {}
    for i in range(request.k):
        cluster_mask = cluster_labels == i
        cluster_users = user_movie_matrix.values[cluster_mask]

        # Calculate mean ratings for this cluster
        mean_ratings = np.mean(cluster_users, axis=0)

        # Get top 5 movies
        top_indices = np.argsort(mean_ratings)[::-1][:5]
        top_movies = []
        for idx in top_indices:
            movie_name = str(movie_titles[idx])
            avg_rating = float(mean_ratings[idx])
            top_movies.append({
                "movie": movie_name,
                "avg_rating": round(avg_rating, 2)
            })

        cluster_profiles[f"Cluster {i+1}"] = top_movies

    return {
        "k": request.k,
        "inertia": round(inertia, 2),
        "silhouette_score": round(silhouette, 4),
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": cluster_profiles
    }


@router.get("/elbow_analysis")
async def elbow_analysis():
    """Run elbow analysis for k=2 to 10"""
    load_data()

    results = {
        "k_values": [],
        "inertias": [],
        "silhouette_scores": []
    }

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(user_movie_matrix.values)

        results["k_values"].append(k)
        results["inertias"].append(round(float(kmeans.inertia_), 2))
        results["silhouette_scores"].append(
            round(float(silhouette_score(user_movie_matrix.values, cluster_labels)), 4)
        )

    return results


@router.post("/collaborative_filtering")
async def run_collaborative_filtering(request: CollaborativeFilteringRequest):
    """Run collaborative filtering with specified number of neighbors"""
    valid_neighbors = [5, 10, 20, 50, 100, 200]
    if request.n_neighbors not in valid_neighbors:
        raise HTTPException(
            status_code=400,
            detail=f"n_neighbors must be one of {valid_neighbors}"
        )

    load_data()

    # Create a copy of the matrix for train/test split
    matrix = user_movie_matrix.values.copy()
    n_users, n_movies = matrix.shape

    # Create mask for non-zero ratings
    nonzero_mask = matrix != 0
    nonzero_indices = np.argwhere(nonzero_mask)

    # Split into train/test (80/20)
    np.random.seed(42)
    n_test = int(len(nonzero_indices) * 0.2)
    test_indices = np.random.choice(len(nonzero_indices), n_test, replace=False)

    # Create train and test sets
    train_matrix = matrix.copy()
    test_ratings = []
    test_positions = []

    for idx in test_indices:
        i, j = nonzero_indices[idx]
        test_ratings.append(matrix[i, j])
        test_positions.append((i, j))
        train_matrix[i, j] = 0

    # Compute user similarity using cosine similarity
    user_similarity = cosine_similarity(train_matrix)
    np.fill_diagonal(user_similarity, 0)  # Don't consider self-similarity

    # Predict ratings for test set
    predictions = []
    actuals = []

    n_neighbors = min(request.n_neighbors, n_users - 1)

    for (i, j), actual in zip(test_positions, test_ratings):
        # Find k most similar users who rated this movie
        similarities = user_similarity[i].copy()

        # Only consider users who rated this movie in training data
        rated_mask = train_matrix[:, j] != 0
        similarities[~rated_mask] = 0

        # Get top k neighbors
        top_k_indices = np.argsort(similarities)[::-1][:n_neighbors]
        top_k_sims = similarities[top_k_indices]
        top_k_ratings = train_matrix[top_k_indices, j]

        # Weighted average prediction
        if np.sum(top_k_sims) > 0:
            pred = np.sum(top_k_sims * top_k_ratings) / np.sum(top_k_sims)
        else:
            # Fall back to global mean
            pred = np.mean(train_matrix[train_matrix != 0])

        predictions.append(pred)
        actuals.append(actual)

    # Calculate MAE
    mae = float(mean_absolute_error(actuals, predictions))

    # Sample points for scatter plot (max 200 points)
    sample_size = min(200, len(actuals))
    sample_indices = np.random.choice(len(actuals), sample_size, replace=False)

    scatter_data = {
        "actuals": [round(actuals[i], 2) for i in sample_indices],
        "predictions": [round(predictions[i], 2) for i in sample_indices]
    }

    return {
        "n_neighbors": request.n_neighbors,
        "mae": round(mae, 4),
        "n_test_samples": len(test_ratings),
        "scatter_data": scatter_data
    }


@router.get("/cf_analysis")
async def cf_analysis():
    """Run collaborative filtering analysis for all neighbor values"""
    load_data()

    neighbor_values = [5, 10, 20, 50, 100, 200]
    results = {
        "n_neighbors": [],
        "maes": []
    }

    # Create train/test split once
    matrix = user_movie_matrix.values.copy()
    n_users, n_movies = matrix.shape

    nonzero_mask = matrix != 0
    nonzero_indices = np.argwhere(nonzero_mask)

    np.random.seed(42)
    n_test = int(len(nonzero_indices) * 0.2)
    test_indices = np.random.choice(len(nonzero_indices), n_test, replace=False)

    train_matrix = matrix.copy()
    test_ratings = []
    test_positions = []

    for idx in test_indices:
        i, j = nonzero_indices[idx]
        test_ratings.append(matrix[i, j])
        test_positions.append((i, j))
        train_matrix[i, j] = 0

    user_similarity = cosine_similarity(train_matrix)
    np.fill_diagonal(user_similarity, 0)

    for n_neighbors in neighbor_values:
        predictions = []
        k = min(n_neighbors, n_users - 1)

        for (i, j), actual in zip(test_positions, test_ratings):
            similarities = user_similarity[i].copy()
            rated_mask = train_matrix[:, j] != 0
            similarities[~rated_mask] = 0

            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_k_sims = similarities[top_k_indices]
            top_k_ratings = train_matrix[top_k_indices, j]

            if np.sum(top_k_sims) > 0:
                pred = np.sum(top_k_sims * top_k_ratings) / np.sum(top_k_sims)
            else:
                pred = np.mean(train_matrix[train_matrix != 0])

            predictions.append(pred)

        mae = float(mean_absolute_error(test_ratings, predictions))
        results["n_neighbors"].append(n_neighbors)
        results["maes"].append(round(mae, 4))

    return results


# Serve static files


