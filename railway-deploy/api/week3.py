"""
FastAPI Backend for Week 3: Clustering - Netflix Recommendations
Implements KMeans, Agglomerative, and DBSCAN clustering on user-movie ratings data.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

router = APIRouter()

# CORS middleware
# CORS handled by main app

# Data URL
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/refs/heads/main/netflix_ratings.csv"

# Global cache for data and models
_cache: Dict[str, Any] = {}


def load_data() -> pd.DataFrame:
    """Load and cache the Netflix ratings data."""
    if "data" not in _cache:
        try:
            df = pd.read_csv(DATA_URL)
            _cache["data"] = df
            _cache["movie_columns"] = [col for col in df.columns if col.lower() not in ["user_id", "userid", "id", "user"]]

            # Identify user ID column
            user_col_candidates = ["user_id", "userid", "id", "user", "User_Id", "UserId"]
            _cache["user_col"] = None
            for col in user_col_candidates:
                if col in df.columns:
                    _cache["user_col"] = col
                    break
            if _cache["user_col"] is None:
                # Assume first column is user ID if not found
                _cache["user_col"] = df.columns[0]
                _cache["movie_columns"] = list(df.columns[1:])

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
    return _cache["data"]


def get_features() -> np.ndarray:
    """Get feature matrix (movie ratings) for clustering."""
    if "features" not in _cache:
        df = load_data()
        movie_cols = _cache["movie_columns"]
        features = df[movie_cols].values

        # Handle missing values by imputing with column means
        imputer = SimpleImputer(strategy="mean")
        features_imputed = imputer.fit_transform(features)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        _cache["features"] = features_scaled
        _cache["features_raw"] = features_imputed
        _cache["imputer"] = imputer
        _cache["scaler"] = scaler

    return _cache["features"]


# Request/Response Models
class ClusterRequest(BaseModel):
    algorithm: Literal["kmeans", "agglomerative", "dbscan"] = "kmeans"
    n_clusters: int = Field(default=5, ge=2, le=15)
    eps: float = Field(default=0.5, ge=0.1, le=2.0)
    min_samples: int = Field(default=5, ge=2, le=10)


class ClusterResponse(BaseModel):
    cluster_assignments: List[int]
    silhouette_score: Optional[float]
    inertia: Optional[float]
    n_clusters_found: int
    algorithm: str


class RecommendRequest(BaseModel):
    user_id: int
    movie: str
    n_neighbors: int = Field(default=10, ge=5, le=100)


class RecommendResponse(BaseModel):
    user_id: int
    movie: str
    predicted_rating: float
    similar_users: List[int]
    confidence: str


class ElbowResponse(BaseModel):
    k_values: List[int]
    inertia_values: List[float]
    silhouette_scores: List[float]


class ClusterProfileResponse(BaseModel):
    k: int
    cluster_profiles: Dict[str, Dict[str, float]]
    cluster_sizes: Dict[str, int]


# Endpoints
@router.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "Netflix Recommendations API",
        "version": "1.0.0",
        "week": 3,
        "topic": "Clustering"
    }


@router.get("/movies", response_model=List[str])
def get_movies():
    """List all movie columns in the dataset."""
    load_data()
    return _cache["movie_columns"]


@router.post("/cluster", response_model=ClusterResponse)
def run_clustering(request: ClusterRequest):
    """
    Run clustering with specified algorithm and parameters.

    - **algorithm**: kmeans, agglomerative, or dbscan
    - **n_clusters**: Number of clusters (2-15, used for kmeans and agglomerative)
    - **eps**: DBSCAN epsilon parameter (0.1-2.0)
    - **min_samples**: DBSCAN minimum samples parameter (2-10)
    """
    features = get_features()

    if request.algorithm == "kmeans":
        model = KMeans(
            n_clusters=request.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(features)
        inertia = float(model.inertia_)

    elif request.algorithm == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=request.n_clusters,
            linkage="ward"
        )
        labels = model.fit_predict(features)
        inertia = None

    elif request.algorithm == "dbscan":
        model = DBSCAN(
            eps=request.eps,
            min_samples=request.min_samples
        )
        labels = model.fit_predict(features)
        inertia = None
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

    # Calculate silhouette score (only if more than 1 cluster and less than n_samples clusters)
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

    sil_score = None
    if n_clusters_found > 1 and n_clusters_found < len(features):
        try:
            # For DBSCAN, exclude noise points (-1) from silhouette calculation
            if request.algorithm == "dbscan" and -1 in labels:
                mask = labels != -1
                if mask.sum() > n_clusters_found:
                    sil_score = float(silhouette_score(features[mask], labels[mask]))
            else:
                sil_score = float(silhouette_score(features, labels))
        except Exception:
            sil_score = None

    return ClusterResponse(
        cluster_assignments=[int(x) for x in labels],
        silhouette_score=sil_score,
        inertia=inertia,
        n_clusters_found=n_clusters_found,
        algorithm=request.algorithm
    )


@router.get("/elbow", response_model=ElbowResponse)
def get_elbow_data():
    """
    Get inertia and silhouette scores for k=2 to 15.
    Used to create elbow plots for optimal k selection.
    """
    features = get_features()

    k_values = list(range(2, 16))
    inertia_values = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        inertia_values.append(float(kmeans.inertia_))

        try:
            sil_score = float(silhouette_score(features, labels))
        except Exception:
            sil_score = 0.0
        silhouette_scores.append(sil_score)

    return ElbowResponse(
        k_values=k_values,
        inertia_values=inertia_values,
        silhouette_scores=silhouette_scores
    )


@router.post("/recommend", response_model=RecommendResponse)
def recommend_rating(request: RecommendRequest):
    """
    Predict a user's rating for a movie using KNN collaborative filtering.

    - **user_id**: The user ID to predict for
    - **movie**: The movie name to predict rating for
    - **n_neighbors**: Number of similar users to consider (5-100)
    """
    df = load_data()
    features = get_features()
    user_col = _cache["user_col"]
    movie_cols = _cache["movie_columns"]

    # Validate movie
    if request.movie not in movie_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Movie '{request.movie}' not found. Use /movies endpoint to see available movies."
        )

    # Find user index
    if request.user_id not in df[user_col].values:
        raise HTTPException(
            status_code=400,
            detail=f"User ID {request.user_id} not found in dataset."
        )

    user_idx = df[df[user_col] == request.user_id].index[0]

    # Fit KNN model
    n_neighbors = min(request.n_neighbors, len(features) - 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(features)

    # Find similar users
    distances, indices = knn.kneighbors([features[user_idx]])

    # Exclude the user themselves
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]

    # Get similar user IDs
    similar_user_ids = df.iloc[similar_indices][user_col].tolist()

    # Get raw (unscaled) ratings for the movie
    movie_idx = movie_cols.index(request.movie)
    features_raw = _cache["features_raw"]

    # Weighted average of similar users' ratings (weight = 1 - distance)
    weights = 1 - similar_distances
    weights = np.maximum(weights, 0.01)  # Avoid zero weights

    similar_ratings = features_raw[similar_indices, movie_idx]

    # Filter out NaN ratings
    valid_mask = ~np.isnan(similar_ratings)
    if valid_mask.sum() == 0:
        # If no valid ratings, use overall movie average
        predicted_rating = float(np.nanmean(features_raw[:, movie_idx]))
        confidence = "low"
    else:
        valid_ratings = similar_ratings[valid_mask]
        valid_weights = weights[valid_mask]
        predicted_rating = float(np.average(valid_ratings, weights=valid_weights))

        # Determine confidence based on number of valid neighbors
        if valid_mask.sum() >= n_neighbors * 0.7:
            confidence = "high"
        elif valid_mask.sum() >= n_neighbors * 0.3:
            confidence = "medium"
        else:
            confidence = "low"

    # Clip rating to reasonable range (1-5 for typical rating systems)
    predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

    return RecommendResponse(
        user_id=request.user_id,
        movie=request.movie,
        predicted_rating=round(predicted_rating, 2),
        similar_users=similar_user_ids[:10],  # Return top 10 similar users
        confidence=confidence
    )


@router.get("/cluster_profiles/{k}", response_model=ClusterProfileResponse)
def get_cluster_profiles(k: int):
    """
    Get top movies (highest average ratings) per cluster for a given k.

    - **k**: Number of clusters (2-15)
    """
    if k < 2 or k > 15:
        raise HTTPException(status_code=400, detail="k must be between 2 and 15")

    df = load_data()
    features = get_features()
    features_raw = _cache["features_raw"]
    movie_cols = _cache["movie_columns"]
    user_col = _cache["user_col"]

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Calculate cluster profiles
    cluster_profiles = {}
    cluster_sizes = {}

    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_sizes[f"cluster_{cluster_id}"] = int(cluster_mask.sum())

        if cluster_mask.sum() > 0:
            # Get average ratings for each movie in this cluster
            cluster_ratings = features_raw[cluster_mask]
            avg_ratings = np.nanmean(cluster_ratings, axis=0)

            # Get top 5 movies by average rating
            top_indices = np.argsort(avg_ratings)[::-1][:5]
            top_movies = {}
            for idx in top_indices:
                if not np.isnan(avg_ratings[idx]):
                    top_movies[movie_cols[idx]] = round(float(avg_ratings[idx]), 2)

            cluster_profiles[f"cluster_{cluster_id}"] = top_movies
        else:
            cluster_profiles[f"cluster_{cluster_id}"] = {}

    return ClusterProfileResponse(
        k=k,
        cluster_profiles=cluster_profiles,
        cluster_sizes=cluster_sizes
    )


@router.get("/users")
def get_users():
    """Get list of user IDs in the dataset."""
    df = load_data()
    user_col = _cache["user_col"]
    return {
        "user_column": user_col,
        "user_ids": df[user_col].tolist(),
        "total_users": len(df)
    }


@router.get("/data_info")
def get_data_info():
    """Get information about the loaded dataset."""
    df = load_data()
    user_col = _cache["user_col"]
    movie_cols = _cache["movie_columns"]

    return {
        "total_users": len(df),
        "total_movies": len(movie_cols),
        "user_column": user_col,
        "movie_columns": movie_cols,
        "sample_users": df[user_col].head(5).tolist(),
        "data_shape": list(df.shape)
    }
