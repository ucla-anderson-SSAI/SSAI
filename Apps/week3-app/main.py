import os
"""
FastAPI Backend for Week 3: Clustering - Netflix Recommendations
Implements KMeans, Agglomerative, and DBSCAN clustering on user-movie ratings data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

app = FastAPI(
    title="Netflix Recommendations API",
    description="Week 3 Clustering: User-based movie recommendations using clustering and KNN",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data URL
DATA_URL = "https://raw.githubusercontent.com/ucla-anderson-SSAI/SSAI/main/netflix_ratings.csv"

# Global cache for data and models
_cache: Dict[str, Any] = {}


def load_data() -> pd.DataFrame:
    """Load and cache the Netflix ratings data. Handles long-format CSV (userId, title, rating)."""
    if "data" not in _cache:
        try:
            # Try local file first, then URL
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netflix_ratings.csv")
            if os.path.exists(csv_path):
                df_raw = pd.read_csv(csv_path)
            else:
                df_raw = pd.read_csv(DATA_URL)

            _cache["raw"] = df_raw

            # The CSV is long-format: userId, title, rating — pivot to wide format
            pivot = df_raw.pivot_table(index="userId", columns="title", values="rating")
            _cache["data"] = pivot.reset_index()
            _cache["movie_columns"] = sorted(pivot.columns.tolist())
            _cache["user_col"] = "userId"
            _cache["pivot"] = pivot

            # Build feature matrix
            imputer = SimpleImputer(strategy="mean")
            features_imputed = imputer.fit_transform(pivot.values)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_imputed)

            _cache["features"] = features_scaled
            _cache["features_raw"] = features_imputed
            _cache["imputer"] = imputer
            _cache["scaler"] = scaler

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")
    return _cache["data"]


def get_features() -> np.ndarray:
    """Get feature matrix (movie ratings) for clustering."""
    load_data()
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
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "Netflix Recommendations API",
        "version": "1.0.0",
        "week": 3,
        "topic": "Clustering"
    }


@app.get("/movies", response_model=List[str])
def get_movies():
    """List all movie columns in the dataset."""
    load_data()
    return _cache["movie_columns"]


@app.post("/cluster", response_model=ClusterResponse)
def run_clustering(request: ClusterRequest):
    """Run clustering with specified algorithm and parameters."""
    features = get_features()

    if request.algorithm == "kmeans":
        model = KMeans(n_clusters=request.n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        inertia = float(model.inertia_)
    elif request.algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=request.n_clusters, linkage="ward")
        labels = model.fit_predict(features)
        inertia = None
    elif request.algorithm == "dbscan":
        model = DBSCAN(eps=request.eps, min_samples=request.min_samples)
        labels = model.fit_predict(features)
        inertia = None
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")

    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    sil_score = None
    if n_clusters_found > 1 and n_clusters_found < len(features):
        try:
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


@app.get("/elbow", response_model=ElbowResponse)
def get_elbow_data():
    """Get inertia and silhouette scores for k=2 to 15."""
    features = get_features()

    k_values = list(range(2, 16))
    inertia_values = []
    silhouette_scores_list = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        inertia_values.append(float(kmeans.inertia_))
        try:
            sil_score = float(silhouette_score(features, labels))
        except Exception:
            sil_score = 0.0
        silhouette_scores_list.append(sil_score)

    return ElbowResponse(
        k_values=k_values,
        inertia_values=inertia_values,
        silhouette_scores=silhouette_scores_list
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_rating(request: RecommendRequest):
    """Predict a user's rating for a movie using KNN collaborative filtering."""
    load_data()
    features = _cache["features"]
    pivot = _cache["pivot"]
    movie_cols = _cache["movie_columns"]

    if request.movie not in movie_cols:
        raise HTTPException(status_code=400, detail=f"Movie '{request.movie}' not found.")

    user_ids = pivot.index.tolist()
    if request.user_id not in user_ids:
        raise HTTPException(status_code=400, detail=f"User ID {request.user_id} not found.")

    user_idx = user_ids.index(request.user_id)
    movie_idx = movie_cols.index(request.movie)

    n_neighbors = min(request.n_neighbors, len(features) - 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    knn.fit(features)

    distances, indices = knn.kneighbors([features[user_idx]])
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    similar_user_ids = [int(user_ids[i]) for i in similar_indices]

    features_raw = _cache["features_raw"]
    weights = np.maximum(1 - similar_distances, 0.01)
    similar_ratings = features_raw[similar_indices, movie_idx]
    valid_mask = ~np.isnan(similar_ratings)

    if valid_mask.sum() == 0:
        predicted_rating = float(np.nanmean(features_raw[:, movie_idx]))
        confidence = "low"
    else:
        predicted_rating = float(np.average(similar_ratings[valid_mask], weights=weights[valid_mask]))
        if valid_mask.sum() >= n_neighbors * 0.7:
            confidence = "high"
        elif valid_mask.sum() >= n_neighbors * 0.3:
            confidence = "medium"
        else:
            confidence = "low"

    predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

    return RecommendResponse(
        user_id=request.user_id,
        movie=request.movie,
        predicted_rating=round(predicted_rating, 2),
        similar_users=similar_user_ids[:10],
        confidence=confidence
    )


@app.get("/cluster_profiles/{k}", response_model=ClusterProfileResponse)
def get_cluster_profiles(k: int):
    """Get top movies per cluster for a given k."""
    if k < 2 or k > 15:
        raise HTTPException(status_code=400, detail="k must be between 2 and 15")

    load_data()
    features = _cache["features"]
    features_raw = _cache["features_raw"]
    movie_cols = _cache["movie_columns"]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    cluster_profiles = {}
    cluster_sizes = {}

    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_sizes[f"cluster_{cluster_id}"] = int(cluster_mask.sum())
        if cluster_mask.sum() > 0:
            avg_ratings = np.nanmean(features_raw[cluster_mask], axis=0)
            top_indices = np.argsort(avg_ratings)[::-1][:5]
            top_movies = {}
            for idx in top_indices:
                if not np.isnan(avg_ratings[idx]):
                    top_movies[movie_cols[idx]] = round(float(avg_ratings[idx]), 2)
            cluster_profiles[f"cluster_{cluster_id}"] = top_movies
        else:
            cluster_profiles[f"cluster_{cluster_id}"] = {}

    return ClusterProfileResponse(k=k, cluster_profiles=cluster_profiles, cluster_sizes=cluster_sizes)


@app.get("/users")
def get_users():
    """Get list of user IDs in the dataset."""
    load_data()
    pivot = _cache["pivot"]
    user_ids = sorted(pivot.index.tolist())
    return {
        "user_column": "userId",
        "user_ids": user_ids,
        "total_users": len(user_ids)
    }


@app.get("/data_info")
def get_data_info():
    """Get information about the loaded dataset."""
    load_data()
    pivot = _cache["pivot"]
    movie_cols = _cache["movie_columns"]
    user_ids = sorted(pivot.index.tolist())
    return {
        "total_users": len(user_ids),
        "total_movies": len(movie_cols),
        "user_column": "userId",
        "movie_columns": movie_cols,
        "sample_users": user_ids[:5],
        "data_shape": list(pivot.shape)
    }


# ===================== NEW: Collaborative Filtering Evaluation =====================

class CFEvalRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=100)
    test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)

@app.post("/cf_evaluate")
def cf_evaluate(req: CFEvalRequest):
    """Evaluate KNN CF with train/test split. Returns train MAE, test MAE, per-movie breakdown."""
    load_data()
    raw = _cache["raw"].copy()

    np.random.seed(42)
    test_mask = np.random.rand(len(raw)) < req.test_fraction
    train_df = raw[~test_mask]
    test_df = raw[test_mask]

    all_users = sorted(raw["userId"].unique())
    all_movies = sorted(raw["title"].unique())
    train_pivot = train_df.pivot_table(index="userId", columns="title", values="rating")
    train_pivot = train_pivot.reindex(index=all_users, columns=all_movies)

    imputer = SimpleImputer(strategy="mean")
    train_filled = imputer.fit_transform(train_pivot.values)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)

    k = min(req.k, len(all_users) - 1)
    knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    knn.fit(train_scaled)

    user_idx_map = {uid: i for i, uid in enumerate(all_users)}
    movie_idx_map = {m: i for i, m in enumerate(all_movies)}

    def predict_batch(df, max_samples=2000):
        sample = df.sample(min(max_samples, len(df)), random_state=42) if len(df) > max_samples else df
        preds, actuals, movies_out = [], [], []
        for _, row in sample.iterrows():
            uid, movie, actual = row["userId"], row["title"], row["rating"]
            if uid not in user_idx_map or movie not in movie_idx_map:
                continue
            uidx = user_idx_map[uid]
            midx = movie_idx_map[movie]
            dists, idxs = knn.kneighbors([train_scaled[uidx]])
            neighbor_idxs = idxs[0][1:]
            neighbor_dists = dists[0][1:]
            weights = np.maximum(1 - neighbor_dists, 0.01)
            neighbor_ratings = train_filled[neighbor_idxs, midx]
            valid = ~np.isnan(neighbor_ratings)
            if valid.sum() == 0:
                pred = np.nanmean(train_filled[:, midx])
            else:
                pred = np.average(neighbor_ratings[valid], weights=weights[valid])
            preds.append(round(float(np.clip(pred, 1.0, 5.0)), 3))
            actuals.append(float(actual))
            movies_out.append(movie)
        return preds, actuals, movies_out

    train_preds, train_actuals, _ = predict_batch(train_df, 2000)
    test_preds, test_actuals, test_movies = predict_batch(test_df, 5000)

    train_mae = round(float(np.mean(np.abs(np.array(train_preds) - np.array(train_actuals)))), 4)
    test_mae = round(float(np.mean(np.abs(np.array(test_preds) - np.array(test_actuals)))), 4)

    movie_errors = {}
    for m, p, a in zip(test_movies, test_preds, test_actuals):
        movie_errors.setdefault(m, []).append(abs(p - a))
    movie_mae_sorted = sorted(movie_errors.items(), key=lambda x: -len(x[1]))[:15]

    return {
        "k": req.k, "test_fraction": req.test_fraction,
        "train_mae": train_mae, "test_mae": test_mae,
        "n_train": len(train_preds), "n_test": len(test_preds),
        "movie_mae": [{"movie": m, "mae": round(float(np.mean(e)), 3), "n": len(e)} for m, e in movie_mae_sorted],
    }


@app.post("/cf_sweep")
def cf_sweep():
    """Sweep k=1..50, return train/test MAE. Uses sampling for speed."""
    load_data()
    raw = _cache["raw"].copy()

    np.random.seed(42)
    test_mask = np.random.rand(len(raw)) < 0.2
    train_df = raw[~test_mask]
    test_df = raw[test_mask]

    test_sample = test_df.sample(min(1500, len(test_df)), random_state=42)
    train_sample = train_df.sample(min(1500, len(train_df)), random_state=42)

    all_users = sorted(raw["userId"].unique())
    all_movies = sorted(raw["title"].unique())
    train_pivot = train_df.pivot_table(index="userId", columns="title", values="rating")
    train_pivot = train_pivot.reindex(index=all_users, columns=all_movies)

    imputer = SimpleImputer(strategy="mean")
    train_filled = imputer.fit_transform(train_pivot.values)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_filled)

    user_idx_map = {uid: i for i, uid in enumerate(all_users)}
    movie_idx_map = {m: i for i, m in enumerate(all_movies)}

    knn = NearestNeighbors(n_neighbors=52, metric="cosine")
    knn.fit(train_scaled)

    sample_users = set(test_sample["userId"].tolist() + train_sample["userId"].tolist())
    user_neighbors = {}
    for uid in sample_users:
        if uid in user_idx_map:
            uidx = user_idx_map[uid]
            dists, idxs = knn.kneighbors([train_scaled[uidx]])
            user_neighbors[uid] = (dists[0], idxs[0])

    def compute_mae(sample_df, k_val):
        errors = []
        for _, row in sample_df.iterrows():
            uid, movie, actual = row["userId"], row["title"], row["rating"]
            if uid not in user_neighbors or movie not in movie_idx_map:
                continue
            dists_all, idxs_all = user_neighbors[uid]
            midx = movie_idx_map[movie]
            end = min(k_val + 1, len(idxs_all))
            neighbor_idxs = idxs_all[1:end]
            neighbor_dists = dists_all[1:end]
            weights = np.maximum(1 - neighbor_dists, 0.01)
            ratings = train_filled[neighbor_idxs, midx]
            valid = ~np.isnan(ratings)
            if valid.sum() == 0:
                pred = np.nanmean(train_filled[:, midx])
            else:
                pred = np.average(ratings[valid], weights=weights[valid])
            errors.append(abs(float(np.clip(pred, 1.0, 5.0)) - actual))
        return round(float(np.mean(errors)), 4) if errors else None

    results = []
    for k_val in range(1, 51):
        results.append({
            "k": k_val,
            "train_mae": compute_mae(train_sample, k_val),
            "test_mae": compute_mae(test_sample, k_val),
        })
    return results


@app.post("/predict_movie")
def predict_movie(user_id: int = 1, movie: str = "The Matrix", k: int = 10):
    """Predict a single user-movie rating with user context."""
    load_data()
    pivot = _cache["pivot"]
    features = _cache["features"]
    features_raw = _cache["features_raw"]
    movie_cols = _cache["movie_columns"]
    user_ids = sorted(pivot.index.tolist())

    if movie not in movie_cols:
        raise HTTPException(400, f"Movie '{movie}' not found")
    if user_id not in user_ids:
        raise HTTPException(400, f"User {user_id} not found")

    user_idx = user_ids.index(user_id)
    movie_idx = movie_cols.index(movie)

    k_use = min(k, len(user_ids) - 1)
    knn = NearestNeighbors(n_neighbors=k_use + 1, metric="cosine")
    knn.fit(features)

    dists, idxs = knn.kneighbors([features[user_idx]])
    neighbor_idxs = idxs[0][1:]
    neighbor_dists = dists[0][1:]

    weights = np.maximum(1 - neighbor_dists, 0.01)
    ratings = features_raw[neighbor_idxs, movie_idx]
    valid = ~np.isnan(ratings)

    if valid.sum() == 0:
        pred = float(np.nanmean(features_raw[:, movie_idx]))
        confidence = "low"
    else:
        pred = float(np.average(ratings[valid], weights=weights[valid]))
        frac = valid.sum() / len(valid)
        confidence = "high" if frac > 0.7 else "medium" if frac > 0.3 else "low"

    pred = float(np.clip(pred, 1.0, 5.0))

    actual_val = pivot.loc[user_id, movie]
    actual = float(actual_val) if not pd.isna(actual_val) else None

    user_row = pivot.loc[user_id].dropna().sort_values(ascending=False)
    user_top = [[m, int(r)] for m, r in user_row.head(5).items()]
    user_bottom = [[m, int(r)] for m, r in user_row.tail(3).items()]

    return {
        "predicted": round(pred, 2),
        "actual": actual,
        "confidence": confidence,
        "k": k,
        "user_top_rated": user_top,
        "user_bottom_rated": user_bottom,
        "similar_user_ids": [int(user_ids[i]) for i in neighbor_idxs[:5]],
    }


# Serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/app")
async def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
