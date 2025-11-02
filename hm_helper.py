import os, time, json, numpy as np, pandas as pd
from pathlib import Path
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel

# Quiet backends (esp. in Colab)
os.environ["USE_TF"] = "0"; os.environ["USE_FLAX"] = "0"; os.environ["USE_JAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"; os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tic(msg: str):
    print(f"[START] {msg}", flush=True); return time.time()

def toc(t0: float, msg: str):
    dt = time.time() - t0; print(f"[DONE ] {msg} in {dt:.2f}s", flush=True)

# ---- Embeddings ----
def encode_texts(texts, model_name="openai/clip-vit-base-patch32", batch_size=64, max_length=77, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    is_clip = "clip" in model_name.lower()
    if is_clip:
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name)
        if max_length > tokenizer.model_max_length:
            max_length = tokenizer.model_max_length
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    model.to(device).eval()

    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if is_clip:
                enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            else:
                enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            if is_clip:
                pooled = out.pooler_output
            else:
                token_emb = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).type_as(token_emb)
                pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks)

# ---- Monthly panel ----
def to_month_frame(df: pd.DataFrame) -> pd.DataFrame:
    date_col = "week_start" if "week_start" in df.columns else ("month_start" if "month_start" in df.columns else None)
    if date_col is None:
        raise ValueError("CSV must include a 'week_start' or 'month_start' column.")
    d = df.copy()
    if date_col == "month_start":
        d = d.rename(columns={"month_start":"week_start"})
    d["month_ts"] = d["week_start"].dt.to_period("M").dt.to_timestamp()
    keep_first_cols = [c for c in ["detail_desc","product_type_name","graphical_appearance_name",
                                   "colour_group_name","index_group_name","garment_group_name"]
                       if c in d.columns]
    agg_dict = {"demand":"sum", "mean_price":"mean"}
    for c in keep_first_cols: agg_dict[c] = "first"
    m = (d.groupby(["article_id","sales_channel_id","month_ts"], as_index=False).agg(agg_dict))
    m["month_num"] = m["month_ts"].dt.month.astype("Int64")
    return m

def build_monthly_features(dfm: pd.DataFrame) -> pd.DataFrame:
    d = dfm.sort_values(["article_id","sales_channel_id","month_ts"]).reset_index(drop=True)
    grp = d.groupby(["article_id","sales_channel_id"], group_keys=False)

    # Lags & price dynamics (leakage-safe)
    d["lag_m1"] = grp["demand"].shift(1); d["lag_m2"] = grp["demand"].shift(2); d["lag_m3"] = grp["demand"].shift(3)
    d["price_change"] = grp["mean_price"].diff()
    price_exp_mean = grp["mean_price"].transform(lambda s: s.shift(1).expanding().mean())
    d["price_rel_avg"] = d["mean_price"] / price_exp_mean.replace(0, np.nan)

    # Rolling from lags
    lag_cols = ["lag_m1","lag_m2","lag_m3"]
    d["roll_mean_3m"] = d[lag_cols].mean(axis=1)
    d["roll_std_3m"]  = d[lag_cols].std(axis=1).fillna(0)

    # Drop NA rows from shifts/ratios
    need = ["lag_m1","lag_m2","lag_m3","price_change","price_rel_avg"]
    d = d.dropna(subset=need).reset_index(drop=True)

    # Nonlinear
    eps = 1e-9
    d["log_price"] = np.log(d["mean_price"].clip(lower=eps))
    d["price_sq"]  = np.square(d["mean_price"])
    for c in ["lag_m1","lag_m2","lag_m3","roll_mean_3m","roll_std_3m"]:
        d[f"{c}_sq"] = np.square(d[c]); d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))

    # Seasonality, lifecycle
    m = d["month_num"].astype(float)
    d["sin_m"] = np.sin(2*np.pi*m/12.0)
    first_ts = grp["month_ts"].transform("min")
    month_idx_current = d["month_ts"].dt.year * 12 + d["month_ts"].dt.month
    month_idx_first   = first_ts.dt.year * 12 + first_ts.dt.month
    d["age_m"] = (month_idx_current - month_idx_first).astype(float)
    d["is_new_3m"] = (d["age_m"] <= 3).astype(float)
    return d

from scipy.sparse import csr_matrix, hstack

def design_matrices(train, test, *, include_numeric=True, include_month_ohe=True, include_channel_ohe=True,
                    include_meta_ohe=False, include_embeddings=False, limit_embed_dims=None,
                    inter_month=False, inter_age=False, inter_channel=False, model_name="openai/clip-vit-base-patch32",
                    dev_skip_embed=False):
    y_train = train["demand"].values; y_test  = test["demand"].values; y_train_log = np.log1p(y_train)

    X_penalized_tr_raw, X_penalized_te_raw = [], []
    X_unpenalized_tr, X_unpenalized_te = [], []
    X_chan_train = None

    # Unpenalized OHE
    if include_month_ohe:
        ohe_month = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(train[["month_num"]])
        X_month_train = ohe_month.transform(train[["month_num"]])
        X_month_test  = ohe_month.transform(test[["month_num"]])
        X_unpenalized_tr.append(X_month_train); X_unpenalized_te.append(X_month_test)

    if include_channel_ohe:
        ohe_chan  = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(train[["sales_channel_id"]])
        X_chan_train = ohe_chan.transform(train[["sales_channel_id"]])
        X_chan_test  = ohe_chan.transform(test[["sales_channel_id"]])
        X_unpenalized_tr.append(X_chan_train); X_unpenalized_te.append(X_chan_test)

    if include_meta_ohe:
        meta_cols_all = ["product_type_name","graphical_appearance_name","colour_group_name","index_group_name","garment_group_name"]
        meta_cols = [c for c in meta_cols_all if c in train.columns]
        if meta_cols:
            ohe_meta = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(train[meta_cols])
            X_meta_train = ohe_meta.transform(train[meta_cols]); X_meta_test = ohe_meta.transform(test[meta_cols])
        else:
            X_meta_train = csr_matrix((len(train), 0)); X_meta_test = csr_matrix((len(test), 0))
        X_unpenalized_tr.append(X_meta_train); X_unpenalized_te.append(X_meta_test)

    # Penalized numeric
    if include_numeric:
        num_cols = ["mean_price","lag_m1","lag_m2","lag_m3","price_change","price_rel_avg",
                    "log_price","price_sq",
                    "lag_m1_sq","lag_m2_sq","lag_m3_sq",
                    "log_lag_m1","log_lag_m2","log_lag_m3",
                    "roll_mean_3m","roll_std_3m","roll_mean_3m_sq","roll_std_3m_sq",
                    "log_roll_mean_3m","log_roll_std_3m","sin_m","age_m","is_new_3m"]
        X_num_train_raw = sp.csr_matrix(train[num_cols].to_numpy())
        X_num_test_raw  = sp.csr_matrix(test[num_cols].to_numpy())
        X_penalized_tr_raw.append(X_num_train_raw); X_penalized_te_raw.append(X_num_test_raw)

    # Penalized embeddings (+ interactions)
    if include_embeddings and not dev_skip_embed:
        prod_desc = pd.concat([train[["article_id","detail_desc"]], test[["article_id","detail_desc"]]]).drop_duplicates("article_id")
        prod_desc["detail_desc"] = prod_desc["detail_desc"].fillna("").astype(str)
        prod_ids = prod_desc["article_id"].tolist(); prod_texts = prod_desc["detail_desc"].tolist()
        emb = encode_texts(prod_texts, model_name=model_name, device=("cuda" if torch.cuda.is_available() else "cpu"))
        if (limit_embed_dims is not None) and (limit_embed_dims < emb.shape[1]): emb = emb[:, :limit_embed_dims]

        emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb, columns=emb_cols); emb_df.insert(0, "article_id", prod_ids)
        train_e = train.merge(emb_df, on="article_id", how="left")
        test_e  = test.merge(emb_df, on="article_id", how="left")
        X_emb_train_raw = csr_matrix(train_e[emb_cols].to_numpy()); X_emb_test_raw  = csr_matrix(test_e[emb_cols].to_numpy())
        X_penalized_tr_raw.append(X_emb_train_raw); X_penalized_te_raw.append(X_emb_test_raw)

        if inter_month:
            vtr = train[["sin_m"]].values.ravel(); vte = test[["sin_m"]].values.ravel()
            Dtr = sp.diags(vtr); Dte = sp.diags(vte)
            X_penalized_tr_raw.append(Dtr.dot(X_emb_train_raw)); X_penalized_te_raw.append(Dte.dot(X_emb_test_raw))

        if inter_age:
            vtr = train[["is_new_3m"]].values.ravel(); vte = test[["is_new_3m"]].values.ravel()
            Dtr = sp.diags(vtr); Dte = sp.diags(vte)
            X_penalized_tr_raw.append(Dtr.dot(X_emb_train_raw)); X_penalized_te_raw.append(Dte.dot(X_emb_test_raw))

        if inter_channel and (X_chan_train is not None):
            for j in range(X_chan_train.shape[1]):
                col_tr = X_chan_train[:, j].toarray().ravel(); col_te = X_chan_test[:, j].toarray().ravel()
                Dtr = sp.diags(col_tr); Dte = sp.diags(col_te)
                X_penalized_tr_raw.append(Dtr.dot(X_emb_train_raw)); X_penalized_te_raw.append(Dte.dot(X_emb_test_raw))

    # Scale penalized block
    if X_penalized_tr_raw:
        from scipy.sparse import hstack as _hstack
        X_penalized_train_raw_all = _hstack(X_penalized_tr_raw, format="csr")
        X_penalized_test_raw_all  = _hstack(X_penalized_te_raw, format="csr")
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_penalized_train_raw_all)
        X_test_scaled  = scaler.transform(X_penalized_test_raw_all)
    else:
        X_train_scaled = csr_matrix((len(train), 0)); X_test_scaled  = csr_matrix((len(test), 0))

    # Combine with unpenalized
    final_tr = [b for b in [X_train_scaled] + X_unpenalized_tr if b is not None and b.shape[1] > 0]
    final_te = [b for b in [X_test_scaled] + X_unpenalized_te if b is not None and b.shape[1] > 0]
    from scipy.sparse import hstack as _hstack
    X_train = _hstack(final_tr, format="csr") if final_tr else csr_matrix((len(train), 0))
    X_test  = _hstack(final_te, format="csr") if final_te else csr_matrix((len(test), 0))
    return X_train, X_test, y_train, y_test, y_train_log

# ---- IO & fit ----
def load_and_prepare(CSV_PATH, PRODUCT_TYPE_FILTER, N_PRODUCTS, SAMPLE, RANDOM_STATE):
    t0 = tic(f"Loading {CSV_PATH}")
    peek = pd.read_csv(CSV_PATH, nrows=1)
    parse_col = "week_start" if "week_start" in peek.columns else ("month_start" if "month_start" in peek.columns else None)
    if parse_col is None:
        raise ValueError("CSV must include 'week_start' or 'month_start'.")
    df = pd.read_csv(CSV_PATH, dtype={"article_id":"string"}, parse_dates=[parse_col])
    if parse_col == "month_start" and "week_start" not in df.columns:
        df = df.rename(columns={"month_start":"week_start"})
    toc(t0, "Loaded")

    t0 = tic("Cleaning")
    df["detail_desc"] = df["detail_desc"].fillna("").astype(str)
    df["sales_channel_id"] = df["sales_channel_id"].astype("Int64")
    df = df.dropna(subset=["demand","mean_price","sales_channel_id"]).reset_index(drop=True)
    toc(t0, "Cleaned")

    np.random.seed(RANDOM_STATE)
    if PRODUCT_TYPE_FILTER:
        if "product_type_name" not in df.columns:
            raise ValueError("--product_type_filter specified, but 'product_type_name' column not in CSV.")
        keep_ids = df[df["product_type_name"].str.contains(PRODUCT_TYPE_FILTER, case=False, na=False)]["article_id"].unique()
        if len(keep_ids) > 0: df = df[df["article_id"].isin(keep_ids)].copy()
    else:
        unique_ids = pd.Series(df["article_id"].dropna().unique())
        if SAMPLE == "random":
            keep_ids = unique_ids.sample(n=min(N_PRODUCTS, len(unique_ids)), random_state=RANDOM_STATE)
        elif SAMPLE == "top":
            keep_ids = (df.groupby("article_id")["demand"].size()
                          .sort_values(ascending=False)
                          .head(min(N_PRODUCTS, len(unique_ids))).index)
        else:
            raise ValueError("SAMPLE must be 'random' or 'top'")
        df = df[df["article_id"].isin(keep_ids)].copy()

    dfm = to_month_frame(df); dfm = build_monthly_features(dfm)
    months = np.sort(dfm["month_ts"].unique())
    holdout_months = months[-2:] if len(months) >= 6 else months[-max(1, len(months)//5):]
    train = dfm[~dfm["month_ts"].isin(holdout_months)].reset_index(drop=True)
    test  = dfm[dfm["month_ts"].isin(holdout_months)].reset_index(drop=True)
    if len(train)==0 or len(test)==0:
        raise RuntimeError("No data left in train or test set after filtering and feature engineering.")
    return train, test

def fit_lasso(X_train, y_train_log, X_test, y_test, label):
    custom_alphas = np.logspace(-6, -1, 100)
    lasso = LassoCV(alphas=custom_alphas, cv=5, random_state=0, n_jobs=-1, verbose=10)
    lasso.fit(X_train, y_train_log)
    pred = np.expm1(lasso.predict(X_test))
    r2   = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    mae  = mean_absolute_error(y_test, pred)
    n_coef_total = int(len(lasso.coef_))
    n_coef_nnz = int(np.sum(np.abs(lasso.coef_) > 1e-10))
    alpha = getattr(lasso, "alpha_", np.nan)
    return dict(r2=r2, rmse=rmse, mae=mae, alpha=alpha, n_coef_total=n_coef_total, n_coef_nnz=n_coef_nnz)
