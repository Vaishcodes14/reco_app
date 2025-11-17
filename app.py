# app.py
import streamlit as st
import joblib, json, os, pandas as pd, numpy as np, math
from io import BytesIO

BASE_DIR = "./data"

# -------------------------------
# SAFE LOADER
# -------------------------------
@st.cache_resource
def load_artifacts(base_dir):
    required_files = [
        "als_model.joblib",
        "user_le.joblib",
        "item_le.joblib",
        "user_item_matrix.joblib",
        "prod_meta.csv",
        "co_view_top.json",
        "popular_items.joblib"
    ]
    # Accept optional lgb_reranker.joblib
    missing = [f for f in required_files if not os.path.exists(os.path.join(base_dir, f))]
    if missing:
        st.error(f"Missing required files in data/: {missing}")
        return None

    try:
        model = joblib.load(os.path.join(base_dir, "als_model.joblib"))
        user_le = joblib.load(os.path.join(base_dir, "user_le.joblib"))
        item_le = joblib.load(os.path.join(base_dir, "item_le.joblib"))
        user_item_matrix = joblib.load(os.path.join(base_dir, "user_item_matrix.joblib"))
        co_view_top = json.load(open(os.path.join(base_dir, "co_view_top.json")))
        popular_items = joblib.load(os.path.join(base_dir, "popular_items.joblib"))
        prod_meta_df = pd.read_csv(os.path.join(base_dir, "prod_meta.csv"))
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

    # Try load LightGBM reranker if present
    lgb_reranker = None
    lgb_path = os.path.join(base_dir, "lgb_reranker.joblib")
    if os.path.exists(lgb_path):
        try:
            lgb_reranker = joblib.load(lgb_path)
        except Exception as e:
            st.warning(f"Could not load lgb_reranker: {e}. Continuing without reranker.")

    # Normalize prod_meta to mapping keyed by item_code (int)
    # Accept either item_code column or item_id and map using item_le
    if "item_code" in prod_meta_df.columns:
        prod_meta_df["item_code"] = prod_meta_df["item_code"].astype(int)
        prod_meta_map = prod_meta_df.set_index("item_code").to_dict(orient="index")
    elif "item_id" in prod_meta_df.columns:
        # map item_id -> code via item_le if available
        mapping = {}
        for _, row in prod_meta_df.iterrows():
            iid = str(row["item_id"])
            try:
                code = int(item_le.transform([iid])[0])
                mapping[code] = row.to_dict()
            except Exception:
                # skip items unknown to item_le
                continue
        prod_meta_map = mapping
    else:
        prod_meta_map = {}

    # Build item_pop_map from prod_meta if present, else zero
    item_pop_map = {}
    for code, meta in prod_meta_map.items():
        # try 'sales_count' or 'item_popularity' or fallback 0
        item_pop_map[int(code)] = float(meta.get("sales_count", meta.get("item_popularity", 0)) or 0)

    return {
        "als_model": model,
        "user_le": user_le,
        "item_le": item_le,
        "user_item_matrix": user_item_matrix,
        "co_view_top": co_view_top,
        "popular_items": popular_items,
        "prod_meta_map": prod_meta_map,
        "item_pop_map": item_pop_map,
        "lgb_reranker": lgb_reranker
    }

art = load_artifacts(BASE_DIR)
if art is None:
    st.stop()

model = art["als_model"]
user_le = art["user_le"]
item_le = art["item_le"]
user_item_matrix = art["user_item_matrix"]
co_view_top = art["co_view_top"]
popular_items = art["popular_items"]
prod_meta_map = art["prod_meta_map"]
item_pop_map = art["item_pop_map"]
lgb_reranker = art["lgb_reranker"]

# -------------------------------
# Lightweight feature builder for web (fast)
# -------------------------------
def compute_features_web(user_code, cand_code, context_code=None):
    """
    user_code, cand_code, context_code are integer encoded codes (or None).
    Returns a dict of numeric features expected by the LightGBM reranker.
    Keep features simple and fast.
    """
    f = {}
    # item popularity
    f["item_pop"] = float(item_pop_map.get(int(cand_code), 0))
    # co-view probability with context
    if context_code is not None:
        try:
            f["co_with_context"] = float(co_view_top.get(str(context_code), {}).count(int(cand_code))) if False else 0.0
            # above is placeholder; we will compute below using stored structure
        except Exception:
            f["co_with_context"] = 0.0
    else:
        f["co_with_context"] = 0.0

    # Better co_prob: if context_code present and exists in co_view_top, compute normalized count
    if context_code is not None:
        try:
            # co_view_top keys are str(item_code) -> list of item_codes most common
            lst = co_view_top.get(str(context_code), [])
            # approximate probability as position-weighted score
            if len(lst) > 0 and int(cand_code) in lst:
                pos = lst.index(int(cand_code))
                # higher weight for earlier positions
                f["co_with_context"] = max(0.0, (len(lst) - pos) / len(lst))
            else:
                f["co_with_context"] = 0.0
        except Exception:
            f["co_with_context"] = 0.0

    # price difference (relative) from prod_meta_map
    try:
        cand_meta = prod_meta_map.get(int(cand_code), {})
        ctx_meta = prod_meta_map.get(int(context_code), {}) if context_code is not None else {}
        p_c = float(cand_meta.get("price", np.nan)) if cand_meta.get("price") not in (None,"") else np.nan
        p_ctx = float(ctx_meta.get("price", np.nan)) if ctx_meta.get("price") not in (None,"") else np.nan
        if not np.isnan(p_c) and not np.isnan(p_ctx) and p_ctx!=0:
            f["price_diff_pct"] = abs(p_c - p_ctx) / (p_ctx + 1e-9)
        else:
            f["price_diff_pct"] = 0.0
    except Exception:
        f["price_diff_pct"] = 0.0

    # brand/category affinity approximations: check if user has interacted with same brand/category recently
    # For speed we do a cheap check: if prod_meta_map has category/brand we set flags 0/1 by scanning user history not done here
    f["same_brand_as_context"] = 0.0
    try:
        b_cand = str(cand_meta.get("brand","")).lower() if cand_meta else ""
        b_ctx = str(ctx_meta.get("brand","")).lower() if ctx_meta else ""
        f["same_brand_as_context"] = 1.0 if (b_cand and b_ctx and b_cand==b_ctx) else 0.0
    except:
        f["same_brand_as_context"] = 0.0

    # basic popularity indicator
    f["is_top_popular"] = 1.0 if f["item_pop"] > 0 and f["item_pop"] >= np.percentile(list(item_pop_map.values()) or [0], 90) else 0.0

    # simple placeholder for user-item historic interaction counts:
    # try to see if user_item_matrix contains an entry for this user & item (user_item_matrix is csr user x item)
    try:
        if user_code is not None:
            # user_item_matrix is csr (users x items)
            val = 0
            try:
                # note: user_item_matrix may be big; using .toarray() not feasible - use indexing
                val = user_item_matrix[user_code, cand_code]
            except Exception:
                # some sparse matrices don't support direct indexing; fallback 0
                val = 0
            f["user_item_interaction"] = float(val or 0)
        else:
            f["user_item_interaction"] = 0.0
    except Exception:
        f["user_item_interaction"] = 0.0

    # last-resort features
    f["const_1"] = 1.0
    return f

# -------------------------------
# Candidate generation helpers (fast)
# -------------------------------
def get_co_view_candidates(item_id, topk=50):
    # item_id is item_id string (e.g., 'p101') or integer item_code
    try:
        code = int(item_le.transform([item_id])[0])
    except Exception:
        # maybe item_id is already code
        try:
            code = int(item_id)
        except:
            return []
    lst = co_view_top.get(str(code), [])
    return [item_le.inverse_transform([int(c)])[0] for c in lst[:topk]]

def get_als_candidates_for_user(user_id, topk=50):
    try:
        u_idx = int(user_le.transform([user_id])[0])
    except Exception:
        return []
    try:
        recs = model.recommend(u_idx, user_item_matrix, N=topk)
        return [item_le.inverse_transform([int(r[0])])[0] for r in recs]
    except Exception:
        return []

def get_popular_candidates(topk=50):
    return [item_le.inverse_transform([int(i)])[0] for i in popular_items[:topk]]

# utility to build candidate set
def build_candidates(user_id=None, current_item_id=None, limit=200):
    cands = []
    # ALS candidates if user known
    if user_id:
        cands += get_als_candidates_for_user(user_id, topk=limit//3)
    # co-view candidates from current item
    if current_item_id:
        cands += get_co_view_candidates(current_item_id, topk=limit//3)
    # popular
    cands += get_popular_candidates(topk=limit//3)
    # dedupe preserving order
    seen = set()
    out = []
    for it in cands:
        if it not in seen:
            seen.add(it); out.append(it)
        if len(out) >= limit:
            break
    return out

# -------------------------------
# Display helpers
# -------------------------------
def get_meta_by_item_id(item_id):
    try:
        code = int(item_le.transform([item_id])[0])
        return prod_meta_map.get(code, {})
    except Exception:
        return {}

def show_item_table(item_list):
    rows=[]
    for it in item_list:
        meta = get_meta_by_item_id(it)
        rows.append({
            "item_id": it,
            "title": meta.get("title",""),
            "brand": meta.get("brand",""),
            "category": meta.get("category_id",""),
            "price": meta.get("price","")
        })
    return pd.DataFrame(rows)

# -------------------------------
# UI: product-name friendly input (keeps earlier behavior)
# -------------------------------
import difflib
# build title map from prod_meta_map
title_map = {}
for code,meta in prod_meta_map.items():
    t = str(meta.get("title","")).strip()
    iid = meta.get("item_id", None)
    if t:
        title_map[t] = {"item_code": int(code), "item_id": iid}

def find_titles(q, maxr=8):
    if not q: return []
    ql = q.lower().strip()
    subs = [t for t in title_map.keys() if ql in t.lower()]
    if len(subs) < maxr:
        fuzzy = difflib.get_close_matches(q, list(title_map.keys()), n=maxr, cutoff=0.5)
        combined = []
        for s in subs + fuzzy:
            if s not in combined:
                combined.append(s)
        subs = combined[:maxr]
    return subs

st.title("🛒 Smart Product Recommendation — Reranker enabled")
st.write("Type product name (or item_id) and optional user id for personalized results.")

user_id_input = st.text_input("User ID (optional):")
product_name_input = st.text_input("Product name or item_id:")

selected_item_id = None
if product_name_input:
    matches = find_titles(product_name_input)
    if matches:
        chosen = st.selectbox("Pick the exact product:", ["-- choose --"] + matches)
        if chosen and chosen != "-- choose --":
            selected_item_id = title_map[chosen]["item_id"]
            st.write("Selected:", chosen, selected_item_id)
    else:
        # if looks like item_id, accept
        if product_name_input.startswith("p"):
            selected_item_id = product_name_input

N = st.slider("Number of recommendations", 1, 20, 6)

if st.button("Get Recommendations"):
    # build candidates
    candidates = build_candidates(user_id=user_id_input if user_id_input else None,
                                  current_item_id=selected_item_id if selected_item_id else None,
                                  limit=200)
    if not candidates:
        st.write("No candidates generated. Try a different product or user.")
        st.stop()

    # prepare feature rows
    rows = []
    cand_codes = []
    for cand in candidates:
        try:
            cand_code = int(item_le.transform([cand])[0])
        except Exception:
            continue
        # get user_code if available
        try:
            user_code = int(user_le.transform([user_id_input])[0]) if user_id_input and user_id_input in user_le.classes_ else None
        except Exception:
            user_code = None
        # context code
        try:
            ctx_code = int(item_le.transform([selected_item_id])[0]) if selected_item_id else None
        except Exception:
            ctx_code = None

        feat = compute_features_web(user_code, cand_code, context_code=ctx_code)
        rows.append(feat)
        cand_codes.append(cand)

    if len(rows)==0:
        st.write("No candidate features computed.")
        st.stop()

    X = pd.DataFrame(rows).fillna(0)

    # Score candidates with reranker if available
    if lgb_reranker is not None:
        try:
            scores = lgb_reranker.predict(X)
            ranked = [c for _,c in sorted(zip(scores, cand_codes), key=lambda x: -x[0])]
        except Exception as e:
            st.warning(f"Reranker scoring failed: {e}. Falling back to heuristic ranking.")
            ranked = cand_codes
    else:
        # fallback: use heuristic ordering by item_pop + co_with_context + user_item_interaction
        scored = []
        for f,c in zip(rows, cand_codes):
            score = f.get("item_pop",0)*0.6 + f.get("co_with_context",0)*0.3 + f.get("user_item_interaction",0)*0.2 - f.get("price_diff_pct",0)*0.1
            scored.append((score,c))
        ranked = [c for _,c in sorted(scored, key=lambda x:-x[0])]

    topk = ranked[:N]
    st.subheader("Top Recommendations")
    st.table(show_item_table(topk))

    st.subheader("Why these?")
    for it in topk:
        why = "Popular"
        # quick check if in ALS candidates
        if user_id_input and it in get_als_candidates_for_user(user_id_input, topk=50):
            why = "Personalized (ALS)"
        elif selected_item_id and it in get_co_view_candidates(selected_item_id, topk=50):
            why = f"Frequently bought with {selected_item_id}"
        st.write(f"- {it} — {why}")
