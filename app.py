# app.py
import os
import joblib
import json
import pandas as pd
import streamlit as st

BASE_DIR = "./data"   # keep artifacts inside repo/data/

# ---------------------------
# Load artifacts safely
# ---------------------------
@st.cache_resource
def load_artifacts(base_dir):
    required_files = {
        "als": "als_model.joblib",
        "user_le": "user_le.joblib",
        "item_le": "item_le.joblib",
        "user_item_matrix": "user_item_matrix.joblib",
        "co_view": "co_view_top.json",
        "popular": "popular_items.joblib",
        "prod_meta": "prod_meta.csv"
    }

    missing = [v for k, v in required_files.items() if not os.path.exists(os.path.join(base_dir, v))]
    if missing:
        st.error(f"Missing files in `{base_dir}`: {missing}. Upload them and refresh.")
        return None

    try:
        model = joblib.load(os.path.join(base_dir, required_files["als"]))
        user_le = joblib.load(os.path.join(base_dir, required_files["user_le"]))
        item_le = joblib.load(os.path.join(base_dir, required_files["item_le"]))
        user_item_matrix = joblib.load(os.path.join(base_dir, required_files["user_item_matrix"]))
        with open(os.path.join(base_dir, required_files["co_view"]), "r") as f:
            co_view_top = json.load(f)
        popular_items = joblib.load(os.path.join(base_dir, required_files["popular"]))
        prod_meta = pd.read_csv(os.path.join(base_dir, required_files["prod_meta"]))
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

    # Normalize prod_meta: prefer integer item_code index; create maps for quick lookup
    prod_meta_index = {}
    prod_meta_by_itemid = {}

    if "item_code" in prod_meta.columns:
        # ensure item_code is string keys for consistent lookup later
        for _, r in prod_meta.iterrows():
            key = str(int(r["item_code"])) if pd.notna(r["item_code"]) else None
            if key is not None:
                prod_meta_index[key] = {
                    "item_id": str(r.get("item_id", "")),
                    "title": r.get("title", ""),
                    "category_id": r.get("category_id", ""),
                    "brand": r.get("brand", ""),
                    "price": r.get("price", "")
                }
    if "item_id" in prod_meta.columns:
        for _, r in prod_meta.iterrows():
            prod_meta_by_itemid[str(r["item_id"])] = {
                "title": r.get("title", ""),
                "category_id": r.get("category_id", ""),
                "brand": r.get("brand", ""),
                "price": r.get("price", "")
            }

    return {
        "model": model,
        "user_le": user_le,
        "item_le": item_le,
        "user_item_matrix": user_item_matrix,
        "co_view_top": co_view_top,
        "popular_items": popular_items,
        "prod_meta_index": prod_meta_index,
        "prod_meta_by_itemid": prod_meta_by_itemid
    }


art = load_artifacts(BASE_DIR)
if art is None:
    st.stop()

model = art["model"]
user_le = art["user_le"]
item_le = art["item_le"]
user_item_matrix = art["user_item_matrix"]
co_view_top = art["co_view_top"]
popular_items = art["popular_items"]
prod_meta_index = art["prod_meta_index"]
prod_meta_by_itemid = art["prod_meta_by_itemid"]

# ---------------------------
# Helpers
# ---------------------------
def get_meta_by_item_id(item_id):
    """Return metadata dict for an item_id (tries item_le -> code -> prod_meta_index, then prod_meta_by_itemid)."""
    # try mapping item_id -> code using item_le
    try:
        code = int(item_le.transform([item_id])[0])
        meta = prod_meta_index.get(str(code))
        if meta:
            return meta
    except Exception:
        pass
    # fallback by item_id string
    return prod_meta_by_itemid.get(str(item_id), {"title":"", "category_id":"", "brand":"", "price": ""})

def als_recommend(user_id, N):
    """Return top-N item_ids from ALS for a user_id. If user unknown return empty list."""
    try:
        if user_id not in user_le.classes_:
            return []
        u_idx = int(user_le.transform([user_id])[0])
        recs = model.recommend(u_idx, user_item_matrix, N=N)
        item_codes = [int(x[0]) for x in recs]
        return [item_le.inverse_transform([c])[0] for c in item_codes]
    except Exception:
        return []

def co_view_recommend(item_id, N):
    """Return top-N co-viewed item_ids for given item_id"""
    try:
        code = int(item_le.transform([item_id])[0])
        related = co_view_top.get(str(code), [])[:N]
        return [item_le.inverse_transform([int(c)])[0] for c in related]
    except Exception:
        return []

def filter_by_categories(candidates, allowed_categories):
    """Filter list of item_ids to those in allowed_categories (if allowed_categories is None -> return original)."""
    if allowed_categories is None:
        return candidates
    out = []
    for it in candidates:
        meta = get_meta_by_item_id(it)
        if meta.get("category_id") in allowed_categories:
            out.append(it)
    return out

def show_item_info(items):
    rows = []
    for it in items:
        meta = get_meta_by_item_id(it)
        rows.append({
            "item_id": it,
            "title": meta.get("title",""),
            "brand": meta.get("brand",""),
            "category": meta.get("category_id",""),
            "price": meta.get("price","")
        })
    return pd.DataFrame(rows)

# ---------------------------
# Load optional category relationships (if present)
# ---------------------------
category_rel_path = os.path.join(BASE_DIR, "category_relationships_many_products.csv")
category_rel_map = {}
if os.path.exists(category_rel_path):
    try:
        cr = pd.read_csv(category_rel_path)
        for _, r in cr.iterrows():
            mc = str(r["main_category"])
            rc = str(r["related_category"])
            category_rel_map.setdefault(mc, set()).add(rc)
    except Exception:
        category_rel_map = {}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🛒 Smart Product Recommendation — Demo")
st.write("Enter a user ID and a current item ID to get category-aware recommendations.")

user_id_input = st.text_input("User ID (e.g., u1)", "")
item_id_input = st.text_input("Current Item ID (e.g., p101)", "")
N = st.slider("Number of recommendations", 1, 20, 6)

if st.button("Get Recommendations"):
    # Determine allowed categories from the current item
    allowed_categories = None
    if item_id_input:
        meta = get_meta_by_item_id(item_id_input)
        curr_cat = meta.get("category_id")
        if curr_cat:
            allowed_categories = {curr_cat}
            # include related categories if mapping exists
            if curr_cat in category_rel_map:
                allowed_categories.update(category_rel_map[curr_cat])

    # Fetch expanded candidate lists (get more to allow filtering)
    als_cands = als_recommend(user_id_input, N*4)
    co_cands = co_view_recommend(item_id_input, N*4)
    pop_cands = [item_le.inverse_transform([i])[0] for i in popular_items[:N*6]]

    # Filter by allowed categories (if any)
    als_f = filter_by_categories(als_cands, allowed_categories)
    co_f = filter_by_categories(co_cands, allowed_categories)
    pop_f = filter_by_categories(pop_cands, allowed_categories)

    # Merge with priority ALS > Co-view > Popular
    final = []
    for group in (als_f, co_f, pop_f):
        for it in group:
            if it not in final:
                final.append(it)
            if len(final) >= N:
                break
        if len(final) >= N:
            break

    # If still short, append unfiltered popular items as fallback
    if len(final) < N:
        for it in pop_cands:
            if it not in final:
                final.append(it)
            if len(final) >= N:
                break

    # Display results
    st.subheader("Top Recommendations")
    st.table(show_item_info(final))

    st.subheader("Why these?")
    for it in final:
        reason = "Popular"
        if it in als_f:
            reason = "Personalized (ALS)"
        elif it in co_f:
            reason = "Co-view"
        st.write(f"- {it} — {reason}")
