import streamlit as st
import joblib, json, os, pandas as pd

BASE_DIR = "./data"

# -------------------------------
# LOAD ALL ARTIFACTS SAFELY
# -------------------------------
@st.cache_resource
def load_artifacts(base_dir):
    required_files = [
        "als_model.joblib",
        "user_le.joblib",
        "item_le.joblib",
        "user_item_matrix.joblib",
        "item_user_matrix.joblib",
        "prod_meta.csv",
        "co_view_top.json",
        "popular_items.joblib"
    ]

    # Check missing files
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
        prod_meta = pd.read_csv(os.path.join(base_dir, "prod_meta.csv"))
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

    # Ensure item_code exists
    if "item_code" not in prod_meta.columns:
        st.error("prod_meta.csv must contain a column named 'item_code'.")
        return None

    prod_meta = prod_meta.set_index("item_code").to_dict(orient="index")

    return model, user_le, item_le, user_item_matrix, co_view_top, popular_items, prod_meta


artifacts = load_artifacts(BASE_DIR)
if artifacts is None:
    st.stop()

model, user_le, item_le, user_item_matrix, co_view_top, popular_items, prod_meta = artifacts

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🛒 Smart Product Recommendation System")
st.write("Enter a user ID and product ID to generate recommendations.")

user_id_input = st.text_input("User ID (e.g., u1):")
item_id_input = st.text_input("Current Item ID (e.g., p101):")
N = st.slider("Number of Recommendations", 1, 20, 6)


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def als_recommend(user_id, N):
    """ALS-based recommendations"""
    if user_id not in user_le.classes_:
        return []
    u_idx = int(user_le.transform([user_id])[0])
    recs = model.recommend(u_idx, user_item_matrix, N=N)
    item_codes = [int(x[0]) for x in recs]
    return [item_le.inverse_transform([c])[0] for c in item_codes]


def co_view_recommend(item_id, N):
    """Co-view based fallback recommendations"""
    try:
        code = int(item_le.transform([item_id])[0])
    except:
        return []
    related_codes = co_view_top.get(str(code), [])[:N]
    return [item_le.inverse_transform([c])[0] for c in related_codes]


def show_item_info(items):
    """Display metadata table"""
    rows = []
    for it in items:
        try:
            code = int(item_le.transform([it])[0])
            meta = prod_meta.get(code, {})
            rows.append({
                "item_id": it,
                "title": meta.get("title", ""),
                "category": meta.get("category_id", ""),
                "brand": meta.get("brand", ""),
                "price": meta.get("price", "")
            })
        except:
            rows.append({"item_id": it})
    return pd.DataFrame(rows)


# -------------------------------
# ACTION BUTTON
# -------------------------------
if st.button("Get Recommendations"):
    als_list = als_recommend(user_id_input, N)
    co_list = co_view_recommend(item_id_input, N)
    pop_list = [item_le.inverse_transform([i])[0] for i in popular_items[:N]]

    final = []

    # Merge ALS → Co-view → Popular
    for group in [als_list, co_list, pop_list]:
        for it in group:
            if it not in final:
                final.append(it)
            if len(final) >= N:
                break

    st.subheader("Recommended Products")
    st.table(show_item_info(final))

    st.subheader("Why these recommendations?")
    for it in final:
        if it in als_list:
            st.write(f"👉 {it} — Personalized (ALS)")
        elif it in co_list:
            st.write(f"👉 {it} — Related to product you're viewing")
        else:
            st.write(f"👉 {it} — Popular fallback")
