import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import re

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Google Ads Keyword Intelligence", layout="wide")
st.title("🚀 Google Ads Keyword Intelligence Dashboard")

st.markdown("""
Upload multiple **Google Ads Search Term Reports (CSV)**.  
This tool will let you:
- See what customers are searching (exact phrases)
- Identify top-performing keywords
- List all search terms under each keyword
- Find **search gaps** (terms not covered by your paid keywords)
- View **Search Term → Keyword Mapping**
- Get **AI-powered campaign insights**
""")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload CSV files (Google Ads exports)", 
    accept_multiple_files=True, 
    type=["csv"]
)

if not uploaded_files:
    st.warning("⬆️ Please upload your CSV reports.")
    st.stop()

# ----------------------------
# DATA LOADING & CLEANING
# ----------------------------
all_dfs = []
for file in uploaded_files:
    try:
        df = pd.read_csv(file, skiprows=2)  # Google Ads reports often have 2 metadata rows
        df["account"] = file.name
        all_dfs.append(df)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

df = pd.concat(all_dfs, ignore_index=True)

# Standardize column names
df = df.rename(columns={
    "Search term": "search_term",
    "Keyword": "keyword",
    "Campaign": "campaign",
    "Ad group": "ad_group",
    "Impr.": "impressions",
    "Interactions": "clicks",
    "Cost": "cost",
    "Match type": "match_type"
})

# Clean numbers
for col in ["impressions", "clicks", "cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df.dropna(subset=["search_term"])

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Filters")

accounts = df["account"].unique()
account_sel = st.sidebar.selectbox("Select Account", ["All"] + list(accounts))
filtered_df = df if account_sel == "All" else df[df["account"] == account_sel]

campaigns = filtered_df["campaign"].unique()
campaign_sel = st.sidebar.selectbox("Select Campaign", ["All"] + list(campaigns))
if campaign_sel != "All":
    filtered_df = filtered_df[filtered_df["campaign"] == campaign_sel]

# ----------------------------
# MAIN ANALYSIS
# ----------------------------
st.subheader("📊 Keyword Summary")

summary = (
    filtered_df.groupby("keyword")
    .agg(
        total_search_terms=("search_term", "nunique"),
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_cost=("cost", "sum")
    )
    .reset_index()
    .sort_values("total_search_terms", ascending=False)
)
summary["CTR"] = (summary["total_clicks"] / summary["total_impressions"]).fillna(0)
summary["CPC"] = (summary["total_cost"] / summary["total_clicks"]).replace([float("inf")], 0)

st.dataframe(summary, use_container_width=True)

# ----------------------------
# EXACT SEARCH TERMS FOR EACH KEYWORD
# ----------------------------
st.subheader("🔍 Exact Search Terms by Keyword")

chosen_kw = st.selectbox("Pick a keyword to see its exact search terms:", summary["keyword"].dropna().tolist())

if chosen_kw:
    sub = filtered_df[filtered_df["keyword"] == chosen_kw]
    term_details = (
        sub.groupby("search_term")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            cost=("cost", "sum")
        )
        .reset_index()
        .sort_values("clicks", ascending=False)
    )
    st.write(f"**Keyword:** `{chosen_kw}` — {len(term_details)} unique search terms")
    st.dataframe(term_details, use_container_width=True)

# ----------------------------
# TOP SEARCH TERMS (GLOBAL)
# ----------------------------
st.subheader("🔝 Top Search Terms by Clicks")
top_terms = (
    filtered_df.groupby("search_term")
    .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
    .sort_values(by="clicks", ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(10,6))
top_terms.sort_values("clicks").plot.barh(y="clicks", ax=ax, legend=False, color="skyblue")
ax.set_xlabel("Clicks")
ax.set_ylabel("Search Term")
st.pyplot(fig)

st.dataframe(top_terms, use_container_width=True)

# ----------------------------
# SEARCH GAP ANALYSIS
# ----------------------------
st.subheader("⚠️ Search Terms Not Covered by Keywords")

# Normalize text (strip quotes/brackets from keywords)
def clean_kw(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[\[\]\"]', '', str(text)).strip().lower()

filtered_df["search_term_clean"] = filtered_df["search_term"].str.lower().str.strip()
filtered_df["keyword_clean"] = filtered_df["keyword"].apply(clean_kw)

# Global sets
all_keywords = set(filtered_df["keyword_clean"].unique()) - {""}
all_search_terms = set(filtered_df["search_term_clean"].unique())
uncovered_terms = all_search_terms - all_keywords

# Filter dataset for uncovered search terms
uncovered_df = filtered_df[filtered_df["search_term_clean"].isin(uncovered_terms)].copy()

if uncovered_df.empty:
    st.info("✅ All search terms are already covered by your paid keywords!")
else:
    uncovered_summary = (
        uncovered_df.groupby("search_term")
        .agg(
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
            total_cost=("cost", "sum")
        )
        .reset_index()
        .sort_values(by=["total_clicks", "total_impressions"], ascending=[False, False])
    )
    st.write("These search terms drive traffic but are **not directly mapped to your keywords**:")
    st.markdown("### 📊 Aggregated Summary")
    st.dataframe(uncovered_summary, use_container_width=True)

    st.markdown("### 📋 Full Detailed List")
    st.dataframe(
        uncovered_df[
            ["search_term", "match_type", "campaign", "ad_group", 
             "impressions", "clicks", "cost", "keyword"]
        ].sort_values(by="clicks", ascending=False),
        use_container_width=True
    )

# ----------------------------
# SEARCH TERM → KEYWORD MAPPING
# ----------------------------
st.subheader("🔗 Search Term → Keyword Mapping")

mapping = (
    filtered_df[["search_term", "keyword", "impressions", "clicks", "cost"]]
    .sort_values(by="clicks", ascending=False)
    .reset_index(drop=True)
)

st.write("This table shows exactly **what people typed (Search Term)** and **which keyword triggered your ad**:")
st.dataframe(mapping, use_container_width=True)

# ----------------------------
# AI CAMPAIGN INSIGHTS
# ----------------------------
st.subheader("🤖 AI Campaign Insights")

if st.button("Generate AI Insights"):
    with st.spinner("⌛ Generating insights…"):
        try:
            terms_list = top_terms.index.tolist()
            campaign_name = campaign_sel if campaign_sel != "All" else "all campaigns"

            prompt = f"""
            You are a marketing strategist. These are the top search terms from campaign: {campaign_name}.
            Search terms: {terms_list}
            
            Provide actionable business suggestions:
            - What do these searches reveal about customer needs?
            - What services or content should the company improve?
            - What growth opportunities are visible?
            Keep it simple, clear, and focused ONLY on this campaign.
            """

            api_key = st.secrets["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"]

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",  # Replace with your app URL if deployed
                "X-Title": "Google Ads Dashboard"
            }

            payload = {
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You are a marketing strategist AI."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 400
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

            result = response.json()

            if "choices" in result:
                st.success(result["choices"][0]["message"]["content"])
            else:
                st.error(f"AI response error: {result}")

        except Exception as e:
            st.error(f"AI request failed: {e}")
