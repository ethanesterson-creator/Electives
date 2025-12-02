import numpy as np
import pandas as pd
import streamlit as st
from math import ceil
from io import StringIO

# -------------------------------------------------------------------
# Page Setup + Style
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bauercrest Electives Matcher",
    page_icon="🏕️",
    layout="wide",
)

PRIMARY_COLOR = "#150b4f"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: white;
    }}
    h1, h2, h3, h4, h5 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# HEADER (logo optional)
# -------------------------------------------------------------------
cols = st.columns([1, 4])
with cols[0]:
    try:
        st.image("logo-header-2.png", use_container_width=True)
    except:
        st.write("")

with cols[1]:
    st.title("Bauercrest Electives Matcher")
    st.subheader("Assign electives based on ranked ballots)

st.markdown("---")

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
st.sidebar.header("Settings")
slots_per_camper = st.sidebar.number_input(
    "Electives per camper per day",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
)
seed = st.sidebar.number_input(
    "Random Seed (for reproducibility)",
    min_value=0,
    max_value=10000,
    value=42,
)

# -------------------------------------------------------------------
# TEMPLATE DATAFRAMES
# -------------------------------------------------------------------
def template_campers_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "first_name": ["Jake", "Noah"],
        "last_name": ["Rosen", "Levy"],
        "bunk": ["Bunk 1", "Bunk 3"],
        "age_group": ["Junior", "Junior"]
    })

def template_electives_df():
    return pd.DataFrame({
        "elective_id": ["WF", "BB"],
        "elective_name": ["Waterfront", "Basketball"],
        "period_capacity": [25, 10]
    })

def template_ballots_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "rank_1": ["WF", "WF"],
        "rank_2": ["BB", "BB"],
        "rank_3": ["TEN", "ART"],
        "rank_4": ["ART", "TEN"],
        "rank_5": ["DIY", "CRD"],
        "rank_6": ["CRD", "DIY"]
    })

# -------------------------------------------------------------------
# FILE UPLOADS
# -------------------------------------------------------------------
st.header("1️⃣ Upload Data Files")

campers_file = st.file_uploader("Upload campers.csv", type=["csv"])
ballots_file = st.file_uploader("Upload ballots.csv", type=["csv"])
electives_file = st.file_uploader("Upload electives.csv", type=["csv"])

have_all = campers_file and ballots_file and electives_file

if not have_all:
    st.info("Upload all three CSVs to run the matcher. Otherwise, download templates below.")

    st.subheader("campers.csv Template")
    st.dataframe(template_campers_df())
    st.download_button(
        "Download campers.csv template",
        template_campers_df().to_csv(index=False).encode(),
        file_name="campers_template.csv"
    )

    st.subheader("electives.csv Template")
    st.dataframe(template_electives_df())
    st.download_button(
        "Download electives.csv template",
        template_electives_df().to_csv(index=False).encode(),
        file_name="electives_template.csv"
    )

    st.subheader("ballots.csv Template")
    st.dataframe(template_ballots_df())
    st.download_button(
        "Download ballots.csv template",
        template_ballots_df().to_csv(index=False).encode(),
        file_name="ballots_template.csv"
    )

else:
    campers_df = pd.read_csv(campers_file)
    ballots_df = pd.read_csv(ballots_file)
    electives_df = pd.read_csv(electives_file)

    st.success("Files uploaded successfully!")
    with st.expander("Preview campers.csv"):
        st.dataframe(campers_df)
    with st.expander("Preview ballots.csv"):
        st.dataframe(ballots_df)
    with st.expander("Preview electives.csv"):
        st.dataframe(electives_df)

    # -------------------------------------------------------------------
    # RUN MATCHER
    # -------------------------------------------------------------------
    st.header("2️⃣ Run Matcher")

    if st.button("Generate Elective Assignments"):
        rng = np.random.default_rng(seed)

        # STEP 1 — Calculate demand
        rank_cols = [c for c in ballots_df.columns if c.startswith("rank_")]
        demand_counts = {}
        for _, row in ballots_df.iterrows():
            for c in rank_cols:
                e = row[c]
                if pd.isna(e): continue
                demand_counts[e] = demand_counts.get(e, 0) + 1

        electives_df["demand"] = electives_df["elective_id"].apply(lambda x: demand_counts.get(x, 0))

        # STEP 2 — Compute number of periods (max 2)
        def periods_needed(demand, cap):
            return min(2, max(1, ceil(demand / cap)))

        electives_df["periods"] = electives_df.apply(
            lambda row: periods_needed(row["demand"], row["period_capacity"]),
            axis=1
        )

        electives_df["total_capacity"] = electives_df["periods"] * electives_df["period_capacity"]
        electives_df["remaining_capacity"] = electives_df["total_capacity"]

        electives_dict = {row["elective_id"]: row for _, row in electives_df.iterrows()}

        # STEP 3 — Merge camper/ballot info
        df = pd.merge(ballots_df, campers_df, on="camper_id", how="inner")

        # STEP 4 — Assignment
        assignments = []
        camper_state = {
            cid: {"assigned": [], "best_rank": None}
            for cid in df["camper_id"].unique().tolist()
        }

        for rank_idx, rank_col in enumerate(rank_cols, start=1):
            camper_ids = df["camper_id"].unique().tolist()
            rng.shuffle(camper_ids)

            for cid in camper_ids:
                if len(camper_state[cid]["assigned"]) >= slots_per_camper:
                    continue

                elective_id = df[df["camper_id"] == cid].iloc[0][rank_col]
                if pd.isna(elective_id): continue
                if elective_id not in electives_dict: continue

                if elective_id in camper_state[cid]["assigned"]: continue

                row = electives_dict[elective_id]
                if row["remaining_capacity"] <= 0: continue

                # Assign
                camper_state[cid]["assigned"].append(elective_id)
                electives_dict[elective_id]["remaining_capacity"] -= 1

                if camper_state[cid]["best_rank"] is None or rank_idx < camper_state[cid]["best_rank"]:
                    camper_state[cid]["best_rank"] = rank_idx

        # STEP 5 — Build assignments dataframe
        for cid, state in camper_state.items():
            for slot_index, elective_id in enumerate(state["assigned"], start=1):
                assignments.append({
                    "camper_id": cid,
                    "slot": slot_index,
                    "elective_id": elective_id,
                    "best_rank": state["best_rank"]
                })

        assignments_df = pd.DataFrame(assignments)
        assignments_df = assignments_df.merge(
            electives_df[["elective_id", "elective_name"]],
            on="elective_id",
            how="left"
        )

        st.success("Assignments generated!")

        # -------------------------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------------------------
        st.subheader("Assignments by Camper")
        wide = assignments_df.pivot_table(
            index="camper_id", columns="slot", values="elective_name", aggfunc="first"
        )
        wide.columns = [f"slot_{c}" for c in wide.columns]
        wide = wide.reset_index()
        wide = wide.merge(
            campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
            on="camper_id",
            how="left",
        )
        st.dataframe(wide)

        st.download_button(
            "Download camper assignments CSV",
            wide.to_csv(index=False).encode(),
            file_name="camper_assignments.csv"
        )

        st.subheader("Rosters by Elective")
        rosters = assignments_df.merge(
            campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
            on="camper_id",
            how="left",
        )

        sort_cols = [col for col in ["elective_name","bunk","last_name"] if col in rosters.columns]
        rosters = rosters.sort_values(sort_cols)

        st.dataframe(rosters)

        st.download_button(
            "Download elective rosters CSV",
            rosters.to_csv(index=False).encode(),
            file_name="elective_rosters.csv"
        )

        st.subheader("Elective Capacity Usage")
        st.dataframe(electives_df[[
            "elective_id","elective_name","demand","period_capacity","periods",
            "total_capacity","remaining_capacity"
        ]])
