import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

# --------------------------
# Page config & basic styling
# --------------------------
st.set_page_config(
    page_title="Bauercrest Electives Ballot Matcher",
    page_icon="🏕️",
    layout="wide",
)

PRIMARY_COLOR = "#150b4f"

CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: white;
    }}
    h1, h2, h3, h4, h5, h6 {{
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
# --------------------------
# Header with Bauercrest logo
# --------------------------



# --------------------------
# Helper constants
# --------------------------
AGE_GROUP_ORDER = {
    "Freshman": 1,
    "Sophomore": 2,
    "Junior": 3,
    "Senior": 4,
    "Waiter": 5,
    "CI": 6,
}

AGE_GROUP_ORDER_INV = {v: k for k, v in AGE_GROUP_ORDER.items()}


# --------------------------
# Matching logic
# --------------------------
def run_matching(campers_df, ballots_df, electives_df, slots_per_camper=3, seed=42):
    rng = np.random.default_rng(seed)

    # Merge campers + ballots
    df = pd.merge(ballots_df, campers_df, on="camper_id", how="inner")

    # Clean electives dict
    electives_df = electives_df.copy()
    electives_df["remaining_capacity"] = electives_df["daily_capacity"].astype(int)

    electives_dict = {
        row["elective_id"]: row for _, row in electives_df.iterrows()
    }

    # Prepare result structure
    assignments = []

    # Work per age group for fairness
    for age_group, group_df in df.groupby("age_group"):
        if age_group not in AGE_GROUP_ORDER:
            continue

        # Shuffle campers once for fairness, reproducibly
        camper_ids = group_df["camper_id"].unique().tolist()
        rng.shuffle(camper_ids)

        # Track camper assignment state
        camper_state = {
            cid: {
                "assigned": [],
                "best_rank": None,
            }
            for cid in camper_ids
        }

        # Preference ranks from ballot columns
        rank_cols = [c for c in ballots_df.columns if c.startswith("rank_")]
        rank_cols_sorted = sorted(rank_cols, key=lambda x: int(x.split("_")[1]))

        for rank_idx, rank_col in enumerate(rank_cols_sorted, start=1):
            # Fairness enhancement:
            # sort campers by worst best_rank (None considered high)
            def camper_sort_key(cid):
                best = camper_state[cid]["best_rank"]
                return (999 if best is None else best, rng.integers(0, 1_000_000))

            camper_ids_sorted = sorted(camper_ids, key=camper_sort_key)

            for cid in camper_ids_sorted:
                state = camper_state[cid]
                if len(state["assigned"]) >= slots_per_camper:
                    continue

                row = group_df[group_df["camper_id"] == cid].iloc[0]
                elective_id = row[rank_col]

                if pd.isna(elective_id) or elective_id == "":
                    continue
                if elective_id not in electives_dict:
                    continue

                # Already assigned this elective?
                if elective_id in state["assigned"]:
                    continue

                elective_row = electives_dict[elective_id]
                remaining = elective_row["remaining_capacity"]

                if remaining <= 0:
                    continue

                # Age group bounds
                camper_age_rank = AGE_GROUP_ORDER[age_group]
                min_rank = int(elective_row["min_age_group_rank"])
                max_rank = int(elective_row["max_age_group_rank"])

                if not (min_rank <= camper_age_rank <= max_rank):
                    continue

                # Assign
                state["assigned"].append(elective_id)
                electives_dict[elective_id]["remaining_capacity"] -= 1
                if state["best_rank"] is None or rank_idx < state["best_rank"]:
                    state["best_rank"] = rank_idx

        # Flush assignments for this age group
        for cid in camper_ids:
            state = camper_state[cid]
            for slot_index, elective_id in enumerate(state["assigned"], start=1):
                assignments.append(
                    {
                        "camper_id": cid,
                        "age_group": age_group,
                        "slot": slot_index,
                        "elective_id": elective_id,
                        "best_rank": state["best_rank"],
                    }
                )

    assignments_df = pd.DataFrame(assignments)

    # Join elective names for readability
    if not assignments_df.empty:
        assignments_df = assignments_df.merge(
            electives_df[["elective_id", "elective_name"]],
            on="elective_id",
            how="left",
        )

    return assignments_df, electives_dict


def compute_summary(assignments_df, ballots_df):
    if assignments_df.empty:
        return {}

    # Build camper -> min rank actually assigned
    best_rank_by_camper = (
        assignments_df.groupby("camper_id")["best_rank"].min().to_dict()
    )

    n_campers = len(best_rank_by_camper)
    if n_campers == 0:
        return {}

    n_top1 = sum(1 for r in best_rank_by_camper.values() if r == 1)
    n_top3 = sum(1 for r in best_rank_by_camper.values() if r is not None and r <= 3)

    return {
        "n_campers": n_campers,
        "pct_top1": 100.0 * n_top1 / n_campers,
        "pct_top3": 100.0 * n_top3 / n_campers,
    }


# --------------------------
# Template generators
# --------------------------
def get_template_campers():
    data = [
        ["C001", "Jake", "Rosen", "Bunk 1", "Freshman"],
        ["C002", "Noah", "Levy", "Bunk 3", "Sophomore"],
        ["C003", "Eli", "Klein", "Bunk 11", "Junior"],
        ["C004", "Max", "Gold", "Bunk 19", "Senior"],
        ["C005", "Sam", "Fisher", "Bunk 18", "Waiter"],
        ["C006", "Ben", "Cohen", "Bunk 20", "CI"],
    ]
    cols = ["camper_id", "first_name", "last_name", "bunk", "age_group"]
    return pd.DataFrame(data, columns=cols)


def get_template_electives():
    data = [
        ["WF", "Waterfront", 60, 1, 6],
        ["BB", "Basketball", 40, 1, 6],
        ["TEN", "Tennis", 24, 2, 6],
        ["ART", "Arts & Crafts", 30, 1, 6],
        ["CRD", "Cards & Reading", 25, 1, 6],
        ["FIT", "Fitness", 30, 3, 6],
    ]
    cols = [
        "elective_id",
        "elective_name",
        "daily_capacity",
        "min_age_group_rank",
        "max_age_group_rank",
    ]
    return pd.DataFrame(data, columns=cols)


def get_template_ballots():
    data = [
        ["C001", "WF", "BB", "TEN", "ART", "CRD", "FIT"],
        ["C002", "WF", "TEN", "BB", "FIT", "CRD", "ART"],
        ["C003", "BB", "WF", "TEN", "FIT", "ART", "CRD"],
        ["C004", "FIT", "BB", "WF", "TEN", "ART", "CRD"],
        ["C005", "WF", "FIT", "BB", "CRD", "ART", "TEN"],
        ["C006", "TEN", "WF", "FIT", "BB", "ART", "CRD"],
    ]
    cols = ["camper_id", "rank_1", "rank_2", "rank_3", "rank_4", "rank_5", "rank_6"]
    return pd.DataFrame(data, columns=cols)


# --------------------------
# UI
# --------------------------

# Header with logo
cols = st.columns([1, 4])
with cols[0]:
    try:
        st.image("assets/bauercrest_logo.png", use_container_width=True)
    except Exception:
        st.write(" ")

with cols[1]:
    st.title("Bauercrest Electives Ballot Matcher")
    st.subheader("Preference-based elective assignments")


st.markdown("---")

st.sidebar.header("Settings")
slots_per_camper = st.sidebar.number_input(
    "Electives per camper per day",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
)
seed = st.sidebar.number_input(
    "Random seed (for fairness & reproducibility)",
    min_value=0,
    max_value=10_000,
    value=42,
    step=1,
)


st.header("1️⃣ Upload Data Files")

campers_file = st.file_uploader("Upload campers.csv", type=["csv"])
ballots_file = st.file_uploader("Upload ballots.csv", type=["csv"])
electives_file = st.file_uploader("Upload electives.csv", type=["csv"])

have_all_files = campers_file is not None and ballots_file is not None and electives_file is not None

if have_all_files:
    campers_df = pd.read_csv(campers_file)
    ballots_df = pd.read_csv(ballots_file)
    electives_df = pd.read_csv(electives_file)

    st.success("Files uploaded successfully. Preview below:")

    with st.expander("Preview campers.csv"):
        st.dataframe(campers_df.head())

    with st.expander("Preview ballots.csv"):
        st.dataframe(ballots_df.head())

    with st.expander("Preview electives.csv"):
        st.dataframe(electives_df.head())

    st.header("2️⃣ Run Matching")

    if st.button("Generate Elective Assignments"):
        with st.spinner("Running ballot matcher..."):
            assignments_df, electives_state = run_matching(
                campers_df, ballots_df, electives_df, slots_per_camper, seed
            )

        if assignments_df.empty:
            st.error("No assignments were generated. Check capacities and ballots.")
        else:
            st.success("Assignments generated!")

            # Summary
            summary = compute_summary(assignments_df, ballots_df)
            if summary:
                c1, c2, c3 = st.columns(3)
                c1.metric("Campers with assignments", summary["n_campers"])
                c2.metric("Got #1 choice", f"{summary['pct_top1']:.1f}%")
                c3.metric("Got Top 3 choice", f"{summary['pct_top3']:.1f}%")

            st.subheader("Assignments by Camper")

            # Pivot to wide for download/readability
            wide_assignments = (
                assignments_df
                .sort_values(["camper_id", "slot"])
                .pivot_table(
                    index="camper_id",
                    columns="slot",
                    values="elective_name",
                    aggfunc="first"
                )
            )
            wide_assignments.columns = [f"slot_{c}" for c in wide_assignments.columns]
            wide_assignments = wide_assignments.reset_index()
            wide_assignments = wide_assignments.merge(
                campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
                on="camper_id",
                how="left",
            )

            st.dataframe(wide_assignments)

            csv_camper = wide_assignments.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download camper assignments CSV",
                data=csv_camper,
                file_name="camper_assignments.csv",
                mime="text/csv",
            )
                        
            st.subheader("Rosters by Elective")

            # Merge camper info onto assignments
            elective_rosters = assignments_df.merge(
                campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
                on="camper_id",
                how="left",
            )

            # Only sort by columns that actually exist (avoids KeyError)
            sort_cols = []
            for col in ["elective_name", "age_group", "bunk", "last_name"]:
                if col in elective_rosters.columns:
                    sort_cols.append(col)

            if sort_cols:
                elective_rosters = elective_rosters.sort_values(sort_cols)

            st.dataframe(elective_rosters)

            csv_rosters = elective_rosters.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download elective rosters CSV",
                data=csv_rosters,
                file_name="elective_rosters.csv",
                mime="text/csv",
            )

            st.subheader("Elective Capacity Usage")
            elective_usage = (
                elective_rosters.groupby(["elective_id", "elective_name"])
                .agg(assigned=("camper_id", "count"))
                .reset_index()
            )
            elective_usage = elective_usage.merge(
                electives_df[["elective_id", "daily_capacity"]],
                on="elective_id",
                how="left",
            )
            elective_usage["remaining"] = (
                elective_usage["daily_capacity"] - elective_usage["assigned"]
            )
            st.dataframe(elective_usage)

else:
    st.info(
        "Upload campers.csv, ballots.csv, and electives.csv to run the matcher. "
        "Or download editable templates below to get started."
    )

    st.header("📁 CSV Templates")

    tmpl_campers = get_template_campers()
    tmpl_ballots = get_template_ballots()
    tmpl_electives = get_template_electives()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("campers.csv template")
        st.caption("Required columns: camper_id, first_name, last_name, bunk, age_group")
        st.dataframe(tmpl_campers)
        campers_csv = tmpl_campers.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download campers.csv template",
            data=campers_csv,
            file_name="campers_template.csv",
            mime="text/csv",
        )

    with c2:
        st.subheader("ballots.csv template")
        st.caption("Required columns: camper_id, rank_1, rank_2, rank_3, rank_4, rank_5, rank_6")
        st.dataframe(tmpl_ballots)
        ballots_csv = tmpl_ballots.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ballots.csv template",
            data=ballots_csv,
            file_name="ballots_template.csv",
            mime="text/csv",
        )

    st.subheader("electives.csv template")
    st.caption(
        "Required columns: elective_id, elective_name, daily_capacity, "
        "min_age_group_rank, max_age_group_rank "
        "(1=Freshman, 2=Sophomore, 3=Junior, 4=Senior, 5=Waiter, 6=CI)"
    )
    st.dataframe(tmpl_electives)
    electives_csv = tmpl_electives.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download electives.csv template",
        data=electives_csv,
        file_name="electives_template.csv",
        mime="text/csv",
    )
