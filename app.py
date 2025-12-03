import numpy as np
import pandas as pd
import streamlit as st
from math import ceil

# -------------------------------------------------------------------
# Page Setup + Style
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Bauercrest Electives Cycles Matcher",
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
# Constants
# -------------------------------------------------------------------
ELECTIVES_PER_DAY = 2        # fixed: 2 electives per day
NUM_CYCLES = 3               # Thu–Wed, repeated 3 times
TOTAL_ELECTIVES_PER_CAMPER = ELECTIVES_PER_DAY * NUM_CYCLES  # 6

# -------------------------------------------------------------------
# Header (logo + title)
# -------------------------------------------------------------------
cols = st.columns([1, 4])
with cols[0]:
    try:
        st.image("logo-header-2.png", use_container_width=True)
    except Exception:
        st.write("")

with cols[1]:
    st.title("Bauercrest Electives Cycles Matcher")
    st.subheader("Turn 10-choice ballots into 3 cycles of 2 electives per day")

st.markdown("---")

# -------------------------------------------------------------------
# Sidebar Settings
# -------------------------------------------------------------------
st.sidebar.header("Settings")
seed = st.sidebar.number_input(
    "Random seed (for tie-breaking)",
    min_value=0,
    max_value=10000,
    value=42,
    step=1,
)

# -------------------------------------------------------------------
# Template DataFrames
# -------------------------------------------------------------------
def template_campers_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "first_name": ["John", "Jack"],
        "last_name": ["Doe", "Smith"],
        "bunk": ["Bunk 4", "Bunk 16"],
        "age_group": ["Sophmore", "Junior"],
    })

def template_electives_df():
    return pd.DataFrame({
        "elective_id": ["WF", "BB", "ART"],
        "elective_name": ["Waterfront", "Basketball", "Arts & Crafts"],
        "cycle_capacity": [80, 40, 40],
    })

def template_ballots_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "rank_1": ["WF", "WF"],
        "rank_2": ["BB", "ART"],
        "rank_3": ["ART", "BB"],
        "rank_4": ["DM", "WF"],
        "rank_5": ["SBL", "DM"],
        "rank_6": ["GF", "SBL"],
        "rank_7": ["NJA", "GF"],
        "rank_8": ["WW", "NJA"],
        "rank_9": ["DIY", "WW"],
        "rank_10": ["CRD", "DIY"],
    })

# -------------------------------------------------------------------
# File Uploads
# -------------------------------------------------------------------
st.header("1️⃣ Upload Data Files")

campers_file = st.file_uploader("Upload campers.csv", type=["csv"])
ballots_file = st.file_uploader("Upload ballots.csv (10 ranked choices)", type=["csv"])
electives_file = st.file_uploader("Upload electives.csv (cycle capacities)", type=["csv"])

have_all = campers_file is not None and ballots_file is not None and electives_file is not None

if not have_all:
    st.info(
        "Upload campers.csv, ballots.csv, and electives.csv to run the matcher. "
        "Or download templates below and adapt them to match the CampMinder export."
    )

    st.subheader("campers.csv template")
    tmpl_camp = template_campers_df()
    st.dataframe(tmpl_camp)
    st.download_button(
        "Download campers.csv template",
        tmpl_camp.to_csv(index=False).encode("utf-8"),
        file_name="campers_template.csv",
        mime="text/csv",
    )

    st.subheader("electives.csv template")
    tmpl_elec = template_electives_df()
    st.dataframe(tmpl_elec)
    st.download_button(
        "Download electives.csv template",
        tmpl_elec.to_csv(index=False).encode("utf-8"),
        file_name="electives_template.csv",
        mime="text/csv",
    )

    st.subheader("ballots.csv template")
    tmpl_ballots = template_ballots_df()
    st.dataframe(tmpl_ballots)
    st.download_button(
        "Download ballots.csv template",
        tmpl_ballots.to_csv(index=False).encode("utf-8"),
        file_name="ballots_template.csv",
        mime="text/csv",
    )

else:
    campers_df = pd.read_csv(campers_file)
    ballots_df = pd.read_csv(ballots_file)
    electives_df = pd.read_csv(electives_file)

    st.success("Files uploaded successfully. Preview below:")
    with st.expander("campers.csv"):
        st.dataframe(campers_df.head())
    with st.expander("ballots.csv"):
        st.dataframe(ballots_df.head())
    with st.expander("electives.csv"):
        st.dataframe(electives_df.head())

    # -------------------------------------------------------------------
    # Validate campers.csv
    # -------------------------------------------------------------------
    required_camper_cols = {"camper_id", "first_name", "last_name", "bunk", "age_group"}
    if not required_camper_cols.issubset(set(campers_df.columns)):
        st.error(f"campers.csv must contain columns: {sorted(required_camper_cols)}")
    else:
        # Rank columns
        rank_cols = [c for c in ballots_df.columns if c.startswith("rank_")]
        if len(rank_cols) < 10:
            st.warning("ballots.csv currently has fewer than 10 rank_* columns. The matcher will use whatever exists.")
        rank_cols = sorted(rank_cols, key=lambda x: int(x.split("_")[1]))

        st.header("2️⃣ Generate 3-Cycle Assignments")

        if st.button("Build 3-cycle elective plan"):
            rng = np.random.default_rng(seed)

            # Merge campers + ballots
            merged = pd.merge(ballots_df, campers_df, on="camper_id", how="inner")
            camper_ids = merged["camper_id"].unique().tolist()

            # -----------------------------------------------------------
            # Demand per elective (how many ballots mention it anywhere)
            # -----------------------------------------------------------
            demand_counts = {}
            for _, row in ballots_df.iterrows():
                for col in rank_cols:
                    e = row[col]
                    if pd.isna(e):
                        continue
                    e = str(e).strip()
                    if e == "":
                        continue
                    demand_counts[e] = demand_counts.get(e, 0) + 1

            electives_df = electives_df.copy()

            # Accept either cycle_capacity or period_capacity
            if "cycle_capacity" not in electives_df.columns:
                if "period_capacity" in electives_df.columns:
                    electives_df = electives_df.rename(columns={"period_capacity": "cycle_capacity"})
                else:
                    st.error("electives.csv must have either 'cycle_capacity' or 'period_capacity' column.")
                    st.stop()

            electives_df["cycle_capacity"] = electives_df["cycle_capacity"].astype(int)
            electives_df["demand"] = electives_df["elective_id"].apply(
                lambda e: demand_counts.get(e, 0)
            )

            # Capacities per cycle — same capacity each cycle
            capacities = {
                cycle: {
                    row["elective_id"]: row["cycle_capacity"]
                    for _, row in electives_df.iterrows()
                }
                for cycle in range(1, NUM_CYCLES + 1)
            }

            # Assignment state
            camper_state = {
                cid: {
                    "assigned": [],  # electives across all cycles
                    "by_cycle": {cycle: [] for cycle in range(1, NUM_CYCLES + 1)},
                }
                for cid in camper_ids
            }

            # Map camper_id -> their ballot preference list (in rank order)
            ballot_map = {
                cid: [merged[merged["camper_id"] == cid].iloc[0][col] for col in rank_cols]
                for cid in camper_ids
            }

            # -----------------------------------------------------------
            # Main assignment loop
            #  For each cycle and slot, walk through campers in random order
            #  and try to give them the best remaining elective.
            # -----------------------------------------------------------
            for cycle in range(1, NUM_CYCLES + 1):
                for slot in range(ELECTIVES_PER_DAY):
                    order = camper_ids.copy()
                    rng.shuffle(order)

                    for cid in order:
                        if len(camper_state[cid]["by_cycle"][cycle]) >= ELECTIVES_PER_DAY:
                            continue

                        prefs = ballot_map[cid]
                        for pref in prefs:
                            if pd.isna(pref):
                                continue
                            elective_id = str(pref).strip()
                            if elective_id == "":
                                continue
                            if elective_id not in capacities[cycle]:
                                continue
                            if capacities[cycle][elective_id] <= 0:
                                continue
                            # camper can't repeat the same elective across cycles
                            if elective_id in camper_state[cid]["assigned"]:
                                continue

                            camper_state[cid]["assigned"].append(elective_id)
                            camper_state[cid]["by_cycle"][cycle].append(elective_id)
                            capacities[cycle][elective_id] -= 1
                            break  # move on to next camper

            # -----------------------------------------------------------
            # Build camper-cycle assignment table
            # -----------------------------------------------------------
            assignment_rows = []
            for cid in camper_ids:
                row = {"camper_id": cid}
                for cycle in range(1, NUM_CYCLES + 1):
                    cycle_electives = camper_state[cid]["by_cycle"][cycle]
                    for i in range(ELECTIVES_PER_DAY):
                        key = f"cycle{cycle}_elective{i+1}"
                        row[key] = cycle_electives[i] if i < len(cycle_electives) else ""
                assignment_rows.append(row)

            camper_cycles_df = pd.DataFrame(assignment_rows)
            camper_cycles_df = camper_cycles_df.merge(
                campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
                on="camper_id",
                how="left",
            )

            st.success("3-cycle elective assignments generated!")

            st.subheader("Camper Cycle Overview")
            st.dataframe(camper_cycles_df)

            st.download_button(
                "Download camper cycle assignments CSV",
                camper_cycles_df.to_csv(index=False).encode("utf-8"),
                file_name="camper_cycle_assignments.csv",
                mime="text/csv",
            )

            # -----------------------------------------------------------
            # Build cycle rosters
            # -----------------------------------------------------------
            roster_rows = []
            elec_name_map = dict(zip(electives_df["elective_id"], electives_df["elective_name"]))

            for _, row in camper_cycles_df.iterrows():
                cid = row["camper_id"]
                for cycle in range(1, NUM_CYCLES + 1):
                    for i in range(ELECTIVES_PER_DAY):
                        col = f"cycle{cycle}_elective{i+1}"
                        elective_id = row[col]
                        if isinstance(elective_id, str) and elective_id != "":
                            roster_rows.append({
                                "cycle": cycle,
                                "elective_id": elective_id,
                                "elective_name": elec_name_map.get(elective_id, elective_id),
                                "camper_id": cid,
                                "first_name": row["first_name"],
                                "last_name": row["last_name"],
                                "bunk": row["bunk"],
                                "age_group": row["age_group"],
                            })

            cycle_rosters_df = pd.DataFrame(roster_rows)
            if not cycle_rosters_df.empty:
                cycle_rosters_df = cycle_rosters_df.sort_values(
                    ["cycle", "elective_name", "bunk", "last_name"]
                )

            st.subheader("Rosters by Cycle & Elective")
            if cycle_rosters_df.empty:
                st.warning("No cycle rosters generated. Check capacities and ballots.")
            else:
                st.dataframe(cycle_rosters_df)
                st.download_button(
                    "Download cycle rosters CSV",
                    cycle_rosters_df.to_csv(index=False).encode("utf-8"),
                    file_name="cycle_rosters.csv",
                    mime="text/csv",
                )

            st.subheader("Elective Demand Snapshot")
            st.dataframe(
                electives_df[["elective_id", "elective_name", "cycle_capacity", "demand"]]
            )
