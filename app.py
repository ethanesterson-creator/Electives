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
        "first_name": ["Jake", "Noah"],
        "last_name": ["Rosen", "Levy"],
        "bunk": ["Bunk 1", "Bunk 3"],
        "age_group": ["Freshman", "Junior"],
    })

def template_electives_df():
    return pd.DataFrame({
        "elective_id": ["WBK", "BBSK", "FWTB"],
        "elective_name": ["Wakeboarding", "Basketball Skills", "Fun with the Boys"],
        "cycle_capacity": [40, 40, 60],
    })

def template_ballots_df():
    return pd.DataFrame({
        "camper_id": ["C001", "C002"],
        "rank_1": ["WBK", "WSK"],
        "rank_2": ["BBSK", "TUB"],
        "rank_3": ["FWTB", "SAIL"],
        "rank_4": ["SOCC", "BBSK"],
        "rank_5": ["DM", "NJA"],
        "rank_6": ["PHOT", "DM"],
        "rank_7": ["YRBK", "IMPR"],
        "rank_8": ["ULT", "GLF"],
        "rank_9": ["WOOD", "CERM"],
        "rank_10": ["CMPC", "FWTB"],
    })

# -------------------------------------------------------------------
# Helper: compute satisfaction stats & problem campers
# -------------------------------------------------------------------
def compute_preference_stats(camper_ids, camper_state, ballot_map):
    records = []
    for cid in camper_ids:
        assigned = camper_state[cid]["assigned"]
        prefs = [str(p).strip() for p in ballot_map[cid] if not pd.isna(p) and str(p).strip() != ""]
        ranks = []
        for e in assigned:
            if e in prefs:
                ranks.append(prefs.index(e) + 1)  # 1-based rank
        best_rank = min(ranks) if ranks else None
        records.append({
            "camper_id": cid,
            "assigned_count": len(assigned),
            "best_rank": best_rank,
            "ranks": ranks,
        })

    n = len(records)
    if n == 0:
        return {}, pd.DataFrame()

    got_rank1 = sum(1 for r in records if r["best_rank"] == 1)
    got_top3 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 3)
    got_top5 = sum(1 for r in records if r["best_rank"] is not None and r["best_rank"] <= 5)

    best_ranks_nonnull = [r["best_rank"] for r in records if r["best_rank"] is not None]
    avg_best = sum(best_ranks_nonnull) / len(best_ranks_nonnull) if best_ranks_nonnull else None
    avg_assigned = sum(r["assigned_count"] for r in records) / n

    stats = {
        "n_campers": n,
        "pct_got_rank1": 100.0 * got_rank1 / n,
        "pct_got_top3": 100.0 * got_top3 / n,
        "pct_got_top5": 100.0 * got_top5 / n,
        "avg_best_rank": avg_best,
        "avg_assigned": avg_assigned,
    }

    # Problem campers: fewer than TOTAL_ELECTIVES_PER_CAMPER assigned
    problem_rows = [
        {
            "camper_id": r["camper_id"],
            "assigned_count": r["assigned_count"],
            "missing_slots": TOTAL_ELECTIVES_PER_CAMPER - r["assigned_count"],
            "best_rank": r["best_rank"],
        }
        for r in records
        if r["assigned_count"] < TOTAL_ELECTIVES_PER_CAMPER
    ]
    problem_df = pd.DataFrame(problem_rows)

    return stats, problem_df

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

    # Validate campers.csv
    required_camper_cols = {"camper_id", "first_name", "last_name", "bunk", "age_group"}
    if not required_camper_cols.issubset(set(campers_df.columns)):
        st.error(f"campers.csv must contain columns: {sorted(required_camper_cols)}")
    else:
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

            # Demand per elective
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
            electives_df["demand"] = electives_df["elective_id"].apply(lambda e: demand_counts.get(e, 0))

            # Capacities per cycle, compressing electives into as few cycles as needed
            capacities = {cycle: {} for cycle in range(1, NUM_CYCLES + 1)}
            for _, row in electives_df.iterrows():
                e = row["elective_id"]
                cap = row["cycle_capacity"]
                demand = row["demand"]
                if cap <= 0:
                    continue
                cycles_needed = min(NUM_CYCLES, max(1, ceil(demand / cap))) if demand > 0 else 1
                for cycle in range(1, cycles_needed + 1):
                    capacities[cycle][e] = cap

            # Assignment state
            camper_state = {
                cid: {
                    "assigned": [],
                    "by_cycle": {cycle: [] for cycle in range(1, NUM_CYCLES + 1)},
                }
                for cid in camper_ids
            }

            ballot_map = {
                cid: [merged[merged["camper_id"] == cid].iloc[0][col] for col in rank_cols]
                for cid in camper_ids
            }

            # Main assignment loop
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
                            if elective_id in camper_state[cid]["assigned"]:
                                continue
                            camper_state[cid]["assigned"].append(elective_id)
                            camper_state[cid]["by_cycle"][cycle].append(elective_id)
                            capacities[cycle][elective_id] -= 1
                            break

            # Camper-cycle assignment table
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

            # Preference satisfaction stats & problem campers
            stats, problem_df = compute_preference_stats(camper_ids, camper_state, ballot_map)

            st.subheader("Preference Satisfaction Summary")
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Campers", stats["n_campers"])
                c2.metric("Got ≥1 #1 choice", f"{stats['pct_got_rank1']:.1f}%")
                c3.metric("Got ≥1 Top 3", f"{stats['pct_got_top3']:.1f}%")
                c4.metric("Got ≥1 Top 5", f"{stats['pct_got_top5']:.1f}%")
                st.caption(
                    f"Average best rank: {stats['avg_best_rank']:.2f} | "
                    f"Average electives assigned: {stats['avg_assigned']:.2f} (max {TOTAL_ELECTIVES_PER_CAMPER})"
                )
            else:
                st.write("No stats available.")

            st.subheader("Campers with fewer than 6 electives")
            if problem_df.empty:
                st.success("All campers received 6 electives from their ballots.")
            else:
                problem_df = problem_df.merge(
                    campers_df[["camper_id", "first_name", "last_name", "bunk", "age_group"]],
                    on="camper_id",
                    how="left",
                )
                st.warning("Some campers have fewer than 6 electives. Review and adjust capacities/ballots.")
                st.dataframe(problem_df)

            st.subheader("Camper Cycle Overview")
            st.dataframe(camper_cycles_df)
            st.download_button(
                "Download camper cycle assignments CSV",
                camper_cycles_df.to_csv(index=False).encode("utf-8"),
                file_name="camper_cycle_assignments.csv",
                mime="text/csv",
            )

            # Cycle rosters
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
