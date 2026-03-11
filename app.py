import streamlit as st
from datetime import date, timedelta

from config import FOOTBALL_DATA_API_KEY, GROQ_API_KEY
from db import init_db, save_predictions, get_predictions, get_last_updated
from pipeline.graph import run_pipeline, run_pipeline_averaged

from db import init_db, save_predictions, get_predictions, get_last_updated, migrate_actual_columns, get_accuracy_stats
from pipeline.graph import refresh_actual_results


st.set_page_config(
    page_title="Football Predictions",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
        margin-top: -4rem;
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

init_db()

migrate_actual_columns()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ Football Predictor")
    st.markdown("""
    AI-powered match predictions using live stats, standings, 
    and the latest news for Europe's top clubs.
    
    **Tracked teams:**
    Manchester United · Real Madrid · Barcelona ·
    Arsenal · Chelsea · Manchester City ·
    Bayern Munich · Borussia Dortmund ·
    Inter Milan · Paris Saint-Germain
    """)
    st.divider()

    selected_date = st.date_input(
        "Date",
        value=date.today(),
        min_value=date.today() - timedelta(days=7),
        max_value=date.today() + timedelta(days=7),
    )

    st.divider()

    # Config check
    if not FOOTBALL_DATA_API_KEY or not GROQ_API_KEY:
        st.error("API keys missing. Check your .env file.")
    else:
        # force_refresh = st.checkbox("Force refresh", value=False)

        if st.button("▶ Run Predictions", use_container_width=True, type="primary"):
            cached = get_predictions(selected_date.isoformat())
            if cached:
                st.info("Loaded from cache.")
                st.rerun()
            else:
                with st.spinner("Fetching matches, news, and generating predictions..."):
                    try:
                        predictions = run_pipeline_averaged(selected_date.isoformat(), runs=3)
                        if predictions:
                            save_predictions(predictions)
                            st.success(f"Done — {len(predictions)} match(es) predicted.")
                            st.rerun()
                        else:
                            st.warning("No matches found for this date.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            st.rerun()

    last_updated = get_last_updated()
    if last_updated:
        st.caption(f"Last updated: {last_updated}")

        st.divider()
    if st.button("🔁 Refresh Actual Results", use_container_width=True):
        with st.spinner("Fetching actual results..."):
            updated = refresh_actual_results()
            st.success(f"Updated {updated} match(es).")

    st.divider()
    st.markdown("**Prediction Accuracy**")
    for label, days in [("Last 7 days", 7), ("All time", None)]:
        stats = get_accuracy_stats(days=days)
        if stats:
            st.caption(f"**{label}** — {stats['total']} matches")
            st.caption(f"Outcome: **{stats['accuracy']}%** correct")
            st.caption(f"Home goals off by: **{stats['home_goal_mae']}** avg")
            st.caption(f"Away goals off by: **{stats['away_goal_mae']}** avg")
        else:
            st.caption(f"**{label}** — no data yet")

    st.divider()
    st.caption("Powered by Groq + LangGraph")
    st.caption("Built by **AK**")
    st.caption("© 2026 · All rights reserved")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Match Predictions")
st.caption(selected_date.strftime("%A, %B %d, %Y"))

predictions = get_predictions(selected_date.isoformat())

if not predictions:
    st.info("No predictions yet. Select a date and click **▶ Run Predictions**.")
    st.stop()

# Group by competition
by_competition: dict[str, list] = {}
for p in predictions:
    by_competition.setdefault(p["competition"], []).append(p)

CONF_ICON = {"low": "🔴", "medium": "🟡", "high": "🟢"}
WINNER_LABEL = {"home": lambda p: p["home_team"], "draw": lambda _: "Draw", "away": lambda p: p["away_team"]}

for competition, matches in by_competition.items():
    st.subheader(competition)

    for p in matches:
        with st.container(border=True):

            if "error" in p:
                st.warning(f"{p['home_team']} vs {p['away_team']} — prediction failed: {p['error']}")
                continue

            # Team names + predicted winner
            col_home, col_mid, col_away = st.columns([3, 4, 3])
            with col_home:
                st.markdown(f"### {p['home_team']}")
            with col_mid:
                winner = WINNER_LABEL[p["predicted_winner"]](p)
                st.markdown(f"<div style='text-align:center; font-size:0.85rem; color:gray'>Predicted winner</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-weight:bold'>{winner}</div>", unsafe_allow_html=True)
            with col_away:
                st.markdown(f"<div style='text-align:right'>### {p['away_team']}</div>", unsafe_allow_html=True)

            st.write("")

            # Probability bars
            bar_col1, bar_col2, bar_col3 = st.columns(3)
            with bar_col1:
                st.caption("Home win")
                st.progress(p["home_win_prob"])
                st.caption(f"{p['home_win_prob']*100:.1f}%")
            with bar_col2:
                st.caption("Draw")
                st.progress(p["draw_prob"])
                st.caption(f"{p['draw_prob']*100:.1f}%")
            with bar_col3:
                st.caption("Away win")
                st.progress(p["away_win_prob"])
                st.caption(f"{p['away_win_prob']*100:.1f}%")

            # Confidence
            conf = p.get("confidence", "low")
            st.caption(f"{CONF_ICON.get(conf, '⚪')} Confidence: **{conf.upper()}**")

            # Predicted score
            hg = p.get("home_goals", 0)
            ag = p.get("away_goals", 0)
            st.markdown(
                f"<div style='text-align:center; font-size:1.3rem; font-weight:bold'>"
                f"⚽ {hg:.1f} — {ag:.1f}</div>",
                unsafe_allow_html=True
            )

            # Actual result (if available)
            actual_hg = p.get("actual_home_goals")
            actual_ag = p.get("actual_away_goals")
            if actual_hg is not None and actual_ag is not None:
                correct = p.get("predicted_winner") == p.get("actual_winner")
                icon = "✅" if correct else "❌"
                st.markdown(
                    f"<div style='text-align:center; font-size:1rem;'>"
                    f"**Final Score:** {p['home_team']} "
                    f"**{int(actual_hg)} — {int(actual_ag)}** "
                    f"{p['away_team']} {icon}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("Final score not available yet.")

            # Expandable analysis
            with st.expander("Analysis"):
                factors = p.get("key_factors", [])
                if factors:
                    st.markdown("**Key factors:**")
                    for f in factors:
                        st.markdown(f"- {f}")
                preview = p.get("match_preview", "")
                if preview:
                    st.markdown("**Preview:**")
                    st.write(preview)

    st.write("")
