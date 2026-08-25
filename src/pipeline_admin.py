"""
pipeline_admin.py
Standalone admin tool for controlling and monitoring the data pipeline.
Separate from dashboard.py (which is the public-facing river health view).

Run:  streamlit run src/pipeline_admin.py

This app can:
    - Show scheduler running/not running status
    - Manually trigger a POOPy pull + live inference cycle
    - Show full pipeline run history
    - Start/stop the scheduler process
"""

import streamlit as st
import pandas as pd
import subprocess
import os
import signal
from db_loader import get_conn

st.set_page_config(page_title="Pipeline Admin", page_icon="⚙️", layout="wide")

st.title("⚙️ Freshwater Risk - Pipeline Admin")
st.caption("Internal tool for monitoring and controlling the live data pipeline. Not for public use.")


def get_scheduler_pid():
    """Return the scheduler's process ID if running, else None."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "src/scheduler.py"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        return int(pids[0]) if pids and pids[0] else None
    except Exception:
        return None


# ---- Scheduler control ----------------------------------------------------------

st.header("Scheduler control")

pid = get_scheduler_pid()
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if pid:
        st.success(f"🟢 Scheduler is running (PID {pid})")
    else:
        st.warning("🟡 Scheduler is not running")

with col2:
    if st.button("▶️ Start scheduler", disabled=(pid is not None), use_container_width=True):
        subprocess.Popen(
            ["nohup", "python3", "src/scheduler.py"],
            stdout=open("logs/scheduler_out.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        st.success("Scheduler started. Refresh in a few seconds to confirm.")

with col3:
    if st.button("⏹️ Stop scheduler", disabled=(pid is None), use_container_width=True):
        if pid:
            os.kill(pid, signal.SIGTERM)
            st.success("Scheduler stopped.")

st.markdown("---")

# ---- Manual trigger ---------------------------------------------------------------

st.header("Manual actions")
st.caption("Trigger a pipeline step immediately, without waiting for the schedule.")

m1, m2, m3 = st.columns(3)

with m1:
    if st.button("🌊 Pull POOPy data now", use_container_width=True):
        with st.spinner("Pulling live sewage discharge data from all water companies..."):
            result = subprocess.run(
                ["python3", "src/pull_poopy_now.py"],
                capture_output=True, text=True, timeout=180
            )
            st.code(result.stdout[-2000:] if result.stdout else result.stderr[-2000:])

with m2:
    if st.button("🔮 Run live re-prediction", use_container_width=True):
        with st.spinner("Re-predicting locations affected by new sewage data..."):
            result = subprocess.run(
                ["python3", "src/live_inference.py"],
                capture_output=True, text=True, timeout=180
            )
            st.code(result.stdout[-2000:] if result.stdout else result.stderr[-2000:])

with m3:
    if st.button("💧 Check FreshWater Watch", use_container_width=True):
        with st.spinner("Checking for new FreshWater Watch data..."):
            result = subprocess.run(
                ["python3", "src/pull_fww_weekly.py"],
                capture_output=True, text=True, timeout=180
            )
            st.code(result.stdout[-2000:] if result.stdout else result.stderr[-2000:])

st.markdown("---")

# ---- Run history -----------------------------------------------------------------

st.header("Pipeline run history")

try:
    conn = get_conn()
    log_df = pd.read_sql("""
        SELECT source, status, records_fetched, records_new, error_msg, run_at
        FROM ingestion_log
        ORDER BY run_at DESC
        LIMIT 50
    """, conn)
    conn.close()

    if not log_df.empty:
        def status_icon(s):
            return {"success": "✅", "no_new_data": "⏸️", "failed": "❌", "skipped": "⏭️"}.get(s, "⚠️")

        log_df.insert(0, "", log_df["status"].apply(status_icon))
        st.dataframe(log_df, hide_index=True, use_container_width=True, height=500)
    else:
        st.info("No pipeline runs logged yet.")
except Exception as e:
    st.error(f"Could not load run history: {e}")

st.markdown("---")

# ---- Live data summary -------------------------------------------------------------

st.header("Live data summary")

try:
    conn = get_conn()
    summary = pd.read_sql("""
        SELECT company, status, COUNT(*) as n
        FROM staging_edm_live
        WHERE received_at >= datetime('now', '-1 day')
        GROUP BY company, status
        ORDER BY company
    """, conn)
    matched = pd.read_sql("""
        SELECT
            SUM(CASE WHEN wb_id IS NOT NULL THEN 1 ELSE 0 END) as matched,
            COUNT(*) as total
        FROM staging_edm_live
        WHERE received_at >= datetime('now', '-1 day')
    """, conn)
    conn.close()

    if not summary.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Last 24 hours by company")
            st.dataframe(summary, hide_index=True, use_container_width=True)
        with c2:
            m = matched.iloc[0]
            pct = (m["matched"] / m["total"] * 100) if m["total"] else 0
            st.metric("Matched to a waterbody", f"{m['matched']:,} of {m['total']:,}", f"{pct:.0f}%")
    else:
        st.info("No live data in the last 24 hours.")
except Exception as e:
    st.error(f"Could not load live data summary: {e}")
    