"""
scheduler.py
APScheduler orchestration for live data ingestion.

Runs three jobs on independent schedules:
    - POOPy live sewage data     every 15 minutes
    - FreshWater Watch check     weekly
    - Database snapshot backup   weekly

Does NOT retrain the model. Live data feeds inference only - the
trained model stays fixed until a manual retrain is triggered (see
PIPELINE_RUN_ORDER.md, "Retraining" section).

Run:
    python src/scheduler.py            runs in foreground, Ctrl+C to stop
    nohup python src/scheduler.py &    runs in background, survives logout
"""

import logging
import sys
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from config import SCHEDULE_FWW_SECS, SCHEDULE_EDM_SECS, SCHEDULE_SNAP_SECS
from db_loader import log_run, db_snapshot

# ---- Logging setup ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")


# ---- Job wrappers ---------------------------------------------------------------
# Each job is wrapped so one failure never crashes the whole scheduler -
# APScheduler will just skip that run and try again next interval.

def job_poopy_pull():
    """Pull current sewage discharge status from all water companies via POOPy."""
    log.info("Starting POOPy live pull...")
    try:
        from pull_poopy_now import main as poopy_main
        poopy_main()
        log.info("POOPy pull completed successfully.")
    except Exception as e:
        log.error(f"POOPy pull FAILED: {e}")
        log_run("edm_live_scheduled", 0, 0, "failed", str(e))


def job_fww_check():
    """Check for new FreshWater Watch records since the last pull."""
    log.info("Starting FWW weekly check...")
    try:
        from pull_fww_weekly import main as fww_main
        fww_main()
        log.info("FWW check completed successfully.")
    except Exception as e:
        log.error(f"FWW check FAILED: {e}")
        log_run("fww_weekly_scheduled", 0, 0, "failed", str(e))


def job_db_snapshot():
    """Weekly full database backup, per the data management plan."""
    log.info("Starting database snapshot...")
    try:
        path = db_snapshot()
        log.info(f"Snapshot saved: {path}")
    except Exception as e:
        log.error(f"Snapshot FAILED: {e}")
        log_run("db_snapshot_scheduled", 0, 0, "failed", str(e))


def job_heartbeat():
    """Simple proof-of-life log line, useful when checking the scheduler is alive."""
    log.info(f"Scheduler heartbeat - alive at {datetime.now().isoformat()}")


# ---- Build and run the scheduler -------------------------------------------------

def build_scheduler():
    """
    Create the scheduler with all jobs registered.
    ThreadPoolExecutor lets jobs run without blocking each other if two
    intervals happen to fire close together.
    """
    scheduler = BlockingScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=3)}
    )

    scheduler.add_job(
        job_poopy_pull, "interval", seconds=SCHEDULE_EDM_SECS,
        id="poopy_pull", next_run_time=datetime.now(),  # run once immediately on start
        misfire_grace_time=120,
    )

    scheduler.add_job(
        job_fww_check, "interval", seconds=SCHEDULE_FWW_SECS,
        id="fww_check", misfire_grace_time=3600,
    )

    scheduler.add_job(
        job_db_snapshot, "interval", seconds=SCHEDULE_SNAP_SECS,
        id="db_snapshot", misfire_grace_time=3600,
    )

    # heartbeat every 10 minutes - reassurance that the process is alive,
    # separate from the actual data jobs
    scheduler.add_job(job_heartbeat, "interval", minutes=10, id="heartbeat")

    return scheduler


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)

    log.info("=" * 60)
    log.info("Starting scheduler")
    log.info(f"  POOPy live pull    : every {SCHEDULE_EDM_SECS // 60} minutes")
    log.info(f"  FWW check          : every {SCHEDULE_FWW_SECS // 86400} days")
    log.info(f"  Database snapshot  : every {SCHEDULE_SNAP_SECS // 86400} days")
    log.info("=" * 60)

    sched = build_scheduler()

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped by user.")