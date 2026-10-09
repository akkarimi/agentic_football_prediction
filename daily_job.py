"""Daily update: fetch final scores, then predict the next two days' fixtures.

Runs automatically inside the app (see ensure_scheduler), or by hand / from cron:
    python daily_job.py
"""
import threading
import time
from datetime import datetime, timedelta, timezone

from config import DAILY_RUN_TIME
from db import init_db, migrate_actual_columns, get_predictions, save_predictions, get_meta, set_meta
from pipeline.graph import fetch_tracked_matches, predict_match, refresh_actual_results

THREAD_NAME = "daily-predictions"
RETRY_AFTER_CRASH_SECONDS = 30 * 60


def latest_due(now: datetime = None) -> datetime:
    """The most recent scheduled run time (local) that has already passed."""
    now = now or datetime.now()
    hour, minute = (int(x) for x in DAILY_RUN_TIME.split(":"))
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return due if due <= now else due - timedelta(days=1)


def run_daily_job(due: datetime, status=lambda message: None) -> dict:
    """Update results, then predict the two days after `due`. Already-stored matches are skipped."""
    errors = []
    updated = predicted = 0

    status("Fetching final scores...")
    try:
        updated = refresh_actual_results()
    except Exception as e:
        errors.append(f"results: {e}")

    for offset in (1, 2):
        target = (due.date() + timedelta(days=offset)).isoformat()
        status(f"Fetching fixtures for {target}...")
        try:
            fixtures = fetch_tracked_matches(target)
        except Exception as e:
            errors.append(f"fixtures {target}: {e}")
            continue

        stored = {p["match_id"] for p in get_predictions(target)}
        for match in fixtures:
            # Never predict a match that has kicked off: the news would leak the result
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if match["id"] in stored or match["utcDate"] <= now_utc:
                continue
            name = f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}"
            status(f"Predicting {name}...")
            p = predict_match(match)
            if "error" in p:
                errors.append(f"{name}: {p['error']}")
            else:
                save_predictions([p])
                predicted += 1

    summary = {"updated": updated, "predicted": predicted, "errors": errors}
    set_meta("last_daily_run", due.isoformat())
    set_meta("last_daily_run_finished", datetime.now().isoformat(timespec="seconds"))
    set_meta("last_daily_run_errors", "\n".join(errors))
    return summary


class _Scheduler(threading.Thread):
    def __init__(self):
        super().__init__(name=THREAD_NAME, daemon=True)
        self.status = ""

    def _set_status(self, message: str):
        self.status = message

    def run(self):
        while True:
            due = latest_due()
            last = get_meta("last_daily_run")
            # Also catches up on a run that was missed while the app was not running
            if not last or last < due.isoformat():
                try:
                    run_daily_job(due, self._set_status)
                except Exception as e:
                    print(f"Daily job crashed, retrying later: {e}")
                    self.status = ""
                    time.sleep(RETRY_AFTER_CRASH_SECONDS)
                    continue
                self.status = ""
            time.sleep(60)


def _scheduler_thread():
    return next((t for t in threading.enumerate() if t.name == THREAD_NAME), None)


def ensure_scheduler():
    """Start the background scheduler once per server process."""
    if _scheduler_thread() is None:
        _Scheduler().start()


def scheduler_status() -> str:
    """What the daily job is doing right now, or '' when idle."""
    thread = _scheduler_thread()
    return getattr(thread, "status", "") if thread else ""


if __name__ == "__main__":
    init_db()
    migrate_actual_columns()
    result = run_daily_job(latest_due(), status=print)
    print(f"Results updated: {result['updated']} · matches predicted: {result['predicted']}")
    for error in result["errors"]:
        print("ERROR:", error)
