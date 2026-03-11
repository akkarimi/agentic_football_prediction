import sqlite3
import json
from datetime import date
from config import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id         INTEGER,
                home_team        TEXT,
                away_team        TEXT,
                competition      TEXT,
                match_date       TEXT,
                utc_date         TEXT,
                home_win_prob    REAL,
                draw_prob        REAL,
                away_win_prob    REAL,
                home_goals       REAL,
                away_goals       REAL,
                confidence       TEXT,
                key_factors      TEXT,
                match_preview    TEXT,
                predicted_winner TEXT,
                created_at       TEXT DEFAULT (datetime('now')),
                actual_home_goals REAL,
                actual_away_goals REAL,
                actual_winner     TEXT,
                UNIQUE(match_id, match_date)
            )
        """)
        conn.commit()


def save_predictions(predictions: list[dict]):
    with sqlite3.connect(DB_PATH) as conn:
        for p in predictions:
            if "error" in p:
                continue
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (match_id, home_team, away_team, competition, match_date, utc_date,
                home_win_prob, draw_prob, away_win_prob, home_goals, away_goals,
                confidence, key_factors, match_preview, predicted_winner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, (
                p["match_id"], p["home_team"], p["away_team"],
                p["competition"], p["match_date"], p.get("utc_date", ""),
                p["home_win_prob"], p["draw_prob"], p["away_win_prob"],
                p.get("home_goals", 0.0), p.get("away_goals", 0.0),
                p["confidence"], json.dumps(p["key_factors"]),

                p["match_preview"], p["predicted_winner"]
            ))
        conn.commit()


def get_predictions(target_date: str = None) -> list[dict]:
    if not target_date:
        target_date = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE match_date = ?
            ORDER BY competition, home_team
        """, (target_date,)).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["key_factors"] = json.loads(d["key_factors"] or "[]")
        result.append(d)
    return result


def get_last_updated() -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT MAX(created_at) FROM predictions").fetchone()
        return row[0] if row and row[0] else None

def migrate_actual_columns():
    """Add actual result columns to existing tables if missing."""
    with sqlite3.connect(DB_PATH) as conn:
        for col in ["actual_home_goals REAL", "actual_away_goals REAL", "actual_winner TEXT"]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        conn.commit()


def get_unresolved_predictions() -> list[dict]:
    """Get past predictions that have no actual result yet."""
    from datetime import date
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT match_id, match_date FROM predictions
            WHERE actual_winner IS NULL AND match_date < ?
        """, (date.today().isoformat(),)).fetchall()
    return [dict(r) for r in rows]


def update_actual_result(match_id: int, home_goals: float, away_goals: float, winner: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            UPDATE predictions
            SET actual_home_goals = ?, actual_away_goals = ?, actual_winner = ?
            WHERE match_id = ?
        """, (home_goals, away_goals, winner, match_id))
        conn.commit()


def get_accuracy_stats(days: int = None) -> dict:
    """Return accuracy metrics. days=None means all-time."""
    from datetime import date, timedelta
    with sqlite3.connect(DB_PATH) as conn:
        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            rows = conn.execute("""
                SELECT predicted_winner, actual_winner,
                       home_goals, actual_home_goals,
                       away_goals, actual_away_goals
                FROM predictions
                WHERE actual_winner IS NOT NULL AND match_date >= ?
            """, (cutoff,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT predicted_winner, actual_winner,
                       home_goals, actual_home_goals,
                       away_goals, actual_away_goals
                FROM predictions
                WHERE actual_winner IS NOT NULL
            """).fetchall()

    if not rows:
        return {}

    total = len(rows)
    correct = sum(1 for r in rows if r[0] == r[1])
    home_mae = sum(abs(r[2] - r[3]) for r in rows if None not in (r[2], r[3])) / total
    away_mae = sum(abs(r[4] - r[5]) for r in rows if None not in (r[4], r[5])) / total

    return {
        "total":    total,
        "correct":  correct,
        "accuracy": round(correct / total * 100, 1),
        "home_goal_mae": round(home_mae, 2),
        "away_goal_mae": round(away_mae, 2),
    }
