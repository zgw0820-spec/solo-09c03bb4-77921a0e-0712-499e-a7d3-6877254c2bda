"""Input drift monitoring against a completed training baseline.

A *baseline* is a training job that (a) finished and (b) stored the per-column
mean/std of its raw submitted sample.  Jobs created before this feature
existed have no statistics; the standardised design matrix cannot be
un-scaled back to engineering units, so such jobs are refused outright
instead of guessing.

Readings are reported one raw measurement at a time (a request may carry a
batch of them).  Every 16 readings form one non-overlapping window.  A window
binds to the baseline that is current when its first reading arrives; when it
fills, its per-column means are compared with that baseline:

* a column alarms when ``|window_mean - baseline_mean| > 2 * baseline_std``;
* a column that is constant in the baseline (std ~ 0) alarms as soon as any
  reading in the window differs from the baseline constant.

Switching the baseline discards the partial window, so readings collected
under the old baseline are never evaluated against the new one; windows that
already filled keep their conclusions, permanently tagged with the baseline
that evaluated them.  Reports carry a client ``report_id``: a retry with the
same id returns the stored first response and is never counted twice.

All state lives in the shared SQLite database (singleton ``monitor_state``
row), so a partially filled window survives an API restart.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

from .db import reader, transaction
from .store import JobNotFound, StateConflict

WINDOW_SIZE = 16
ALARM_SIGMAS = 2.0
# Same constant-column threshold as features.normalise.
CONSTANT_STD = 1e-12
# Tolerance for "a reading differs from the baseline constant".
VALUE_TOLERANCE = 1e-12


class BaselineNotEligible(RuntimeError):
    """The job cannot be a drift baseline (unfinished or no raw statistics)."""


class NoBaseline(RuntimeError):
    """Readings were reported before any baseline was set."""


class BadReading(ValueError):
    """The reported readings are malformed."""


def _state(conn: sqlite3.Connection) -> sqlite3.Row:
    """The singleton monitor row, creating it inside a write transaction."""
    conn.execute(
        "INSERT OR IGNORE INTO monitor_state(id, window_seq, window_count) VALUES(1, 0, 0)"
    )
    return conn.execute("SELECT * FROM monitor_state WHERE id=1").fetchone()


def _baseline_stats(conn: sqlite3.Connection, job_id: str) -> tuple[list[float], list[float]]:
    row = conn.execute(
        "SELECT feature_means_json, feature_stds_json FROM jobs WHERE job_id=?",
        (job_id,),
    ).fetchone()
    if row is None or row["feature_means_json"] is None or row["feature_stds_json"] is None:
        raise NoBaseline(f"baseline job {job_id} is missing its raw-sample statistics")
    means = [float(v) for v in json.loads(row["feature_means_json"])]
    stds = [float(v) for v in json.loads(row["feature_stds_json"])]
    return means, stds


def set_baseline(db_path, job_id: str) -> dict[str, Any]:
    """Make ``job_id`` the drift baseline.

    Only a completed job with stored raw-sample statistics is eligible.  When
    the baseline actually changes, the partially filled window is discarded:
    readings collected under the old baseline must never be evaluated against
    the new one.  Re-setting the same baseline is a no-op for the window.
    """
    now = time.time()
    with transaction(db_path) as conn:
        row = conn.execute(
            "SELECT status, feature_means_json, feature_stds_json FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        if row["status"] != "completed":
            raise BaselineNotEligible(
                f"job {job_id} is {row['status']}; only a completed job can be the drift baseline"
            )
        if row["feature_means_json"] is None or row["feature_stds_json"] is None:
            raise BaselineNotEligible(
                f"job {job_id} has no raw-sample column statistics (it predates drift "
                "monitoring); refusing to reconstruct them from standardised samples"
            )
        means = [float(v) for v in json.loads(row["feature_means_json"])]
        stds = [float(v) for v in json.loads(row["feature_stds_json"])]

        state = _state(conn)
        previous = state["baseline_job_id"]
        dropped = 0
        set_at = state["baseline_set_at"]
        if previous != job_id:
            dropped = int(state["window_count"])
            set_at = now
            conn.execute(
                """
                UPDATE monitor_state
                   SET baseline_job_id=?, baseline_set_at=?,
                       window_baseline_job_id=NULL, window_count=0,
                       window_sums_json=NULL, window_violations_json=NULL
                 WHERE id=1
                """,
                (job_id, now),
            )
    return {
        "baseline_job_id": job_id,
        "previous_baseline_job_id": previous,
        "feature_dim": len(means),
        "baseline_means": means,
        "baseline_stds": stds,
        "constant_columns": [i for i, s in enumerate(stds) if s < CONSTANT_STD],
        "dropped_readings": dropped,
        "window_size": WINDOW_SIZE,
        "alarm_sigmas": ALARM_SIGMAS,
        "set_at": set_at,
    }


def _evaluate_window(
    sums: list[float],
    means: list[float],
    stds: list[float],
    violations: dict[str, float],
) -> list[dict[str, Any]]:
    """Alarm columns of one full window, in column order."""
    alarms: list[dict[str, Any]] = []
    for column in range(len(means)):
        window_mean = sums[column] / WINDOW_SIZE
        if stds[column] < CONSTANT_STD:
            offending = violations.get(str(column))
            if offending is not None:
                alarms.append(
                    {
                        "column": column,
                        "reason": "constant_column",
                        "baseline_mean": means[column],
                        "baseline_std": stds[column],
                        "value": offending,
                        "offset": offending - means[column],
                    }
                )
            continue
        offset = window_mean - means[column]
        threshold = ALARM_SIGMAS * stds[column]
        if abs(offset) > threshold:
            alarms.append(
                {
                    "column": column,
                    "reason": "mean_shift",
                    "baseline_mean": means[column],
                    "baseline_std": stds[column],
                    "window_mean": window_mean,
                    "offset": offset,
                    "threshold": threshold,
                }
            )
    return alarms


def submit_readings(
    db_path, report_id: str, readings: list[list[float]]
) -> dict[str, Any]:
    """Append raw readings to the current window and evaluate full windows.

    Idempotent on ``report_id``: a retry returns the first response with
    ``duplicate=True`` and does not count the readings again.
    """
    if not readings:
        raise BadReading("submit at least one reading")
    width = len(readings[0])
    if width == 0:
        raise BadReading("each reading needs at least one value")
    for reading in readings:
        if len(reading) != width:
            raise BadReading("every reading must have the same number of values")
        for value in reading:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise BadReading("readings must be finite numbers")

    now = time.time()
    with transaction(db_path) as conn:
        existing = conn.execute(
            "SELECT response_json FROM monitor_reports WHERE report_id=?", (report_id,)
        ).fetchone()
        if existing is not None:
            response = json.loads(existing["response_json"])
            response["duplicate"] = True
            return response

        state = _state(conn)
        baseline_id = state["baseline_job_id"]
        if baseline_id is None:
            raise NoBaseline("no drift baseline set; POST /monitor/baseline first")
        means, stds = _baseline_stats(conn, baseline_id)
        if width != len(means):
            raise BadReading(
                f"baseline expects {len(means)} values per reading, got {width}"
            )

        count = int(state["window_count"])
        seq = int(state["window_seq"])
        sums = json.loads(state["window_sums_json"]) if state["window_sums_json"] else None
        violations: dict[str, float] = (
            json.loads(state["window_violations_json"]) if state["window_violations_json"] else {}
        )
        window_baseline = state["window_baseline_job_id"]
        if count > 0 and window_baseline != baseline_id:
            # Unreachable via the API (switching the baseline clears the
            # window in the same transaction); guard against manual edits.
            raise StateConflict("partial window is bound to a different baseline")

        completed: list[dict[str, Any]] = []
        for reading in readings:
            if count == 0:
                # A window binds to the baseline that is current when its
                # first reading arrives.
                sums = [0.0] * len(means)
                violations = {}
                window_baseline = baseline_id
            for column, value in enumerate(reading):
                sums[column] += value
                if (
                    stds[column] < CONSTANT_STD
                    and abs(value - means[column]) > VALUE_TOLERANCE
                ):
                    violations.setdefault(str(column), float(value))
            count += 1
            if count == WINDOW_SIZE:
                seq += 1
                alarms = _evaluate_window(sums, means, stds, violations)
                window_means = [s / WINDOW_SIZE for s in sums]
                conn.execute(
                    """
                    INSERT INTO monitor_windows(
                        window_seq, baseline_job_id, window_means_json, alarms_json, created_at
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (seq, window_baseline, json.dumps(window_means), json.dumps(alarms), now),
                )
                if alarms:
                    completed.append(
                        {
                            "window_seq": seq,
                            "baseline_job_id": window_baseline,
                            "columns": alarms,
                        }
                    )
                count = 0
                sums = None
                violations = {}
                window_baseline = None

        conn.execute(
            """
            UPDATE monitor_state
               SET window_seq=?, window_count=?, window_baseline_job_id=?,
                   window_sums_json=?, window_violations_json=?
             WHERE id=1
            """,
            (
                seq,
                count,
                window_baseline,
                json.dumps(sums) if sums is not None else None,
                json.dumps(violations) if violations else None,
            ),
        )
        response = {
            "report_id": report_id,
            "duplicate": False,
            "accepted": len(readings),
            "baseline_job_id": baseline_id,
            "window": {"window_size": WINDOW_SIZE, "seq": seq, "filled": count},
            "alarms": completed,
        }
        conn.execute(
            "INSERT INTO monitor_reports(report_id, response_json, created_at) VALUES(?, ?, ?)",
            (report_id, json.dumps(response), now),
        )
        return response


def get_status(db_path) -> dict[str, Any]:
    """Current baseline and partial-window progress."""
    with reader(db_path) as conn:
        state = conn.execute("SELECT * FROM monitor_state WHERE id=1").fetchone()
    if state is None:
        return {
            "baseline_job_id": None,
            "baseline_set_at": None,
            "window_size": WINDOW_SIZE,
            "alarm_sigmas": ALARM_SIGMAS,
            "window": {"seq": 0, "filled": 0},
        }
    return {
        "baseline_job_id": state["baseline_job_id"],
        "baseline_set_at": state["baseline_set_at"],
        "window_size": WINDOW_SIZE,
        "alarm_sigmas": ALARM_SIGMAS,
        "window": {
            "seq": int(state["window_seq"]),
            "filled": int(state["window_count"]),
        },
    }


def list_windows(db_path, limit: int = 100) -> list[dict[str, Any]]:
    """Evaluated windows, newest first; conclusions keep their baseline."""
    with reader(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM monitor_windows ORDER BY window_seq DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {
            "window_seq": int(row["window_seq"]),
            "baseline_job_id": row["baseline_job_id"],
            "window_means": json.loads(row["window_means_json"]),
            "alarms": json.loads(row["alarms_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
