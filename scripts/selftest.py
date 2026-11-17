#!/usr/bin/env python3
"""Self-checks that do not need Docker.

1. ``config``: the Compose file is valid and wires two workers and the API to
   one shared volume.
2. ``equivalence``: with real worker processes, kill one mid-training and prove
   the finished weights are bit-identical to an uninterrupted run.
3. ``late``: a frozen (still-running) old worker cannot write after takeover.
4. ``drift``: the drift-monitoring API - baseline eligibility, window alarms,
   idempotent retries, partial-window survival across an API restart, and
   baseline-switch semantics.

Each check prints PASS/FAIL and exits non-zero if anything fails.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASE = 2.0
HEARTBEAT = 0.3
DELAY = 0.2


def check_compose() -> bool:
    try:
        import yaml
    except ImportError:
        print("  SKIP: PyYAML not installed")
        return True
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = config.get("services", {})
    workers = [name for name in services if name.startswith("worker-")]
    assert services["api"]["ports"] == ["8000:8000"], services["api"]["ports"]
    assert len(workers) == 2, workers
    for name in workers:
        volumes = services[name]["volumes"]
        assert volumes == ["training-data:/data"], (name, volumes)
        assert services[name]["command"][-1] == name, services[name]["command"]
    assert services["api"]["volumes"] == ["training-data:/data"]
    assert "training-data" in config["volumes"]
    print(f"  PASS: compose has api + {workers} sharing volume training-data")
    return True


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(LEASE),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(DELAY),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


def get(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read())


def post(port: int, path: str, body: dict):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def post_raw(port: int, path: str, body: dict) -> tuple[int, dict]:
    """POST that also returns error responses instead of raising."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_for(predicate, timeout: float, message: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise SystemExit(f"  FAIL: timed out waiting for {message}")


def start_api(db: Path) -> tuple[subprocess.Popen, int]:
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=ROOT,
        env=env_for(db),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for(lambda: _probe(port), 30, "API")
    return proc, port


def _probe(port: int) -> bool:
    try:
        get(port, "/health")
        return True
    except OSError:
        return False


def start_worker(db: Path, worker_id: str, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, **env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop(proc: subprocess.Popen, hard: bool = False) -> None:
    if proc.poll() is not None:
        return
    proc.kill() if hard else proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dataset(seed: int = 21, records: int = 40) -> dict:
    import random

    rng = random.Random(seed)
    features = [
        [round(rng.gauss(0, 1), 4) for _ in range(4)] for _ in range(records)
    ]
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": 3}


def completed_state(port: int, job_id: str) -> dict:
    return wait_for(
        lambda: (state := get(port, f"/jobs/{job_id}"))["status"] == "completed" and state,
        120,
        "job completion",
    )


def run_reference(tmp: Path) -> dict:
    db = tmp / "reference.db"
    api, port = start_api(db)
    worker = start_worker(db, "solo")
    try:
        job = post(port, "/jobs", dataset())
        return completed_state(port, job["job_id"])
    finally:
        stop(worker)
        stop(api)


def check_equivalence(tmp: Path) -> bool:
    reference = run_reference(tmp)
    db = tmp / "crash.db"
    api, port = start_api(db)
    job = post(port, "/jobs", dataset())
    worker_a = start_worker(db, "worker-a")
    try:
        wait_for(
            lambda: get(port, f"/jobs/{job['job_id']}")["completed_steps"] >= 3,
            60,
            "first batches",
        )
    finally:
        stop(worker_a, hard=True)
    at_crash = get(port, f"/jobs/{job['job_id']}")
    worker_b = start_worker(db, "worker-b")
    try:
        final = completed_state(port, job["job_id"])
        checkpoints = get(port, f"/jobs/{job['job_id']}/checkpoints")
        events = get(port, f"/jobs/{job['job_id']}/events")
    finally:
        stop(worker_b)
        stop(api)

    assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
    assert {c["worker_id"] for c in checkpoints} == {"worker-a", "worker-b"}
    resumed = [e for e in events if e["event"] == "training_started"]
    assert resumed[-1]["detail"]["resumed_from_step"] == at_crash["completed_steps"]
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"]
    assert final["final_weights"] == reference["final_weights"], (
        final["final_weights"],
        reference["final_weights"],
    )
    assert final["loss"] == reference["loss"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    print(
        f"  PASS: killed at step {at_crash['completed_steps']},"
        f" resumed to {final['completed_steps']}, weights match the uninterrupted run"
    )
    return True


def check_late_worker(tmp: Path) -> bool:
    db = tmp / "late.db"
    pause = tmp / "pause"
    api, port = start_api(db)
    job = post(port, "/jobs", dataset(seed=7, records=24))
    worker_a = start_worker(db, "worker-a", TRAINING_DEMO_PAUSE_FILE=str(pause))
    try:
        wait_for(
            lambda: get(port, f"/jobs/{job['job_id']}")["completed_steps"] >= 2,
            60,
            "A progress",
        )
        pause.write_text("pause")
        wait_for(
            lambda: get(port, "/workers")["leases"][0]["expired"],
            30,
            "A lease expiry",
        )
        frozen = get(port, f"/jobs/{job['job_id']}")
        worker_b = start_worker(db, "worker-b")
        try:
            final = completed_state(port, job["job_id"])
            before = get(port, f"/jobs/{job['job_id']}/checkpoints")
            pause.unlink()
            time.sleep(DELAY * 6)
            after = get(port, f"/jobs/{job['job_id']}")
            after_checkpoints = get(port, f"/jobs/{job['job_id']}/checkpoints")
        finally:
            stop(worker_a)
            stop(worker_b)
            stop(api)
    finally:
        stop(worker_a)
        pause.unlink(missing_ok=True)

    assert after["completed_steps"] == final["completed_steps"]
    assert after["weights"] == final["weights"]
    assert after["finished_at"] == final["finished_at"]
    assert after_checkpoints == before
    assert max(
        c["step"] for c in before if c["worker_id"] == "worker-a"
    ) <= frozen["completed_steps"]
    print(
        f"  PASS: old worker frozen at step {frozen['completed_steps']},"
        f" new worker finished at {final['completed_steps']}, no rollback or false completion"
    )
    return True


def check_drift_monitor(tmp: Path) -> bool:
    db = tmp / "drift.db"
    api, port = start_api(db)
    worker = start_worker(db, "drift-worker")
    try:
        data_a = dataset(seed=31)
        job_a = post(port, "/jobs", data_a)
        completed_state(port, job_a["job_id"])

        # Baseline eligibility: completed + stored raw-sample statistics.
        baseline = post(port, "/monitor/baseline", {"job_id": job_a["job_id"]})
        assert baseline["baseline_job_id"] == job_a["job_id"], baseline
        pending = post(port, "/jobs", dataset(seed=33))
        code, body = post_raw(port, "/monitor/baseline", {"job_id": pending["job_id"]})
        assert code == 409 and "completed" in body["detail"], (code, body)
        legacy = post(port, "/jobs", dataset(seed=34))
        completed_state(port, legacy["job_id"])
        # Simulate a job created before drift monitoring existed.
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE jobs SET feature_means_json=NULL, feature_stds_json=NULL "
                "WHERE job_id=?",
                (legacy["job_id"],),
            )
        code, body = post_raw(port, "/monitor/baseline", {"job_id": legacy["job_id"]})
        assert code == 409 and "statistics" in body["detail"], (code, body)

        # A window shifted by +5 on every column alarms on all four columns,
        # naming the columns, their offsets and the baseline job.
        shifted = [[value + 5.0 for value in row] for row in data_a["features"][:16]]
        result = post(port, "/monitor/readings", {"report_id": "shift-1", "readings": shifted})
        assert result["window"] == {"window_size": 16, "seq": 1, "filled": 0}, result
        assert len(result["alarms"]) == 1, result
        alarm = result["alarms"][0]
        assert alarm["baseline_job_id"] == job_a["job_id"], alarm
        assert {c["column"] for c in alarm["columns"]} == {0, 1, 2, 3}, alarm
        assert all(abs(c["offset"]) > c["threshold"] for c in alarm["columns"]), alarm

        # Retrying the same report does not count twice.
        retry = post(port, "/monitor/readings", {"report_id": "shift-1", "readings": shifted})
        assert retry["duplicate"] is True, retry
        assert get(port, "/monitor/status")["window"] == {"seq": 1, "filled": 0}

        # Five more readings stay pending; an API restart must keep them.
        calm = data_a["features"][16:21]
        result = post(port, "/monitor/readings", {"report_id": "calm-1", "readings": calm})
        assert result["window"]["filled"] == 5, result
        stop(api)
        api, port = start_api(db)
        status = get(port, "/monitor/status")
        assert status["baseline_job_id"] == job_a["job_id"], status
        assert status["window"] == {"seq": 1, "filled": 5}, status

        # Switching the baseline discards the partial window...
        data_b = dataset(seed=32)
        job_b = post(port, "/jobs", data_b)
        completed_state(port, job_b["job_id"])
        switched = post(port, "/monitor/baseline", {"job_id": job_b["job_id"]})
        assert switched["dropped_readings"] == 5, switched
        assert get(port, "/monitor/status")["window"] == {"seq": 1, "filled": 0}

        # ...while the already-formed alarm window still points at baseline A.
        windows = get(port, "/monitor/windows")
        assert windows[0]["window_seq"] == 1, windows
        assert windows[0]["baseline_job_id"] == job_a["job_id"], windows
        assert windows[0]["alarms"], windows

        # Readings near the new baseline raise no alarm.
        result = post(
            port, "/monitor/readings", {"report_id": "calm-2", "readings": data_b["features"][:16]}
        )
        assert result["alarms"] == [], result
        assert result["window"] == {"window_size": 16, "seq": 2, "filled": 0}, result
    finally:
        stop(worker)
        stop(api)
    print(
        "  PASS: baseline eligibility, 16-reading window alarm with columns/offsets, "
        "idempotent retry, restart-kept partial window, baseline switch semantics"
    )
    return True


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for name, check in (
            ("compose", check_compose),
            ("killed-worker equivalence", lambda: check_equivalence(tmp)),
            ("late-worker fencing", lambda: check_late_worker(tmp)),
            ("drift monitor", lambda: check_drift_monitor(tmp)),
        ):
            print(f"[{name}]")
            try:
                check()
            except AssertionError as exc:
                print(f"  FAIL: {exc}")
                failures.append(name)
        if failures:
            print(f"\nFAILED: {failures}")
            return 1
    print("\nAll self-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
