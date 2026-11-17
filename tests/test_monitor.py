"""Drift-monitoring API tests: baselines, windows, alarms, idempotency."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import lease as lm
from app import store
from app.config import Settings
from app.db import connect
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    monkeypatch.setenv("TRAINING_DB_PATH", str(db))
    monkeypatch.setenv("TRAINING_LEASE_SECONDS", "0.6")
    monkeypatch.setenv("TRAINING_HEARTBEAT_INTERVAL", "0.1")

    import app.main as main_module

    monkeypatch.setattr(main_module, "settings", Settings.from_env())
    with TestClient(app) as test_client:
        yield test_client


def db_path() -> object:
    return Settings.from_env().db_path


def training_payload(records: int = 16, seed: int = 0, constant_col: int | None = None):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 2))
    features[:, 1] = features[:, 1] * 0.5 + 5.0  # column 1: mean ~5, std ~0.5
    if constant_col is not None:
        features[:, constant_col] = 3.5
    return {
        "features": features.tolist(),
        "labels": [i % 2 for i in range(records)],
        "epochs": 2,
    }


def submit_job(client, **kwargs) -> dict:
    response = client.post("/jobs", json=training_payload(**kwargs))
    assert response.status_code == 201
    return response.json()


def complete_job(job_id: str) -> None:
    path = db_path()
    lm.register_worker(path, "w1")
    lease = lm.acquire(path, job_id, "w1", 60.0)
    while True:
        result = store.advance_job(path, lease)
        if result["status"] == "completed":
            return


def completed_job(client, **kwargs) -> dict:
    job = submit_job(client, **kwargs)
    complete_job(job["job_id"])
    return client.get(f"/jobs/{job['job_id']}").json()


def strip_stats(job_id: str) -> None:
    """Simulate a job created before drift monitoring existed."""
    with connect(db_path()) as conn:
        conn.execute(
            "UPDATE jobs SET feature_means_json=NULL, feature_stds_json=NULL "
            "WHERE job_id=?",
            (job_id,),
        )


def set_baseline(client, job_id: str):
    return client.post("/monitor/baseline", json={"job_id": job_id})


def report(client, report_id: str, readings: list[list[float]]):
    return client.post(
        "/monitor/readings", json={"report_id": report_id, "readings": readings}
    )


def at_baseline(means: list[float], **overrides: float) -> list[float]:
    reading = list(means)
    for column, value in overrides.items():
        reading[int(column)] = value
    return reading


def test_job_stores_raw_column_stats(client):
    payload = training_payload(seed=3)
    job = client.post("/jobs", json=payload).json()
    detail = client.get(f"/jobs/{job['job_id']}").json()
    matrix = np.asarray(payload["features"])
    assert detail["feature_means"] == pytest.approx(matrix.mean(axis=0).tolist())
    assert detail["feature_stds"] == pytest.approx(matrix.std(axis=0).tolist())


def test_baseline_requires_existing_completed_job_with_stats(client):
    assert set_baseline(client, "does-not-exist").status_code == 404

    pending = submit_job(client, seed=1)
    response = set_baseline(client, pending["job_id"])
    assert response.status_code == 409
    assert "completed" in response.json()["detail"]

    legacy = completed_job(client, seed=2)
    strip_stats(legacy["job_id"])
    response = set_baseline(client, legacy["job_id"])
    assert response.status_code == 409
    assert "statistics" in response.json()["detail"]

    modern = completed_job(client, seed=3)
    response = set_baseline(client, modern["job_id"])
    assert response.status_code == 200
    body = response.json()
    assert body["baseline_job_id"] == modern["job_id"]
    assert body["feature_dim"] == 2
    assert body["window_size"] == 16
    assert body["alarm_sigmas"] == 2.0
    assert body["dropped_readings"] == 0


def test_readings_require_a_baseline(client):
    response = report(client, "r1", [[0.1, 5.0]])
    assert response.status_code == 409
    assert "baseline" in response.json()["detail"]


def test_window_alarms_on_mean_shift(client):
    job = completed_job(client, seed=4)
    means, stds = job["feature_means"], job["feature_stds"]
    set_baseline(client, job["job_id"])

    shifted = [at_baseline(means, **{"0": means[0] + 10 * stds[0]}) for _ in range(16)]
    response = report(client, "r1", shifted)
    assert response.status_code == 200
    body = response.json()
    assert body["duplicate"] is False
    assert body["accepted"] == 16
    assert body["window"] == {"window_size": 16, "seq": 1, "filled": 0}
    assert len(body["alarms"]) == 1
    alarm_window = body["alarms"][0]
    assert alarm_window["baseline_job_id"] == job["job_id"]
    assert alarm_window["window_seq"] == 1
    assert [c["column"] for c in alarm_window["columns"]] == [0]
    column = alarm_window["columns"][0]
    assert column["reason"] == "mean_shift"
    assert column["offset"] == pytest.approx(10 * stds[0])
    assert column["threshold"] == pytest.approx(2 * stds[0])
    assert column["baseline_mean"] == pytest.approx(means[0])

    calm = [at_baseline(means) for _ in range(16)]
    body = report(client, "r2", calm).json()
    assert body["alarms"] == []
    assert body["window"]["seq"] == 2


def test_constant_column_alarms_on_any_different_reading(client):
    job = completed_job(client, seed=5, constant_col=1)
    means = job["feature_means"]
    assert job["feature_stds"][1] == pytest.approx(0.0)
    response = set_baseline(client, job["job_id"])
    assert response.json()["constant_columns"] == [1]

    readings = [at_baseline(means) for _ in range(16)]
    readings[7] = at_baseline(means, **{"1": 3.7})
    body = report(client, "r1", readings).json()
    assert len(body["alarms"]) == 1
    columns = body["alarms"][0]["columns"]
    assert [c["column"] for c in columns] == [1]
    assert columns[0]["reason"] == "constant_column"
    assert columns[0]["value"] == pytest.approx(3.7)
    assert columns[0]["offset"] == pytest.approx(3.7 - means[1])
    assert body["alarms"][0]["baseline_job_id"] == job["job_id"]


def test_report_retry_is_idempotent(client):
    job = completed_job(client, seed=6)
    means = job["feature_means"]
    set_baseline(client, job["job_id"])

    first = report(client, "r1", [at_baseline(means) for _ in range(8)]).json()
    assert first["window"]["filled"] == 8

    retry = report(client, "r1", [at_baseline(means) for _ in range(8)]).json()
    assert retry["duplicate"] is True
    assert retry["window"]["filled"] == 8
    assert {k: v for k, v in retry.items() if k != "duplicate"} == {
        k: v for k, v in first.items() if k != "duplicate"
    }

    # A retry with a *different* body under the same key is still the first report.
    shifted = [at_baseline(means, **{"0": means[0] + 99.0}) for _ in range(8)]
    again = report(client, "r1", shifted).json()
    assert again["duplicate"] is True
    assert client.get("/monitor/status").json()["window"]["filled"] == 8


def test_windows_are_sixteen_readings_and_non_overlapping(client):
    job = completed_job(client, seed=7)
    means = job["feature_means"]
    set_baseline(client, job["job_id"])

    body = report(client, "r1", [at_baseline(means) for _ in range(10)]).json()
    assert body["window"] == {"window_size": 16, "seq": 0, "filled": 10}
    body = report(client, "r2", [at_baseline(means) for _ in range(6)]).json()
    assert body["window"] == {"window_size": 16, "seq": 1, "filled": 0}
    # One report may complete several windows and start the next.
    body = report(client, "r3", [at_baseline(means) for _ in range(40)]).json()
    assert body["window"] == {"window_size": 16, "seq": 3, "filled": 8}
    windows = client.get("/monitor/windows").json()
    assert [w["window_seq"] for w in windows] == [3, 2, 1]
    assert all(w["baseline_job_id"] == job["job_id"] for w in windows)


def test_switching_baseline_discards_partial_window(client):
    job_a = completed_job(client, seed=8)
    job_b = completed_job(client, seed=9)
    set_baseline(client, job_a["job_id"])

    means_a = job_a["feature_means"]
    assert report(client, "r1", [at_baseline(means_a) for _ in range(5)]).json()["window"][
        "filled"
    ] == 5

    response = set_baseline(client, job_b["job_id"])
    assert response.status_code == 200
    assert response.json()["previous_baseline_job_id"] == job_a["job_id"]
    assert response.json()["dropped_readings"] == 5

    status = client.get("/monitor/status").json()
    assert status["baseline_job_id"] == job_b["job_id"]
    assert status["window"] == {"seq": 0, "filled": 0}

    # New windows are evaluated against the new baseline only.
    means_b, stds_b = job_b["feature_means"], job_b["feature_stds"]
    shifted = [at_baseline(means_b, **{"0": means_b[0] + 9 * stds_b[0]}) for _ in range(16)]
    body = report(client, "r2", shifted).json()
    assert body["alarms"][0]["baseline_job_id"] == job_b["job_id"]

    # Re-setting the same baseline keeps the partial window.
    assert report(client, "r3", [at_baseline(means_b) for _ in range(4)]).json()["window"][
        "filled"
    ] == 4
    assert set_baseline(client, job_b["job_id"]).json()["dropped_readings"] == 0
    assert client.get("/monitor/status").json()["window"]["filled"] == 4


def test_formed_conclusions_keep_their_original_baseline(client):
    job_a = completed_job(client, seed=10)
    job_b = completed_job(client, seed=11)
    set_baseline(client, job_a["job_id"])

    means_a, stds_a = job_a["feature_means"], job_a["feature_stds"]
    shifted = [at_baseline(means_a, **{"0": means_a[0] + 8 * stds_a[0]}) for _ in range(16)]
    assert report(client, "r1", shifted).json()["alarms"]

    set_baseline(client, job_b["job_id"])
    windows = client.get("/monitor/windows").json()
    assert len(windows) == 1
    assert windows[0]["baseline_job_id"] == job_a["job_id"]
    assert windows[0]["alarms"][0]["column"] == 0


def test_partial_window_survives_api_restart(client):
    job = completed_job(client, seed=12)
    means = job["feature_means"]
    set_baseline(client, job["job_id"])
    assert report(client, "r1", [at_baseline(means) for _ in range(5)]).json()["window"][
        "filled"
    ] == 5

    # Simulate an API restart: a fresh app instance over the same database.
    with TestClient(app) as restarted:
        status = restarted.get("/monitor/status").json()
        assert status["baseline_job_id"] == job["job_id"]
        assert status["window"] == {"seq": 0, "filled": 5}
        body = report(restarted, "r2", [at_baseline(means) for _ in range(11)]).json()
        assert body["window"] == {"window_size": 16, "seq": 1, "filled": 0}


def test_reading_validation(client):
    job = completed_job(client, seed=13)
    means = job["feature_means"]
    set_baseline(client, job["job_id"])

    assert report(client, "r1", []).status_code == 422
    assert report(client, "r2", [[]]).status_code == 422
    assert report(client, "r3", [[1.0, 2.0], [1.0]]).status_code == 422
    assert report(client, "r4", [[1.0, 2.0, 3.0]]).status_code == 422
    # NaN/Infinity are not strict JSON; send them raw, as a lenient client would.
    for token in ("NaN", "Infinity", "-Infinity"):
        raw = f'{{"report_id": "r5-{token}", "readings": [[{token}, 2.0]]}}'
        response = client.post(
            "/monitor/readings",
            content=raw.encode(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, token
    assert report(client, "r7", [["vibration", 2.0]]).status_code == 422
    # Failed validation must not consume the idempotency key or the window.
    assert report(client, "r4", [at_baseline(means)]).status_code == 200
    assert client.get("/monitor/status").json()["window"]["filled"] == 1


def test_monitor_endpoints_in_openapi(client):
    schema = client.get("/openapi.json").json()
    assert "/monitor/baseline" in schema["paths"]
    assert "/monitor/readings" in schema["paths"]
    assert "/monitor/status" in schema["paths"]
    assert "/monitor/windows" in schema["paths"]
