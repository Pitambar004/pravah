"""Add research replay routes to the existing water-segmentation API."""

from __future__ import annotations

import hmac
import json
from pathlib import Path

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field

from pravah.replay.experiment import audit, freeze, register_evaluation, verify_snapshot


class FreezeRequest(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")
    cutoff: str


def attach_replay_routes(app, root: Path, token: str | None = None):
    def manifest():
        try:
            return json.loads((root / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Prepared Nepal evidence is unavailable") from exc

    def require_token(x_pravah_token: str | None = Header(default=None)):
        if token and not (
            x_pravah_token and hmac.compare_digest(token, x_pravah_token)
        ):
            raise HTTPException(401, "Missing or invalid X-Pravah-Token")

    @app.get("/api/v1/replay/readiness")
    def readiness(cutoff: str = "2026-08-26T08:30:00+05:45"):
        doc = manifest()
        try:
            report = audit(doc, cutoff, root)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            **report,
            "event_id": doc["event_id"],
            "forecast_model_loaded": False,
            "forecast": None,
            "blockers": doc["blockers"],
        }

    @app.post("/api/v1/replay/snapshots", dependencies=[Depends(require_token)])
    def create_snapshot(request: FreezeRequest):
        try:
            return freeze(root, manifest(), request.cutoff, request.run_id)
        except FileExistsError as exc:
            raise HTTPException(
                409, "Experiment already exists; frozen inputs cannot be overwritten"
            ) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(503, "Replay storage is unavailable") from exc

    @app.get("/api/v1/replay/snapshots/{run_id}")
    def get_snapshot(run_id: str):
        try:
            return verify_snapshot(root, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Experiment not found") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post(
        "/api/v1/replay/snapshots/{run_id}/reveal",
        dependencies=[Depends(require_token)],
    )
    def reveal(run_id: str):
        try:
            return register_evaluation(root, manifest(), run_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                404, "Freeze the experiment before revealing references"
            ) from exc
        except FileExistsError as exc:
            raise HTTPException(409, "Evaluation reveal already recorded") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return app
