from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from .config import AstarConfig
from .types import (
    InitialState,
    RoundBudget,
    RoundDetail,
    Settlement,
    SimulationObservation,
    Viewport,
    terrain_grid_to_class_grid,
)
from .utils import load_json, save_json, stable_cache_key

LOGGER = logging.getLogger(__name__)


class AstarIslandAPI:
    def __init__(self, config: AstarConfig):
        self.config = config
        self.config.ensure_dirs()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.user_agent
        self.session.headers["Accept"] = "application/json"
        if config.token:
            if config.auth_mode.lower() == "cookie":
                self.session.cookies.set("access_token", config.token)
            else:
                self.session.headers["Authorization"] = f"Bearer {config.token}"

    def _cache_path(self, method: str, url: str, payload: dict[str, Any] | None) -> Path:
        key = stable_cache_key(method, url, payload)
        return self.config.cache.directory / f"{key}.json"

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        use_cache: bool = False,
        auth_required: bool = False,
    ) -> Any:
        url = f"{self.config.base_url}{path}"
        cache_path = self._cache_path(method, url, payload)
        if use_cache and self.config.cache.enabled and cache_path.exists():
            return load_json(cache_path)
        if auth_required and not self.config.token:
            raise RuntimeError("Missing token. Set ASTAR_ISLAND_TOKEN or pass one in AstarConfig.")

        for attempt in range(1, self.config.max_retries + 1):
            response = self.session.request(
                method,
                url,
                json=payload,
                timeout=self.config.request_timeout_s,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                delay = retry_after or self.config.retry_backoff_s * attempt
                LOGGER.warning(
                    "Request %s %s failed with %s; retrying in %.2fs (%s/%s)",
                    method,
                    path,
                    response.status_code,
                    delay,
                    attempt,
                    self.config.max_retries,
                )
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                detail = response.text[:1000]
                raise RuntimeError(f"{method} {path} failed with {response.status_code}: {detail}")
            data = response.json()
            if use_cache and self.config.cache.enabled:
                save_json(cache_path, data)
            return data
        raise RuntimeError(f"{method} {path} failed after {self.config.max_retries} retries")

    def get_rounds(self, use_cache: bool = True) -> list[dict[str, Any]]:
        return self._request("GET", "/rounds", use_cache=use_cache)

    def get_active_round(self) -> dict[str, Any] | None:
        rounds = self.get_rounds(use_cache=False)
        return next((item for item in rounds if item.get("status") == "active"), None)

    def get_round_details(self, round_id: str, use_cache: bool = True) -> RoundDetail:
        payload = self._request("GET", f"/rounds/{round_id}", use_cache=use_cache)
        initial_states = [
            InitialState(
                grid=np.asarray(state["grid"], dtype=np.int16),
                settlements=[Settlement.from_api(item) for item in state.get("settlements", [])],
            )
            for state in payload["initial_states"]
        ]
        return RoundDetail(
            round_id=payload["id"],
            round_number=int(payload["round_number"]),
            status=str(payload["status"]),
            map_width=int(payload["map_width"]),
            map_height=int(payload["map_height"]),
            seeds_count=int(payload["seeds_count"]),
            initial_states=initial_states,
            raw=payload,
        )

    def get_budget(self) -> RoundBudget:
        payload = self._request("GET", "/budget", auth_required=True)
        return RoundBudget(
            round_id=payload["round_id"],
            queries_used=int(payload["queries_used"]),
            queries_max=int(payload["queries_max"]),
            active=bool(payload["active"]),
        )

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> SimulationObservation:
        payload = self._request(
            "POST",
            "/simulate",
            payload={
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": viewport_x,
                "viewport_y": viewport_y,
                "viewport_w": viewport_w,
                "viewport_h": viewport_h,
            },
            auth_required=True,
        )
        grid = np.asarray(payload["grid"], dtype=np.int16)
        viewport = payload["viewport"]
        return SimulationObservation(
            round_id=round_id,
            seed_index=seed_index,
            viewport=Viewport(
                x=int(viewport["x"]),
                y=int(viewport["y"]),
                w=int(viewport["w"]),
                h=int(viewport["h"]),
            ),
            grid=grid,
            class_grid=terrain_grid_to_class_grid(grid),
            settlements=[Settlement.from_api(item) for item in payload.get("settlements", [])],
            queries_used=int(payload["queries_used"]),
            queries_max=int(payload["queries_max"]),
            raw=payload,
        )

    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/submit",
            payload={"round_id": round_id, "seed_index": seed_index, "prediction": prediction},
            auth_required=True,
        )

    def get_my_rounds(self) -> list[dict[str, Any]]:
        return self._request("GET", "/my-rounds", auth_required=True)

    def get_my_predictions(self, round_id: str) -> dict[str, Any]:
        return self._request("GET", f"/my-predictions/{round_id}", auth_required=True)

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        return self._request("GET", f"/analysis/{round_id}/{seed_index}", auth_required=True)
