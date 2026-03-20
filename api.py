"""
REST API client for the Astar Island challenge.

Usage:
    api = AstarAPI(token=os.environ["ASTAR_TOKEN"])
    active = api.get_active_round()
    result = api.simulate(round_id, seed_index=0, vx=0, vy=0, vw=15, vh=15)
    api.submit(round_id, seed_index=0, prediction=tensor.tolist())
"""
from __future__ import annotations

import time
from typing import Any, Optional

import requests

from config import BASE_URL


class APIError(RuntimeError):
    """Non-recoverable API error (4xx client errors, parse failures)."""


class BudgetExhausted(APIError):
    """Raised when the 50-query budget is used up."""


class AstarAPI:
    def __init__(self, token: str, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    # ── Internal ──────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        max_retries: int = 3,
    ) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(max_retries):
            try:
                resp = self._session.request(method, url, json=payload, timeout=45)
            except requests.RequestException as exc:
                if attempt + 1 < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise APIError(f"Network error on {method} {path}: {exc}") from exc

            if resp.status_code == 429:
                text = resp.text or ""
                if "budget" in text.lower() or "exhausted" in text.lower() or "50/50" in text:
                    raise BudgetExhausted(f"Query budget exhausted: {text}")
                # Rate-limited — back off exponentially
                wait = 0.5 * (2 ** attempt)
                print(f"  Rate-limited (429), waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

            if resp.status_code >= 500:
                if attempt + 1 < max_retries:
                    print(f"  Server error {resp.status_code}, retrying...")
                    time.sleep(2.0)
                    continue
                raise APIError(f"Server error {resp.status_code} on {method} {path}: {resp.text[:200]}")

            if resp.status_code >= 400:
                raise APIError(f"HTTP {resp.status_code} on {method} {path}: {resp.text[:400]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise APIError(f"JSON parse error on {method} {path}: {exc}") from exc

        raise APIError(f"All {max_retries} attempts failed for {method} {path}")

    # ── Round discovery ───────────────────────────────────────────────────────

    def get_rounds(self) -> list[dict]:
        payload = self._request("GET", "/astar-island/rounds")
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            for key in ("rounds", "data", "items"):
                if isinstance(payload.get(key), list):
                    return [r for r in payload[key] if isinstance(r, dict)]
        raise APIError(f"Unexpected rounds format: {type(payload)}")

    def get_active_round(self) -> Optional[dict]:
        """Return the active round dict, or None if no round is active."""
        rounds = self.get_rounds()
        return next(
            (r for r in rounds if str(r.get("status", "")).lower() == "active"),
            None,
        )

    def get_round_detail(self, round_id: str) -> dict:
        """Full round details including initial_states for all seeds."""
        return self._request("GET", f"/astar-island/rounds/{round_id}")

    def get_budget(self) -> dict:
        """Current query budget: {round_id, queries_used, queries_max, active}."""
        return self._request("GET", "/astar-island/budget")

    # ── Core gameplay ─────────────────────────────────────────────────────────

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        vx: int,
        vy: int,
        vw: int = 15,
        vh: int = 15,
    ) -> dict:
        """
        Query the simulator for one viewport observation.

        Costs 1 query from the 50-query budget.
        Returns: {grid, settlements, viewport, width, height, queries_used, queries_max}
        """
        result = self._request("POST", "/astar-island/simulate", {
            "round_id":     round_id,
            "seed_index":   seed_index,
            "viewport_x":   vx,
            "viewport_y":   vy,
            "viewport_w":   vw,
            "viewport_h":   vh,
        })
        # Polite delay — API allows 5 req/s; stay well under that
        time.sleep(0.22)
        return result

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        """
        Submit H×W×6 prediction tensor for one seed.

        prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
        Each inner list must sum to 1.0 (±0.01 tolerance).
        """
        result = self._request("POST", "/astar-island/submit", {
            "round_id":   round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        })
        time.sleep(0.55)  # API allows 2 req/s
        return result

    # ── Analysis & leaderboard ────────────────────────────────────────────────

    def get_my_rounds(self) -> list[dict]:
        return self._request("GET", "/astar-island/my-rounds")

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        """
        Post-round ground truth comparison (only available after round completes).
        Returns: {prediction, ground_truth, score, width, height, initial_grid}
        """
        return self._request("GET", f"/astar-island/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self) -> list[dict]:
        return self._request("GET", "/astar-island/leaderboard")

    def get_my_predictions(self, round_id: str) -> list[dict]:
        return self._request("GET", f"/astar-island/my-predictions/{round_id}")
