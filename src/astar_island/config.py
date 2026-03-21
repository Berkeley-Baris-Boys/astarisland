from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QueryPlannerConfig:
    max_queries: int = 50
    min_window: int = 5
    max_window: int = 15
    phase1_queries: int = 20
    phase2_queries: int = 22
    phase3_queries: int = 8
    overlap_penalty: float = 0.7
    repeat_bonus: float = 0.2
    repeat_threshold: float = 0.75
    candidate_stride: int = 4
    per_seed_soft_min: int = 7
    repeated_view_window: int = 9


@dataclass
class PredictorConfig:
    min_probability: float = 0.0025
    direct_observation_strength: float = 35.0
    transfer_strength: float = 8.0
    local_kernel_sigma: float = 4.0
    feature_match_strength: float = 5.0
    prior_strength_static: float = 60.0
    prior_strength_dynamic: float = 9.0
    settlement_sigma: float = 2.2
    settlement_intensity_blend: float = 0.22
    confidence_sharpen_power: float = 1.15
    historical_prior_strength: float = 30.0
    historical_prior_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_HISTORICAL_PRIOR_PATH", "artifacts/historical_priors.json"))
    )


@dataclass
class CacheConfig:
    enabled: bool = True
    directory: Path = field(default_factory=lambda: Path(".cache/astar_island"))


@dataclass
class AstarConfig:
    base_url: str = "https://api.ainm.no/astar-island"
    token: str | None = field(default_factory=lambda: os.getenv("ASTAR_ISLAND_TOKEN"))
    auth_mode: str = field(default_factory=lambda: os.getenv("ASTAR_ISLAND_AUTH_MODE", "bearer"))
    user_agent: str = "astar-island-codex-baseline/0.1"
    request_timeout_s: float = 30.0
    max_retries: int = 5
    retry_backoff_s: float = 1.0
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_LOG_DIR", "artifacts")))
    history_dir: Path = field(default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_HISTORY_DIR", "artifacts/history")))
    cache: CacheConfig = field(default_factory=CacheConfig)
    planner: QueryPlannerConfig = field(default_factory=QueryPlannerConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.cache.enabled:
            self.cache.directory.mkdir(parents=True, exist_ok=True)
