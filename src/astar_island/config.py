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
    direct_observation_strength: float = 18.0
    transfer_strength: float = 8.0
    local_kernel_sigma: float = 4.0
    observation_smoothing_sigma: float = 1.4
    observation_support_temperature: float = 0.85
    observation_repeat_bonus: float = 0.45
    observation_bucket_blend: float = 0.28
    observation_nonactive_blend: float = 0.62
    observation_active_mass_blend: float = 0.24
    observation_active_type_blend: float = 0.18
    observation_bucket_blend_high_activity: float = 0.7
    observation_nonactive_blend_high_activity: float = 0.8
    observation_active_mass_blend_high_activity: float = 0.95
    observation_active_type_blend_high_activity: float = 0.8
    regime_settlement_rate_low: float = 0.12
    regime_settlement_rate_high: float = 0.24
    regime_forest_share_low: float = 0.38
    regime_forest_share_high: float = 0.62
    regime_repeat_fraction_low: float = 0.05
    regime_repeat_fraction_high: float = 0.12
    regime_settlement_weight: float = 0.65
    regime_forest_weight: float = 0.35
    regime_repeat_weight: float = 0.0
    active_dominance_sigma: float = 3.0
    active_dominance_signal_scale: float = 0.18
    active_dominance_base_ratio: float = 0.32
    active_dominance_support_gain: float = 0.95
    active_dominance_additive: float = 0.015
    active_dominance_initial_bonus: float = 0.18
    active_dominance_regime_relaxation: float = 0.35
    active_dominance_regime_initial_bonus: float = 0.08
    active_dominance_settlement_bonus_high_activity: float = 0.18
    active_dominance_settlement_margin_threshold: float = 0.14
    active_dominance_settlement_min_probability: float = 0.08
    active_dominance_port_bonus_high_activity: float = 0.08
    active_dominance_port_margin_threshold: float = 0.08
    active_dominance_port_min_probability: float = 0.03
    feature_match_strength: float = 5.0
    prior_strength_static: float = 60.0
    prior_strength_dynamic: float = 9.0
    settlement_sigma: float = 2.2
    settlement_intensity_blend: float = 0.22
    confidence_sharpen_power: float = 1.15
    active_confidence_sharpen_power: float = 1.02
    historical_prior_strength: float = 6.0
    forest_retention_boost: float = 1.5
    forest_empty_suppression: float = 0.55
    forest_settlement_suppression: float = 0.92
    forest_port_suppression: float = 0.88
    forest_ruin_suppression: float = 0.97
    forest_retention_boost_high_activity: float = 1.15
    forest_empty_suppression_high_activity: float = 0.6
    forest_settlement_suppression_high_activity: float = 1.08
    forest_port_suppression_high_activity: float = 0.98
    forest_ruin_suppression_high_activity: float = 1.0
    plains_settlement_base: float = 0.9
    plains_settlement_gain: float = 0.5
    plains_settlement_power: float = 3.0
    plains_support_intensity_weight: float = 0.85
    plains_empty_base: float = 1.12
    plains_empty_support_slope: float = 0.18
    plains_settlement_base_high_activity: float = 1.28
    plains_settlement_gain_high_activity: float = 1.6
    plains_empty_base_high_activity: float = 0.88
    plains_empty_support_slope_high_activity: float = 0.42
    coastal_port_support_gain: float = 1.2
    coastal_port_support_gain_high_activity: float = 1.35
    initial_settlement_boost: float = 1.05
    initial_settlement_empty_suppression: float = 0.97
    initial_settlement_boost_high_activity: float = 1.18
    initial_settlement_empty_suppression_high_activity: float = 0.9
    initial_port_boost: float = 1.12
    initial_port_empty_suppression: float = 0.95
    initial_port_boost_high_activity: float = 1.2
    initial_port_empty_suppression_high_activity: float = 0.9
    settlement_focus_blend_high_activity: float = 0.35
    settlement_focus_power_high_activity: float = 1.1
    settlement_support_intensity_weight: float = 0.45
    settlement_support_frontier_weight: float = 0.2
    settlement_support_density_weight: float = 0.15
    settlement_support_prediction_weight: float = 0.1
    settlement_support_initial_bonus: float = 0.1
    post_port_focus_blend_high_activity: float = 0.0
    post_port_focus_power_high_activity: float = 1.1
    post_port_support_intensity_weight: float = 0.4
    post_port_support_frontier_weight: float = 0.15
    post_port_support_border_weight: float = 0.2
    post_port_support_prediction_weight: float = 0.1
    post_port_support_initial_settlement_bonus: float = 0.08
    post_port_support_initial_port_bonus: float = 0.35
    rare_class_min_support: float = 1e-4
    port_focus_blend: float = 0.7
    port_focus_min_blend: float = 0.3
    port_focus_rate_floor: float = 0.015
    port_focus_rate_ceiling: float = 0.045
    port_focus_power: float = 1.0
    port_support_intensity_weight: float = 0.5
    port_support_frontier_weight: float = 0.22
    port_support_border_weight: float = 0.18
    port_support_predicted_active_weight: float = 0.10
    port_support_initial_settlement_bonus: float = 0.10
    port_support_initial_port_bonus: float = 0.25
    ruin_focus_blend: float = 0.8
    ruin_focus_power: float = 1.0
    ruin_support_intensity_weight: float = 0.38
    ruin_support_frontier_weight: float = 0.22
    ruin_support_conflict_weight: float = 0.20
    ruin_support_density_weight: float = 0.10
    ruin_support_interior_weight: float = 0.10
    ruin_support_predicted_settlement_weight: float = 0.10
    ruin_support_predicted_ruin_weight: float = 0.05
    ruin_support_forest_weight: float = 0.05
    learned_prior_blend: float = 0.25
    residual_calibrator_blend: float = 0.25
    residual_calibrator_single_observed_blend: float = 0.10
    residual_calibrator_repeated_observed_blend: float = 0.0
    residual_calibrator_active_observed_blend: float = 0.0
    residual_calibrator_high_activity_blend: float = 1.0
    residual_calibrator_high_activity_single_observed_blend: float = 1.0
    residual_calibrator_high_activity_repeated_observed_blend: float = 1.0
    residual_calibrator_high_activity_active_observed_blend: float = 1.0
    prior_blend_gate_strength: float = field(
        default_factory=lambda: float(os.getenv("ASTAR_ISLAND_PRIOR_BLEND_GATE_STRENGTH", "1.0"))
    )
    historical_prior_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_HISTORICAL_PRIOR_PATH", "artifacts/historical_priors.json"))
    )
    learned_prior_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_LEARNED_PRIOR_PATH", "artifacts/learned_prior.json"))
    )
    residual_calibrator_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_RESIDUAL_CALIBRATOR_PATH", "artifacts/residual_calibrator.joblib"))
    )
    prior_blend_gate_path: Path = field(
        default_factory=lambda: Path(os.getenv("ASTAR_ISLAND_PRIOR_BLEND_GATE_PATH", "artifacts/prior_blend_gate.joblib"))
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
