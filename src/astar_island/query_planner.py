from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import QueryPlannerConfig
from .types import QueryPlanStep, RoundDetail, SeedFeatures, Viewport


@dataclass
class CandidateWindow:
    seed_index: int
    viewport: Viewport
    score: float
    reason: str


class QueryPlanner:
    def __init__(self, config: QueryPlannerConfig, round_detail: RoundDetail, seed_features: dict[int, SeedFeatures]):
        self.config = config
        self.round_detail = round_detail
        self.seed_features = seed_features

    def next_step(self, aggregator) -> QueryPlanStep:
        used_queries = len(aggregator.query_history)
        phase = self._phase_name(used_queries)
        # Estimate round activity so we can modulate weights dynamically.
        activity_scale = self._estimate_activity_scale(aggregator)
        candidates = self._generate_candidates(aggregator, phase, activity_scale)
        best = max(candidates, key=lambda item: item.score)
        return QueryPlanStep(
            seed_index=best.seed_index,
            viewport=best.viewport,
            reason=best.reason,
            score=best.score,
            phase=phase,
        )

    def _estimate_activity_scale(self, aggregator) -> float:
        """Compute settlement activity relative to historical average (~10%)."""
        counts = aggregator.class_counts.sum(axis=(0, 1, 2))
        total = float(counts.sum())
        if total < 100.0:
            return 1.0  # Too early to estimate; use neutral scale
        from .types import CLASS_SETTLEMENT, CLASS_PORT
        settlement_mass = float(counts[CLASS_SETTLEMENT] + counts[CLASS_PORT])
        rate = settlement_mass / total
        import numpy as _np
        return float(_np.clip(rate / 0.10, 0.05, 1.0))

    def _phase_name(self, used_queries: int) -> str:
        if used_queries < self.config.phase1_queries:
            return "recon"
        if used_queries < self.config.phase1_queries + self.config.phase2_queries:
            return "targeted"
        return "calibration"

    def _generate_candidates(self, aggregator, phase: str, activity_scale: float = 1.0) -> list[CandidateWindow]:
        candidates: list[CandidateWindow] = []
        for seed_index, features in self.seed_features.items():
            window = self.config.max_window if phase != "calibration" else self.config.repeated_view_window
            stride = self.config.candidate_stride if phase != "calibration" else max(2, self.config.candidate_stride - 1)
            for y in range(0, self.round_detail.map_height - window + 1, stride):
                for x in range(0, self.round_detail.map_width - window + 1, stride):
                    viewport = Viewport(x=x, y=y, w=window, h=window)
                    score, reason = self._score_window(seed_index, viewport, features, aggregator, phase, activity_scale)
                    candidates.append(CandidateWindow(seed_index=seed_index, viewport=viewport, score=score, reason=reason))
        return candidates

    def _score_window(self, seed_index: int, viewport: Viewport, features: SeedFeatures, aggregator, phase: str, activity_scale: float = 1.0) -> tuple[float, str]:
        y0, y1 = viewport.y, viewport.y + viewport.h
        x0, x1 = viewport.x, viewport.x + viewport.w
        frontier = float(features.frontier_mask[y0:y1, x0:x1].mean())
        conflict = float(features.conflict_mask[y0:y1, x0:x1].mean())
        coastal = float(features.coastal_mask[y0:y1, x0:x1].mean())
        initial_active = float(features.initial_settlement_mask[y0:y1, x0:x1].mean())
        observed = float(aggregator.observation_counts[seed_index, y0:y1, x0:x1].mean())
        entropy = float(aggregator.observed_entropy(seed_index)[y0:y1, x0:x1].mean())
        repeat_value = 1.0 / (1.0 + observed)
        coverage_value = math.exp(-observed)
        query_count_seed = sum(1 for step in aggregator.query_history if step.seed_index == seed_index)
        fairness_bonus = 0.4 if query_count_seed < self.config.per_seed_soft_min else 0.0
        # Scale frontier/initial_active signals down for low-activity rounds; boost pure coverage.
        frontier_w = min(activity_scale, 1.0)
        if phase == "recon":
            score = (2.0 + (1.0 - frontier_w)) * coverage_value + 1.8 * frontier_w * frontier + 1.2 * coastal + 1.4 * frontier_w * initial_active + 0.8 * fairness_bonus
            reason = "broad coverage of likely-active region"
        elif phase == "targeted":
            score = 1.6 * frontier_w * frontier + 1.5 * frontier_w * conflict + 1.1 * coastal + 1.3 * entropy + (0.7 + (1.0 - frontier_w)) * repeat_value + fairness_bonus
            reason = "target high-value frontier/conflict cells"
        else:
            score = 1.8 * entropy + 1.1 * repeat_value + self.config.repeat_bonus * (frontier + conflict + coastal)
            reason = "repeat sample for uncertainty calibration"
        overlap_penalty = self._overlap_penalty(seed_index, viewport, aggregator)
        return score - overlap_penalty, reason

    def _overlap_penalty(self, seed_index: int, viewport: Viewport, aggregator) -> float:
        penalty = 0.0
        for step in aggregator.query_history:
            if step.seed_index != seed_index:
                continue
            x_overlap = max(0, min(viewport.x + viewport.w, step.viewport.x + step.viewport.w) - max(viewport.x, step.viewport.x))
            y_overlap = max(0, min(viewport.y + viewport.h, step.viewport.y + step.viewport.h) - max(viewport.y, step.viewport.y))
            overlap_area = x_overlap * y_overlap
            if overlap_area == 0:
                continue
            fraction = overlap_area / max(viewport.area(), 1)
            if step.phase == "calibration":
                penalty += self.config.overlap_penalty * 0.25 * fraction
            else:
                penalty += self.config.overlap_penalty * fraction
        return penalty
