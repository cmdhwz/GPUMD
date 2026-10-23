#!/usr/bin/env python3
"""Executable policy arithmetic checks; the C++ header is source-checked only."""

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

if __package__:
    from . import rpmd_ja_cpp_test_support as cpp_test_support
else:
    import rpmd_ja_cpp_test_support as cpp_test_support


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "measure" / "rpmd_ja_feasibility_policy.cuh"
CPP_TEST = ROOT / "tests" / "rpmd_ja_feasibility_policy_test.cpp"
CXX = cpp_test_support.find_cpp_compiler()


class Phase(Enum):
    PREFLIGHT = 0
    A0 = 1
    A1 = 2
    B = 3
    C = 4
    FREEZE = 5
    BLIND = 6
    PREPARE = 7


@dataclass
class Limits:
    wall: float = 8 * 60 * 60
    gradients: int = 240
    validation_fraction: float = 0.25
    a_branch_gradients: int = 8
    fit_family_gradients: int = 112
    a_wall_fraction: float = 0.10
    family_wall: float = -1.0

    def valid(self):
        values = (self.wall, self.validation_fraction, self.a_wall_fraction)
        return (
            all(math.isfinite(value) for value in values)
            and self.wall > 0
            and self.gradients > 0
            and 0 <= self.validation_fraction < 1
            and self.a_branch_gradients > 0
            and self.fit_family_gradients >= 0
            and 0 < self.a_wall_fraction <= 1
            and math.isfinite(self.family_wall) and self.family_wall >= -1.0
        )


@dataclass
class Budget:
    started: bool = False
    start: float = 0.0
    last: float = 0.0
    elapsed: float = 0.0
    exact_gradients: int = 0
    phase_gradients: dict = field(default_factory=dict)
    phase_seconds: dict = field(default_factory=dict)
    phase: Phase = Phase.PREFLIGHT

    def begin(self, limits, now):
        if not limits.valid():
            return "INVALID_LIMITS"
        if not math.isfinite(now):
            return "INVALID_CLOCK"
        if self.started:
            return "ALREADY_STARTED"
        self.started = True
        self.start = self.last = now
        return "OK"

    def observe(self, now):
        if not math.isfinite(now):
            return "INVALID_CLOCK"
        if now < self.last:
            return "CLOCK_BACKWARDS"
        delta = now - self.last
        self.elapsed += delta
        self.phase_seconds[self.phase] = self.phase_seconds.get(self.phase, 0) + delta
        self.last = now
        return "OK"

    def enter(self, limits, phase, now):
        if not limits.valid():
            return "INVALID_LIMITS"
        if not self.started:
            return "NOT_STARTED"
        result = self.observe(now)
        if result != "OK":
            return result
        self.phase = phase
        return self.phase_gate(limits)

    def phase_gate(self, limits):
        if self.elapsed >= limits.wall:
            return "WALL_LIMIT"
        if self.phase in (Phase.A0, Phase.A1):
            spent = self.phase_seconds.get(Phase.A0, 0) + self.phase_seconds.get(Phase.A1, 0)
            if spent >= limits.wall * limits.a_wall_fraction:
                return "A_WALL_LIMIT"
        if self.phase in (Phase.B, Phase.C) and limits.family_wall >= 0 and \
                self.phase_seconds.get(self.phase, 0) >= limits.family_wall:
            return "FAMILY_WALL_LIMIT"
        protected = limits.wall * limits.validation_fraction
        if self.phase not in (Phase.FREEZE, Phase.BLIND, Phase.PREPARE) and limits.wall - self.elapsed <= protected:
            return "VALIDATION_RESERVED"
        return "OK"

    def observe_time(self, limits, now):
        if not limits.valid():
            return "INVALID_LIMITS"
        if not self.started:
            return "NOT_STARTED"
        result = self.observe(now)
        return self.phase_gate(limits) if result == "OK" else result

    def reserve(self, limits, now):
        result = self.observe_time(limits, now)
        if result != "OK":
            return result
        if self.exact_gradients >= limits.gradients:
            return "GRADIENT_LIMIT"
        if self.phase in (Phase.A0, Phase.A1):
            if self.phase_gradients.get(self.phase, 0) >= limits.a_branch_gradients:
                return "PHASE_GRADIENT_LIMIT"
        elif self.phase in (Phase.B, Phase.C):
            if self.phase_gradients.get(self.phase, 0) >= limits.fit_family_gradients:
                return "FAMILY_GRADIENT_LIMIT"
        self.exact_gradients += 1
        self.phase_gradients[self.phase] = self.phase_gradients.get(self.phase, 0) + 1
        return "OK"


@dataclass
class CandidateCostProfile:
    samples: int = 0
    maximum_seconds: float = 0.0
    partial_elapsed_lower_bound_seconds: float = 0.0


def record_candidate_cost(profile, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        return False
    profile.maximum_seconds = max(profile.maximum_seconds, seconds)
    profile.samples += 1
    return True


def record_partial_candidate_elapsed(profile, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        return False
    profile.partial_elapsed_lower_bound_seconds = max(profile.partial_elapsed_lower_bound_seconds, seconds)
    return True


def measured_candidate_cost_gate(budget, limits, profile, safety_factor):
    if not limits.valid() or not math.isfinite(safety_factor) or not 1 <= safety_factor <= 10:
        return "INVALID_ESTIMATE", 0.0, 0.0
    if profile.samples == 0 and profile.partial_elapsed_lower_bound_seconds <= 0:
        return "NO_MEASURED_BASELINE", 0.0, 0.0
    projected = max(profile.maximum_seconds, profile.partial_elapsed_lower_bound_seconds) * safety_factor
    if not math.isfinite(projected):
        return "INVALID_ESTIMATE", 0.0, 0.0
    reserved = limits.wall * limits.validation_fraction
    available = max(0.0, limits.wall - budget.elapsed - reserved)
    if budget.phase in (Phase.B, Phase.C) and limits.family_wall >= 0:
        available = min(available, max(0.0, limits.family_wall - budget.phase_seconds.get(budget.phase, 0.0)))
    return ("ALLOW" if projected <= available else "INSUFFICIENT_REMAINING_FIT_TIME", available, projected)


class Family(Enum):
    M0 = 0
    M1 = 1


@dataclass
class Candidate:
    family: Family
    index: int
    converged: bool
    score: float
    uncertainty: float
    block_scores: list[float] = field(default_factory=list)


@dataclass
class FrozenSelection:
    frozen: bool = False
    blind_opened: bool = False
    family: Family = Family.M0
    index: int = 0


def choose(candidates, selected):
    if selected.frozen:
        return "ALREADY_FROZEN"
    if sum(c.family is Family.M0 for c in candidates) > 3 or sum(
        c.family is Family.M1 for c in candidates
    ) > 3:
        return "TOO_MANY_CANDIDATES"
    eligible = [
        c for c in candidates
        if c.converged and math.isfinite(c.score)
        and math.isfinite(c.uncertainty) and c.uncertainty >= 0
        and len(c.block_scores) >= 4
        and all(math.isfinite(score) for score in c.block_scores)
    ]
    if not eligible:
        return "NO_ELIGIBLE"
    block_count = len(eligible[0].block_scores)
    if any(len(candidate.block_scores) != block_count for candidate in eligible):
        return "INVALID_PAIRED_BLOCKS"
    unbeaten = []
    for candidate in eligible:
        beaten = False
        for other in eligible:
            if candidate is other:
                continue
            differences = [candidate.block_scores[i] - other.block_scores[i] for i in range(block_count)]
            mean = sum(differences) / block_count
            variance = sum((value - mean) ** 2 for value in differences) / (block_count - 1)
            uncertainty = 2 * math.sqrt(variance / block_count)
            if mean > uncertainty:
                beaten = True
                break
        if not beaten:
            unbeaten.append(candidate)
    if not unbeaten:
        return "NO_PAIRED_WINNER"
    best = min(unbeaten, key=lambda candidate: (
        0 if candidate.family is Family.M0 else 1, candidate.score, candidate.index
    ))
    selected.frozen = True
    selected.family, selected.index = best.family, best.index
    return "SELECTED"


def open_blind_once(selected):
    if not selected.frozen or selected.blind_opened:
        return False
    selected.blind_opened = True
    return True


def should_run_structured_family(has_additions, m0_common_probe_pass):
    return has_additions and not m0_common_probe_pass


def common_soft_probe_capacity(internal_dimension, validation_frames, maximum_soft_modes):
    return min(max(0, internal_dimension), max(0, validation_frames - 1), max(0, maximum_soft_modes))


def blind_status(selected, independent_data, passed):
    if not selected.frozen or not selected.blind_opened:
        return "DEVELOPMENT_CANDIDATE"
    if not independent_data:
        return "DATA_INCONCLUSIVE"
    return "BLIND_PASSED" if passed else "BLIND_FAILED"


def sensitivity_allows_blind(sensitive, inconclusive):
    return not sensitive and not inconclusive


def can_compare_a_tau1(tau_a0, tau_a1):
    return math.isfinite(tau_a0) and math.isfinite(tau_a1) and tau_a0 == 1.0 and tau_a1 == 1.0


def equal_remaining_family_gradient_cap(budget, limits):
    return max(0, limits.gradients - budget.exact_gradients) // 2


def equal_remaining_family_wall_cap(budget, limits):
    remaining = max(0, limits.wall - budget.elapsed - limits.wall * limits.validation_fraction)
    return remaining / 2


class FeasibilityPolicyTests(unittest.TestCase):
    def test_b_failing_common_probe_rescore_starts_c(self):
        initial_b_score_passed = True
        common_probe_b_score_passed = False
        self.assertTrue(initial_b_score_passed)
        self.assertTrue(should_run_structured_family(True, common_probe_b_score_passed))
        self.assertFalse(should_run_structured_family(True, True))

    def test_tight_validation_frame_count_can_prioritize_full_soft_rank(self):
        self.assertEqual(common_soft_probe_capacity(3, 4, 28), 3)
        self.assertEqual(common_soft_probe_capacity(8, 4, 28), 3)
        self.assertEqual(common_soft_probe_capacity(3, 1, 28), 0)

    def test_default_caps_are_the_run_contract(self):
        limits = Limits()
        self.assertEqual((limits.wall, limits.gradients), (28800, 240))
        self.assertEqual((limits.validation_fraction, limits.a_wall_fraction), (0.25, 0.10))
        self.assertEqual(limits.fit_family_gradients, 112)
        self.assertEqual(limits.fit_family_gradients, 112)

    def test_invalid_limits_and_clock_are_rejected(self):
        for limits in (
            Limits(wall=-1),
            Limits(wall=float("nan")),
            Limits(validation_fraction=float("nan")),
            Limits(validation_fraction=-0.1),
            Limits(gradients=0),
            Limits(gradients=-1),
            Limits(a_branch_gradients=-1),
            Limits(a_wall_fraction=float("inf")),
        ):
            self.assertEqual(Budget().begin(limits, 0), "INVALID_LIMITS")
        self.assertEqual(Budget().begin(Limits(), float("inf")), "INVALID_CLOCK")

    def test_solver_rejection_still_consumes_reserved_gradient(self):
        budget, limits = Budget(), Limits(gradients=1)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.reserve(limits, 1), "OK")
        # A later numeric rejection has no release path; the next attempt is denied.
        self.assertEqual(budget.reserve(limits, 2), "GRADIENT_LIMIT")
        self.assertEqual(budget.exact_gradients, 1)

    def test_begin_cannot_reset_a_live_budget(self):
        budget, limits = Budget(), Limits(gradients=2)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.reserve(limits, 1), "OK")
        self.assertEqual(budget.begin(limits, 5), "ALREADY_STARTED")
        self.assertEqual((budget.elapsed, budget.exact_gradients), (1, 1))

    def test_default_exact_gradient_limit_is_240(self):
        budget, limits = Budget(), Limits()
        budget.begin(limits, 0)
        for _ in range(240):
            self.assertEqual(budget.reserve(limits, 0), "OK")
        self.assertEqual(budget.reserve(limits, 0), "GRADIENT_LIMIT")
        self.assertEqual(budget.exact_gradients, 240)

    def test_branch_switch_preserves_global_gradient_count(self):
        budget, limits = Budget(), Limits(gradients=2)
        budget.begin(limits, 0)
        self.assertEqual(budget.reserve(limits, 1), "OK")
        self.assertEqual(budget.enter(limits, Phase.B, 2), "OK")
        self.assertEqual(budget.reserve(limits, 3), "OK")
        self.assertEqual(budget.enter(limits, Phase.C, 4), "OK")
        self.assertEqual(budget.reserve(limits, 5), "GRADIENT_LIMIT")
        self.assertEqual(budget.exact_gradients, 2)

    def test_b_and_c_share_equal_family_gradient_caps(self):
        budget, limits = Budget(), Limits(gradients=20, fit_family_gradients=3)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(equal_remaining_family_gradient_cap(budget, limits), 10)
        self.assertEqual(budget.enter(limits, Phase.B, 1), "OK")
        self.assertEqual([budget.reserve(limits, 1) for _ in range(3)], ["OK"] * 3)
        self.assertEqual(budget.reserve(limits, 1), "FAMILY_GRADIENT_LIMIT")
        self.assertEqual(budget.enter(limits, Phase.C, 2), "OK")
        self.assertEqual([budget.reserve(limits, 2) for _ in range(3)], ["OK"] * 3)
        self.assertEqual((budget.exact_gradients, equal_remaining_family_gradient_cap(budget, limits)), (6, 7))

    def test_b_and_c_receive_equal_remaining_wall_caps_and_validation_reserve(self):
        budget, limits = Budget(), Limits(wall=100, validation_fraction=0.25)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.enter(limits, Phase.A1, 10), "OK")
        limits.family_wall = equal_remaining_family_wall_cap(budget, limits)
        self.assertEqual(limits.family_wall, 32.5)
        self.assertEqual(budget.enter(limits, Phase.B, 10), "OK")
        self.assertEqual(budget.observe_time(limits, 36), "OK")  # B candidate fit.
        self.assertEqual(budget.observe_time(limits, 42.5), "FAMILY_WALL_LIMIT")  # B common-probe gate.
        self.assertEqual(budget.enter(limits, Phase.C, 42.5), "OK")
        self.assertEqual(budget.reserve(limits, 42.5), "OK")
        self.assertEqual(budget.observe_time(limits, 75), "FAMILY_WALL_LIMIT")
        self.assertEqual(budget.enter(limits, Phase.FREEZE, 75), "OK")
        self.assertGreaterEqual(limits.wall - budget.elapsed, limits.wall * limits.validation_fraction)

    def test_single_uninterruptible_b_gate_step_overrun_is_accounted(self):
        budget, limits = Budget(), Limits(wall=100, validation_fraction=0.25)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.enter(limits, Phase.A1, 10), "OK")
        limits.family_wall = equal_remaining_family_wall_cap(budget, limits)
        self.assertEqual(limits.family_wall, 32.5)
        self.assertEqual(budget.enter(limits, Phase.B, 10), "OK")
        self.assertEqual(budget.observe_time(limits, 42), "OK")
        self.assertEqual(budget.observe_time(limits, 43.25), "FAMILY_WALL_LIMIT")
        overrun = budget.phase_seconds[Phase.B] - limits.family_wall
        self.assertEqual(overrun, 0.75)
        self.assertEqual(budget.enter(limits, Phase.C, 43.25), "OK")
        c_available = limits.wall - budget.elapsed - limits.wall * limits.validation_fraction
        self.assertEqual(c_available, limits.family_wall - overrun)

    def test_measured_candidate_cost_stops_before_spending_c_family_allowance(self):
        budget, limits = Budget(), Limits(wall=100, validation_fraction=0.25, family_wall=20)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.enter(limits, Phase.B, 10), "OK")
        profile = CandidateCostProfile()
        self.assertTrue(record_candidate_cost(profile, 12))
        self.assertTrue(record_candidate_cost(profile, 8))
        status, available, projected = measured_candidate_cost_gate(budget, limits, profile, 1.5)
        self.assertEqual((status, available, projected), ("ALLOW", 20, 18))
        self.assertEqual(budget.observe_time(limits, 30), "FAMILY_WALL_LIMIT")
        status, available, projected = measured_candidate_cost_gate(budget, limits, profile, 1.5)
        self.assertEqual((status, available, projected), ("INSUFFICIENT_REMAINING_FIT_TIME", 0, 18))
        self.assertEqual(budget.enter(limits, Phase.C, 30), "OK")
        status, available, projected = measured_candidate_cost_gate(budget, limits, profile, 1.5)
        self.assertEqual((status, available, projected), ("ALLOW", 20, 18))
        self.assertFalse(record_candidate_cost(profile, float("nan")))
        self.assertFalse(record_candidate_cost(profile, 0.0))
        self.assertEqual(measured_candidate_cost_gate(budget, limits, profile, 11)[0], "INVALID_ESTIMATE")

    def test_expensive_incomplete_candidate_blocks_next_same_family_candidate(self):
        budget, limits = Budget(), Limits(wall=500, validation_fraction=0.25, family_wall=200)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(budget.enter(limits, Phase.B, 5), "OK")
        self.assertEqual(budget.observe_time(limits, 125), "OK")
        profile = CandidateCostProfile()
        self.assertTrue(record_candidate_cost(profile, 5))  # completed A baseline
        self.assertTrue(record_partial_candidate_elapsed(profile, 120))  # incomplete B0 lower bound
        status, available, projected = measured_candidate_cost_gate(budget, limits, profile, 1.5)
        self.assertEqual((status, available, projected), ("INSUFFICIENT_REMAINING_FIT_TIME", 80, 180))
        self.assertEqual(profile.samples, 1)  # partial time is not a completed sample
        self.assertEqual(profile.maximum_seconds, 5)
        self.assertEqual(profile.partial_elapsed_lower_bound_seconds, 120)
        self.assertEqual(budget.enter(limits, Phase.C, 125), "OK")
        self.assertEqual(measured_candidate_cost_gate(budget, limits, profile, 1.5)[0], "ALLOW")

    def test_b_and_c_share_equal_family_gradient_caps(self):
        budget, limits = Budget(), Limits(gradients=20, fit_family_gradients=3)
        self.assertEqual(budget.begin(limits, 0), "OK")
        self.assertEqual(equal_remaining_family_gradient_cap(budget, limits), 10)
        self.assertEqual(budget.enter(limits, Phase.B, 1), "OK")
        self.assertEqual([budget.reserve(limits, 1) for _ in range(3)], ["OK"] * 3)
        self.assertEqual(budget.reserve(limits, 1), "FAMILY_GRADIENT_LIMIT")
        self.assertEqual(budget.enter(limits, Phase.C, 2), "OK")
        self.assertEqual([budget.reserve(limits, 2) for _ in range(3)], ["OK"] * 3)
        self.assertEqual((budget.exact_gradients, equal_remaining_family_gradient_cap(budget, limits)), (6, 7))

    def test_validation_reservation_blocks_fit_but_allows_blind(self):
        budget, limits = Budget(), Limits()
        budget.begin(limits, 0)
        self.assertEqual(budget.enter(limits, Phase.B, 1), "OK")
        self.assertEqual(budget.observe_time(limits, 21600), "VALIDATION_RESERVED")
        self.assertEqual(budget.reserve(limits, 21600), "VALIDATION_RESERVED")
        self.assertEqual(budget.enter(limits, Phase.FREEZE, 21600), "OK")
        self.assertEqual(budget.enter(limits, Phase.BLIND, 21600), "OK")
        self.assertEqual(budget.reserve(limits, 21600), "OK")

    def test_backward_and_nonfinite_clock_do_not_advance_budget(self):
        budget, limits = Budget(), Limits()
        budget.begin(limits, 10)
        self.assertEqual(budget.reserve(limits, float("nan")), "INVALID_CLOCK")
        self.assertEqual(budget.reserve(limits, 9), "CLOCK_BACKWARDS")
        self.assertEqual((budget.last, budget.elapsed, budget.exact_gradients), (10, 0, 0))

    def test_a_branches_share_the_wall_cap_and_keep_eight_each(self):
        budget, limits = Budget(), Limits()
        budget.begin(limits, 0)
        for i in range(8):
            self.assertEqual(budget.enter(limits, Phase.A0, i), "OK")
            self.assertEqual(budget.reserve(limits, i), "OK")
        self.assertEqual(budget.reserve(limits, 8), "PHASE_GRADIENT_LIMIT")
        self.assertEqual(budget.enter(limits, Phase.A1, 9), "OK")
        self.assertEqual(budget.reserve(limits, 9), "OK")
        for _ in range(7):
            self.assertEqual(budget.reserve(limits, 9), "OK")
        self.assertEqual(budget.reserve(limits, 9), "PHASE_GRADIENT_LIMIT")
        self.assertEqual(budget.enter(limits, Phase.B, 2880), "OK")
        self.assertEqual(budget.enter(limits, Phase.A1, 2881), "A_WALL_LIMIT")
        self.assertEqual(budget.reserve(limits, 2881), "A_WALL_LIMIT")

    def test_numerically_unconverged_candidates_cannot_win(self):
        selected = FrozenSelection()
        candidates = [
            Candidate(Family.M0, 0, False, 0.01, 0),
            Candidate(Family.M1, 0, True, 0.02, 0, [0.02] * 4),
        ]
        self.assertEqual(choose(candidates, selected), "SELECTED")
        self.assertEqual((selected.family, selected.index), (Family.M1, 0))

    def test_overlapping_candidate_scores_prefer_m0(self):
        selected = FrozenSelection()
        candidates = [
            Candidate(Family.M1, 0, True, 0.01, 0.02, [0.0] * 4),
            Candidate(Family.M0, 0, True, 0.02, 0.02, [0.0] * 4),
        ]
        self.assertEqual(choose(candidates, selected), "SELECTED")
        self.assertEqual(selected.family, Family.M0)

    def test_cross_family_selection_uses_resolved_paired_block_difference(self):
        selected = FrozenSelection()
        candidates = [
            Candidate(Family.M0, 0, True, 50.0, 57.8, [0.0, 100.0, 0.0, 100.0]),
            Candidate(Family.M1, 0, True, 10.0, 57.8, [-40.0, 60.0, -40.0, 60.0]),
        ]
        self.assertEqual(choose(candidates, selected), "SELECTED")
        self.assertEqual((selected.family, selected.index), (Family.M1, 0))

    def test_selector_low_score_wins_independent_of_candidate_order(self):
        cases = (
            [Candidate(Family.M0, 0, True, 10, 0, [10] * 4),
             Candidate(Family.M1, 0, True, 1, 0, [1] * 4)],
            [Candidate(Family.M1, 0, True, 1, 0, [1] * 4),
             Candidate(Family.M0, 0, True, 10, 0, [10] * 4)],
            [Candidate(Family.M0, 0, True, 10, 0, [10] * 4),
             Candidate(Family.M0, 1, True, 5, 0, [5] * 4),
             Candidate(Family.M1, 0, True, 1, 0, [1] * 4)],
            [Candidate(Family.M1, 0, True, 1, 0, [1] * 4),
             Candidate(Family.M0, 0, True, 10, 0, [10] * 4),
             Candidate(Family.M0, 1, True, 5, 0, [5] * 4)],
        )
        for candidates in cases:
            with self.subTest(candidates=[(c.family, c.index) for c in candidates]):
                selected = FrozenSelection()
                self.assertEqual(choose(candidates, selected), "SELECTED")
                self.assertEqual((selected.family, selected.index), (Family.M1, 0))

    def test_unresolved_cross_family_paired_difference_prefers_m0(self):
        selected = FrozenSelection()
        candidates = [
            Candidate(Family.M1, 0, True, 1.0, 1.0, [2.0, -2.0, 2.0, -2.0]),
            Candidate(Family.M0, 0, True, 2.0, 1.0, [0.0, 0.0, 0.0, 0.0]),
        ]
        self.assertEqual(choose(candidates, selected), "SELECTED")
        self.assertEqual(selected.family, Family.M0)

    def test_candidate_without_common_paired_blocks_is_ineligible(self):
        selected = FrozenSelection()
        result = choose([Candidate(Family.M0, 0, True, 0.0, 0.0)], selected)
        self.assertEqual(result, "NO_ELIGIBLE")

    def test_differently_sized_paired_block_ledgers_are_rejected(self):
        selected = FrozenSelection()
        candidates = [
            Candidate(Family.M0, 0, True, 1.0, 0.0, [1.0, 1.0, 1.0, 1.0]),
            Candidate(Family.M1, 0, True, 0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        self.assertEqual(choose(candidates, selected), "INVALID_PAIRED_BLOCKS")

    def test_candidate_cap_and_nonfinite_scores_are_ineligible(self):
        selected = FrozenSelection()
        excess = [Candidate(Family.M0, i, True, 1, 0) for i in range(4)]
        self.assertEqual(choose(excess, selected), "TOO_MANY_CANDIDATES")
        self.assertEqual(choose([Candidate(Family.M0, 0, True, float("nan"), 0)], selected), "NO_ELIGIBLE")

    def test_blind_choice_is_one_shot_and_missing_data_stays_nonproduction(self):
        selected = FrozenSelection()
        self.assertEqual(choose([Candidate(Family.M0, 1, True, 0.1, 0, [0.1] * 4)], selected), "SELECTED")
        self.assertTrue(open_blind_once(selected))
        self.assertFalse(open_blind_once(selected))
        self.assertEqual(blind_status(selected, False, True), "DATA_INCONCLUSIVE")
        self.assertEqual(choose([Candidate(Family.M1, 0, True, 0, 0)], selected), "ALREADY_FROZEN")
        self.assertEqual(blind_status(selected, True, False), "BLIND_FAILED")

    def test_blind_access_requires_resolved_regularization_sensitivity(self):
        self.assertFalse(sensitivity_allows_blind(True, False))
        self.assertFalse(sensitivity_allows_blind(False, True))
        self.assertTrue(sensitivity_allows_blind(False, False))

    def test_a_comparison_is_only_meaningful_at_fixed_tau_one(self):
        self.assertTrue(can_compare_a_tau1(1.0, 1.0))
        self.assertFalse(can_compare_a_tau1(1.0, 0.2))
        self.assertFalse(can_compare_a_tau1(float("nan"), 1.0))

    def test_performance_stop_heuristic_is_explicit_and_tau_scoped(self):
        source = HEADER.read_text(encoding="utf-8")
        self.assertIn("candidate_performance_curvature_window=3", source)
        self.assertIn("candidate_performance_min_relative_objective_improvement=0.001", source)
        self.assertIn("candidate_performance_curvature_to_gradient_cost_ratio=1.0", source)
        self.assertIn("window.tau!=tau", source)
        self.assertIn("window.curvature_seconds>candidate_performance_curvature_to_gradient_cost_ratio*window.reference_gradient_seconds", source)
        self.assertIn("relative_improvement<candidate_performance_min_relative_objective_improvement", source)
        self.assertIn("std::max(1.0,std::abs(window.start_objective))", source)
        self.assertIn("if(!stop)reset_curvature_performance_window(window,tau,current_objective,next_window_gradient_seconds)", source)

    def test_header_exposes_host_only_injected_clock_api(self):
        source = HEADER.read_text(encoding="utf-8")
        selector = source.split("inline SelectionResult choose_development_candidate(", 1)[1].split(
            "inline bool open_blind_once(", 1
        )[0]
        self.assertIn("pair==PairedCandidateResult::RIGHT_WINS", selector)
        self.assertNotIn("pair==PairedCandidateResult::LEFT_WINS", selector)
        for signature in (
            "inline BudgetResult begin(",
            "inline BudgetResult observe_time(",
            "inline BudgetResult enter_phase(",
            "inline BudgetResult reserve_exact_gradient(",
            "inline int equal_remaining_family_gradient_cap(",
            "inline double equal_remaining_family_wall_cap(",
            "inline int equal_remaining_family_gradient_cap(",
            "inline SelectionResult choose_development_candidate(",
            "inline bool open_blind_once(",
            "inline bool regularization_sensitivity_allows_blind(",
            "inline BlindStatus classify_blind_result(",
            "inline bool can_compare_a_tau1(",
            "double now_seconds",
        ):
            self.assertIn(signature, source)
        self.assertNotIn("std::chrono", source)
        self.assertNotIn("cuda_runtime", source)

    def test_header_exposes_per_stage_resource_and_completion_cost_plans(self):
        source = HEADER.read_text(encoding="utf-8")
        for signature in (
            "enum class ResourceStage",
            "struct StageResourceUsage",
            "struct ResourcePlan",
            "struct ResourcePeakEstimate",
            "inline ResourcePeakEstimate estimate_resource_peaks(",
            "inline ResourceGateResult resource_capacity_gate(",
            "inline bool estimate_workspace_growth_peak_bytes(",
            "struct StageCostProfile",
            "struct RemainingWork",
            "struct CompletionCostPrediction",
            "inline CompletionCostPrediction predict_remaining_cost(",
            "partial_elapsed_lower_bound_seconds",
            "device_library_workspace_bytes",
        ):
            self.assertIn(signature, source)

    def test_cuda_workspace_exposes_move_only_resident_buffers_without_factor_cache(self):
        source = (ROOT / "src" / "measure" / "rpmd_ja_cuda_workspace.cuh").read_text(encoding="utf-8")
        for signature in (
            "class DeviceBuffer",
            "void reserve(",
            "std::size_t capacity() const",
            "struct CovarianceFitWorkspace",
            "curvature_ambient",
            "curvature_action",
            "curvature_gram",
            "solver_workspace",
            "DeviceBuffer(const DeviceBuffer&) = delete",
        ):
            self.assertIn(signature, source)
        self.assertNotIn("factor_identity", source)
        self.assertNotIn("cached_factor", source)

    def test_workspace_growth_cpu_fixture_covers_reuse_and_allocate_before_free_peak(self):
        policy = (ROOT / "src" / "measure" / "rpmd_ja_feasibility_policy.cuh").read_text(encoding="utf-8")
        cpp = (ROOT / "tests" / "rpmd_ja_feasibility_policy_test.cpp").read_text(encoding="utf-8")
        self.assertIn("estimate_workspace_growth_peak_bytes", policy)
        self.assertIn("resident_current[]={64,32},resident_same[]={64,16}", cpp)
        self.assertIn("workspace_peak==0", cpp)
        self.assertIn("workspace_peak==80", cpp)
        self.assertIn("solver_current[]={64},solver_growth[]={96}", cpp)

    def test_cpp_asserts_execute_the_header_when_a_compiler_is_available(self):
        if CXX is None:
            self.skipTest("C++ runtime test skipped: c++, g++, clang++, and cl are unavailable")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "rpmd_ja_feasibility_policy_test"
            build = subprocess.run(
                cpp_test_support.compile_command(CXX, 11, CPP_TEST, executable),
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
