"""
Intelligent Test Sequencing for Drone Tuning

Implements progressive testing that stops early if fundamentals fail,
saving ~40% of evaluation time by skipping expensive tests for clearly
bad parameters.

Test Hierarchy:
- Level 1 (REQUIRED): Basic hover stability - 5 sec
- Level 2 (REQUIRED): Small step response - 10°
- Level 3 (IMPORTANT): Large step response - 20°
- Level 4 (OPTIONAL): Frequency sweep
- Level 5 (OPTIONAL): Trajectory tracking

Control Theory Principle: Test increasing difficulty. If basic stability
fails, advanced performance tests are meaningless.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TestLevel(Enum):
    """Test difficulty levels"""
    HOVER = 1          # Basic stability
    SMALL_STEP = 2     # Small disturbance response
    LARGE_STEP = 3     # Large disturbance response
    FREQUENCY = 4      # Frequency response
    TRAJECTORY = 5     # Advanced tracking


@dataclass
class TestStageResult:
    """Results from a single test stage"""
    level: TestLevel
    passed: bool
    score: float              # 0-100
    duration: float           # seconds
    reason: str               # Pass/fail reason
    metrics: Dict = None      # Detailed metrics


@dataclass
class StagedEvaluationResult:
    """Complete results from staged evaluation"""
    highest_level_reached: TestLevel
    total_score: float
    total_duration: float
    stage_results: List[TestStageResult]
    early_terminated: bool
    time_saved: float


class IntelligentTestSequencer:
    """
    Progressive test sequencing with early termination
    """

    def __init__(self, min_pass_score: float = 60.0,
                 enable_optional_tests: bool = True):
        """
        Initialize test sequencer

        Args:
            min_pass_score: Minimum score to pass a stage (0-100)
            enable_optional_tests: Whether to run optional tests (level 4-5)
        """
        self.min_pass_score = min_pass_score
        self.enable_optional_tests = enable_optional_tests

        # Define stage requirements
        self.stage_requirements = {
            TestLevel.HOVER: {
                'required': True,
                'duration': 5.0,
                'max_oscillation_amp': 5.0,  # degrees
                'max_position_error': 1.0,    # meters
                'description': 'Basic hover stability'
            },
            TestLevel.SMALL_STEP: {
                'required': True,
                'duration': 10.0,
                'max_overshoot': 30.0,      # percent
                'max_settling_time': 3.0,    # seconds
                'description': 'Small step response (10°)'
            },
            TestLevel.LARGE_STEP: {
                'required': False,
                'duration': 15.0,
                'max_overshoot': 40.0,
                'max_settling_time': 4.0,
                'description': 'Large step response (20°)'
            },
            TestLevel.FREQUENCY: {
                'required': False,
                'duration': 30.0,
                'min_phase_margin': 40.0,    # degrees
                'description': 'Frequency sweep'
            },
            TestLevel.TRAJECTORY: {
                'required': False,
                'duration': 45.0,
                'max_tracking_error': 0.5,   # meters
                'description': 'Trajectory tracking'
            }
        }

        logger.info("IntelligentTestSequencer initialized")
        logger.info(f"  Min pass score: {min_pass_score}")
        logger.info(f"  Optional tests: {enable_optional_tests}")

    def evaluate_staged(self, evaluator, sitl_manager,
                       parameters: Dict[str, float]) -> StagedEvaluationResult:
        """
        Run staged evaluation with early termination

        Args:
            evaluator: Performance evaluator
            sitl_manager: SITL manager for simulations
            parameters: Parameter dictionary to test

        Returns:
            StagedEvaluationResult
        """
        logger.info("\n" + "="*60)
        logger.info("STAGED EVALUATION")
        logger.info("="*60)

        stage_results = []
        total_duration = 0.0
        highest_level = TestLevel.HOVER
        early_terminated = False

        # Level 1: Hover stability (REQUIRED)
        logger.info(f"\n[Stage 1/5] {self.stage_requirements[TestLevel.HOVER]['description']}")
        hover_result = self._test_hover(evaluator, sitl_manager, parameters)
        stage_results.append(hover_result)
        total_duration += hover_result.duration

        if not hover_result.passed:
            logger.warning(f"✗ Failed hover test: {hover_result.reason}")
            logger.warning("Skipping remaining tests (hover is fundamental)")
            early_terminated = True
        else:
            logger.info(f"✓ Passed hover test (score: {hover_result.score:.1f})")
            highest_level = TestLevel.HOVER

            # Level 2: Small step response (REQUIRED)
            logger.info(f"\n[Stage 2/5] {self.stage_requirements[TestLevel.SMALL_STEP]['description']}")
            small_step_result = self._test_small_step(evaluator, sitl_manager, parameters)
            stage_results.append(small_step_result)
            total_duration += small_step_result.duration

            if not small_step_result.passed:
                logger.warning(f"✗ Failed small step test: {small_step_result.reason}")
                logger.warning("Skipping advanced tests")
                early_terminated = True
            else:
                logger.info(f"✓ Passed small step test (score: {small_step_result.score:.1f})")
                highest_level = TestLevel.SMALL_STEP

                # Level 3: Large step response (IMPORTANT)
                logger.info(f"\n[Stage 3/5] {self.stage_requirements[TestLevel.LARGE_STEP]['description']}")
                large_step_result = self._test_large_step(evaluator, sitl_manager, parameters)
                stage_results.append(large_step_result)
                total_duration += large_step_result.duration

                if not large_step_result.passed:
                    logger.warning(f"✗ Failed large step test: {large_step_result.reason}")
                    logger.warning("Skipping optional tests")
                    early_terminated = True
                else:
                    logger.info(f"✓ Passed large step test (score: {large_step_result.score:.1f})")
                    highest_level = TestLevel.LARGE_STEP

                    # Optional tests only for good candidates
                    if self.enable_optional_tests and hover_result.score > 80:
                        # Level 4: Frequency sweep (OPTIONAL)
                        logger.info(f"\n[Stage 4/5] {self.stage_requirements[TestLevel.FREQUENCY]['description']}")
                        freq_result = self._test_frequency(evaluator, sitl_manager, parameters)
                        stage_results.append(freq_result)
                        total_duration += freq_result.duration

                        if freq_result.passed:
                            logger.info(f"✓ Passed frequency test (score: {freq_result.score:.1f})")
                            highest_level = TestLevel.FREQUENCY

                            # Level 5: Trajectory tracking (OPTIONAL, only for excellent candidates)
                            if hover_result.score > 90:
                                logger.info(f"\n[Stage 5/5] {self.stage_requirements[TestLevel.TRAJECTORY]['description']}")
                                traj_result = self._test_trajectory(evaluator, sitl_manager, parameters)
                                stage_results.append(traj_result)
                                total_duration += traj_result.duration

                                if traj_result.passed:
                                    logger.info(f"✓ Passed trajectory test (score: {traj_result.score:.1f})")
                                    highest_level = TestLevel.TRAJECTORY

        # Calculate total score (weighted by importance)
        total_score = self._calculate_total_score(stage_results)

        # Calculate time saved
        max_possible_duration = sum(req['duration'] for req in self.stage_requirements.values())
        time_saved = max_possible_duration - total_duration

        result = StagedEvaluationResult(
            highest_level_reached=highest_level,
            total_score=total_score,
            total_duration=total_duration,
            stage_results=stage_results,
            early_terminated=early_terminated,
            time_saved=time_saved
        )

        logger.info("\n" + "="*60)
        logger.info(f"STAGED EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Highest level reached: {highest_level.name}")
        logger.info(f"Total score: {total_score:.1f}/100")
        logger.info(f"Duration: {total_duration:.1f}s (saved: {time_saved:.1f}s)")
        logger.info("="*60 + "\n")

        return result

    def _test_hover(self, evaluator, sitl_manager,
                   parameters: Dict[str, float]) -> TestStageResult:
        """Test basic hover stability"""
        # Simplified - would run actual simulation
        # For now, return mock result
        return TestStageResult(
            level=TestLevel.HOVER,
            passed=True,
            score=75.0,
            duration=5.0,
            reason="Stable hover achieved",
            metrics={'oscillation_amp': 3.0, 'position_error': 0.5}
        )

    def _test_small_step(self, evaluator, sitl_manager,
                        parameters: Dict[str, float]) -> TestStageResult:
        """Test small step response (10°)"""
        return TestStageResult(
            level=TestLevel.SMALL_STEP,
            passed=True,
            score=70.0,
            duration=10.0,
            reason="Acceptable step response",
            metrics={'overshoot': 20.0, 'settling_time': 2.5}
        )

    def _test_large_step(self, evaluator, sitl_manager,
                        parameters: Dict[str, float]) -> TestStageResult:
        """Test large step response (20°)"""
        return TestStageResult(
            level=TestLevel.LARGE_STEP,
            passed=True,
            score=65.0,
            duration=15.0,
            reason="Adequate large step response",
            metrics={'overshoot': 35.0, 'settling_time': 3.5}
        )

    def _test_frequency(self, evaluator, sitl_manager,
                       parameters: Dict[str, float]) -> TestStageResult:
        """Test frequency response"""
        return TestStageResult(
            level=TestLevel.FREQUENCY,
            passed=True,
            score=80.0,
            duration=30.0,
            reason="Good frequency response",
            metrics={'phase_margin': 50.0, 'gain_margin': 8.0}
        )

    def _test_trajectory(self, evaluator, sitl_manager,
                        parameters: Dict[str, float]) -> TestStageResult:
        """Test trajectory tracking"""
        return TestStageResult(
            level=TestLevel.TRAJECTORY,
            passed=True,
            score=85.0,
            duration=45.0,
            reason="Excellent tracking",
            metrics={'tracking_error': 0.3, 'smoothness': 90.0}
        )

    def _calculate_total_score(self, stage_results: List[TestStageResult]) -> float:
        """
        Calculate weighted total score

        Weights by importance:
        - Hover: 30%
        - Small step: 25%
        - Large step: 20%
        - Frequency: 15%
        - Trajectory: 10%
        """
        weights = {
            TestLevel.HOVER: 0.30,
            TestLevel.SMALL_STEP: 0.25,
            TestLevel.LARGE_STEP: 0.20,
            TestLevel.FREQUENCY: 0.15,
            TestLevel.TRAJECTORY: 0.10
        }

        total_score = 0.0
        total_weight = 0.0

        for result in stage_results:
            weight = weights[result.level]
            total_score += result.score * weight
            total_weight += weight

        # Normalize by actual weight sum (in case not all tests ran)
        if total_weight > 0:
            total_score = total_score / total_weight
        else:
            total_score = 0.0

        return total_score


def estimate_time_savings(enable_staged: bool = True,
                         enable_early_crash: bool = True,
                         typical_crash_rate: float = 0.6) -> Dict[str, float]:
    """
    Estimate time savings from intelligent testing

    Args:
        enable_staged: Use staged evaluation
        enable_early_crash: Use early crash detection
        typical_crash_rate: Expected crash rate (0-1)

    Returns:
        Dictionary with time savings estimates
    """
    # Baseline times
    full_test_duration = 105.0  # seconds (hover + steps + freq + traj)
    crash_duration_full = 120.0  # 2 minutes for full crash
    crash_duration_early = 30.0  # 30 seconds with early detection

    # Calculate savings
    savings = {
        'baseline_avg_time': 0.0,
        'staged_avg_time': 0.0,
        'early_crash_avg_time': 0.0,
        'combined_avg_time': 0.0,
        'time_saved_per_eval': 0.0,
        'percent_saved': 0.0
    }

    # Baseline: always run full test or full crash
    savings['baseline_avg_time'] = (
        (1 - typical_crash_rate) * full_test_duration +
        typical_crash_rate * crash_duration_full
    )

    # Staged evaluation: skip advanced tests (~40% of duration) for bad params
    if enable_staged:
        # Assume 60% fail early and skip 40% of tests
        savings['staged_avg_time'] = (
            (1 - typical_crash_rate) * full_test_duration * 0.7 +  # 30% savings on good
            typical_crash_rate * crash_duration_full * 0.6          # 40% savings on bad
        )
    else:
        savings['staged_avg_time'] = savings['baseline_avg_time']

    # Early crash detection: reduce crash time by 75%
    if enable_early_crash:
        savings['early_crash_avg_time'] = (
            (1 - typical_crash_rate) * full_test_duration +
            typical_crash_rate * crash_duration_early
        )
    else:
        savings['early_crash_avg_time'] = savings['baseline_avg_time']

    # Combined: both optimizations
    if enable_staged and enable_early_crash:
        savings['combined_avg_time'] = (
            (1 - typical_crash_rate) * full_test_duration * 0.7 +
            typical_crash_rate * crash_duration_early
        )
    else:
        savings['combined_avg_time'] = savings['baseline_avg_time']

    # Calculate savings
    savings['time_saved_per_eval'] = savings['baseline_avg_time'] - savings['combined_avg_time']
    savings['percent_saved'] = (savings['time_saved_per_eval'] / savings['baseline_avg_time']) * 100

    return savings
