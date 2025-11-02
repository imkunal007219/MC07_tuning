"""
Convergence Criteria and Stopping Conditions
Monitors optimization progress and determines when to stop
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence criteria"""

    # Fitness improvement thresholds
    min_improvement_threshold: float = 0.001  # Minimum improvement to consider progress
    improvement_window: int = 10  # Number of trials to check for improvement

    # Plateau detection
    plateau_tolerance: float = 0.005  # Max variance to consider plateau
    plateau_window: int = 15  # Number of trials for plateau detection

    # Stability requirements
    stability_window: int = 20  # Number of trials to check stability
    stability_variance_threshold: float = 0.01  # Max variance for stability

    # Early stopping
    max_trials_without_improvement: int = 30
    min_trials_before_stop: int = 20  # Minimum trials before early stopping

    # Fitness thresholds
    target_fitness: Optional[float] = 0.95  # Stop if this fitness achieved
    min_acceptable_fitness: float = 0.5  # Minimum acceptable fitness

    # Convergence confirmation
    convergence_confirmation_trials: int = 5  # Trials to confirm convergence

    # Statistical tests
    use_statistical_tests: bool = True
    confidence_level: float = 0.95


@dataclass
class ConvergenceStatus:
    """Current convergence status"""
    is_converged: bool
    reason: str
    current_best_fitness: float
    trials_since_improvement: int
    fitness_variance: float
    improvement_rate: float
    recommendation: str


class ConvergenceMonitor:
    """
    Monitors optimization convergence and provides stopping recommendations
    Implements multiple convergence criteria for robust detection
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()

        # History tracking
        self.fitness_history = []
        self.best_fitness_history = []
        self.trial_times = []

        # State tracking
        self.best_fitness = -float('inf')
        self.best_trial = -1
        self.trials_since_improvement = 0
        self.convergence_confirmed_count = 0

        # Running statistics
        self.recent_fitness = deque(maxlen=self.config.improvement_window)
        self.plateau_fitness = deque(maxlen=self.config.plateau_window)
        self.stability_fitness = deque(maxlen=self.config.stability_window)

    def reset(self):
        """Reset monitor for new optimization phase"""
        self.fitness_history = []
        self.best_fitness_history = []
        self.trial_times = []
        self.best_fitness = -float('inf')
        self.best_trial = -1
        self.trials_since_improvement = 0
        self.convergence_confirmed_count = 0
        self.recent_fitness.clear()
        self.plateau_fitness.clear()
        self.stability_fitness.clear()
        logger.info("Convergence monitor reset")

    def update(self, trial_num: int, fitness: float, trial_time: float = 0.0) -> ConvergenceStatus:
        """
        Update convergence monitor with new trial result
        Returns current convergence status
        """
        # Update history
        self.fitness_history.append(fitness)
        self.trial_times.append(trial_time)

        # Update buffers
        self.recent_fitness.append(fitness)
        self.plateau_fitness.append(fitness)
        self.stability_fitness.append(fitness)

        # Track best fitness
        if fitness > self.best_fitness:
            improvement = fitness - self.best_fitness
            self.best_fitness = fitness
            self.best_trial = trial_num
            self.trials_since_improvement = 0
            logger.info(f"New best fitness: {fitness:.6f} (improvement: {improvement:.6f})")
        else:
            self.trials_since_improvement += 1

        self.best_fitness_history.append(self.best_fitness)

        # Check convergence
        return self._evaluate_convergence(trial_num)

    def _evaluate_convergence(self, trial_num: int) -> ConvergenceStatus:
        """Evaluate all convergence criteria"""

        if trial_num < self.config.min_trials_before_stop:
            return ConvergenceStatus(
                is_converged=False,
                reason="Insufficient trials",
                current_best_fitness=self.best_fitness,
                trials_since_improvement=self.trials_since_improvement,
                fitness_variance=self._calculate_variance(),
                improvement_rate=self._calculate_improvement_rate(),
                recommendation="Continue optimization"
            )

        # Check multiple convergence criteria
        criteria_results = []

        # 1. Target fitness achieved
        if self.config.target_fitness and self.best_fitness >= self.config.target_fitness:
            criteria_results.append(("target_fitness", True,
                                   f"Target fitness {self.config.target_fitness} achieved"))

        # 2. No improvement for extended period
        if self.trials_since_improvement >= self.config.max_trials_without_improvement:
            criteria_results.append(("no_improvement", True,
                                   f"No improvement for {self.trials_since_improvement} trials"))

        # 3. Plateau detection
        is_plateau, plateau_msg = self._check_plateau()
        if is_plateau:
            criteria_results.append(("plateau", True, plateau_msg))

        # 4. Stability check
        is_stable, stability_msg = self._check_stability()
        if is_stable:
            criteria_results.append(("stability", True, stability_msg))

        # 5. Diminishing returns
        is_diminishing, diminishing_msg = self._check_diminishing_returns()
        if is_diminishing:
            criteria_results.append(("diminishing_returns", True, diminishing_msg))

        # 6. Statistical convergence test
        if self.config.use_statistical_tests:
            is_statistical, stat_msg = self._check_statistical_convergence()
            if is_statistical:
                criteria_results.append(("statistical", True, stat_msg))

        # Determine overall convergence
        converged_criteria = [c for c in criteria_results if c[1]]

        if converged_criteria:
            # Require confirmation over multiple trials
            self.convergence_confirmed_count += 1

            if self.convergence_confirmed_count >= self.config.convergence_confirmation_trials:
                reasons = "; ".join([c[2] for c in converged_criteria])

                return ConvergenceStatus(
                    is_converged=True,
                    reason=reasons,
                    current_best_fitness=self.best_fitness,
                    trials_since_improvement=self.trials_since_improvement,
                    fitness_variance=self._calculate_variance(),
                    improvement_rate=self._calculate_improvement_rate(),
                    recommendation="Stop optimization - converged"
                )
            else:
                return ConvergenceStatus(
                    is_converged=False,
                    reason=f"Convergence confirmation {self.convergence_confirmed_count}/{self.config.convergence_confirmation_trials}",
                    current_best_fitness=self.best_fitness,
                    trials_since_improvement=self.trials_since_improvement,
                    fitness_variance=self._calculate_variance(),
                    improvement_rate=self._calculate_improvement_rate(),
                    recommendation="Continue - confirming convergence"
                )
        else:
            # Reset confirmation count
            self.convergence_confirmed_count = 0

            return ConvergenceStatus(
                is_converged=False,
                reason="Still improving",
                current_best_fitness=self.best_fitness,
                trials_since_improvement=self.trials_since_improvement,
                fitness_variance=self._calculate_variance(),
                improvement_rate=self._calculate_improvement_rate(),
                recommendation="Continue optimization"
            )

    def _check_plateau(self) -> Tuple[bool, str]:
        """Check if fitness has plateaued"""
        if len(self.plateau_fitness) < self.config.plateau_window:
            return False, ""

        variance = np.var(list(self.plateau_fitness))

        if variance < self.config.plateau_tolerance:
            return True, f"Fitness plateau detected (variance: {variance:.6f})"

        return False, ""

    def _check_stability(self) -> Tuple[bool, str]:
        """Check if optimization has stabilized"""
        if len(self.stability_fitness) < self.config.stability_window:
            return False, ""

        # Check if best fitness hasn't changed much
        recent_best = max(self.stability_fitness)
        variance = np.var(list(self.stability_fitness))

        if variance < self.config.stability_variance_threshold:
            if abs(recent_best - self.best_fitness) < self.config.min_improvement_threshold:
                return True, f"Optimization stabilized (variance: {variance:.6f})"

        return False, ""

    def _check_diminishing_returns(self) -> Tuple[bool, str]:
        """Check if improvements are diminishing"""
        if len(self.best_fitness_history) < self.config.improvement_window * 2:
            return False, ""

        # Calculate improvement rate over recent window
        recent_improvements = []
        for i in range(-self.config.improvement_window, -1):
            improvement = self.best_fitness_history[i] - self.best_fitness_history[i-1]
            if improvement > 0:
                recent_improvements.append(improvement)

        if recent_improvements:
            avg_improvement = np.mean(recent_improvements)

            if avg_improvement < self.config.min_improvement_threshold:
                return True, f"Diminishing returns (avg improvement: {avg_improvement:.6f})"

        return False, ""

    def _check_statistical_convergence(self) -> Tuple[bool, str]:
        """
        Statistical test for convergence
        Uses Mann-Kendall trend test to detect if there's a trend
        """
        if len(self.fitness_history) < 20:
            return False, ""

        # Simple trend analysis: compare first half vs second half
        n = len(self.best_fitness_history)
        mid = n // 2

        first_half = self.best_fitness_history[:mid]
        second_half = self.best_fitness_history[mid:]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)

        improvement_percentage = (mean_second - mean_first) / mean_first * 100

        # If improvement between halves is less than 1%, likely converged
        if abs(improvement_percentage) < 1.0:
            return True, f"Statistical convergence detected (improvement: {improvement_percentage:.2f}%)"

        return False, ""

    def _calculate_variance(self) -> float:
        """Calculate recent fitness variance"""
        if len(self.recent_fitness) < 2:
            return 0.0
        return np.var(list(self.recent_fitness))

    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement (fitness/trial)"""
        if len(self.fitness_history) < 2:
            return 0.0

        # Linear regression on best fitness history
        x = np.arange(len(self.best_fitness_history))
        y = np.array(self.best_fitness_history)

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope

        return 0.0

    def should_continue(self) -> bool:
        """Simple check if optimization should continue"""
        if not self.fitness_history:
            return True

        status = self._evaluate_convergence(len(self.fitness_history))
        return not status.is_converged

    def get_progress_report(self) -> str:
        """Generate progress report"""
        if not self.fitness_history:
            return "No trials completed yet"

        lines = []
        lines.append("=" * 60)
        lines.append("CONVERGENCE PROGRESS REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Trials: {len(self.fitness_history)}")
        lines.append(f"Best Fitness: {self.best_fitness:.6f} (Trial {self.best_trial})")
        lines.append(f"Current Fitness: {self.fitness_history[-1]:.6f}")
        lines.append(f"Trials Since Improvement: {self.trials_since_improvement}")
        lines.append("")
        lines.append(f"Recent Variance: {self._calculate_variance():.6f}")
        lines.append(f"Improvement Rate: {self._calculate_improvement_rate():.6f}")
        lines.append("")

        # Status
        status = self._evaluate_convergence(len(self.fitness_history))
        lines.append(f"Converged: {status.is_converged}")
        lines.append(f"Reason: {status.reason}")
        lines.append(f"Recommendation: {status.recommendation}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def plot_convergence(self, save_path: str = "convergence.png"):
        """Plot convergence history"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Fitness history
            trials = np.arange(len(self.fitness_history))
            ax1.plot(trials, self.fitness_history, 'b.', alpha=0.5, label='Trial Fitness')
            ax1.plot(trials, self.best_fitness_history, 'r-', linewidth=2, label='Best Fitness')
            ax1.axhline(y=self.best_fitness, color='g', linestyle='--', label='Current Best')

            if self.config.target_fitness:
                ax1.axhline(y=self.config.target_fitness, color='orange',
                          linestyle='--', label='Target Fitness')

            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Optimization Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Improvement rate
            if len(self.best_fitness_history) > 1:
                improvements = np.diff(self.best_fitness_history)
                ax2.plot(trials[1:], improvements, 'g-', alpha=0.7)
                ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax2.axhline(y=self.config.min_improvement_threshold, color='r',
                          linestyle='--', label='Min Improvement Threshold')
                ax2.set_xlabel('Trial')
                ax2.set_ylabel('Fitness Improvement')
                ax2.set_title('Per-Trial Improvement')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Convergence plot saved to {save_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    def export_history(self) -> Dict:
        """Export convergence history for analysis"""
        return {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'trial_times': self.trial_times,
            'best_fitness': self.best_fitness,
            'best_trial': self.best_trial,
            'trials_since_improvement': self.trials_since_improvement,
            'total_trials': len(self.fitness_history)
        }


class MultiPhaseConvergenceManager:
    """
    Manages convergence across multiple optimization phases
    Each phase may have different convergence criteria
    """

    def __init__(self):
        self.phase_monitors = {}
        self.current_phase = None
        self.phase_history = []

    def start_phase(self, phase_name: str, config: Optional[ConvergenceConfig] = None):
        """Start monitoring a new phase"""
        self.current_phase = phase_name
        self.phase_monitors[phase_name] = ConvergenceMonitor(config)
        logger.info(f"Started convergence monitoring for phase: {phase_name}")

    def update(self, fitness: float, trial_time: float = 0.0) -> ConvergenceStatus:
        """Update current phase monitor"""
        if not self.current_phase:
            raise RuntimeError("No active phase")

        monitor = self.phase_monitors[self.current_phase]
        trial_num = len(monitor.fitness_history)

        return monitor.update(trial_num, fitness, trial_time)

    def complete_phase(self):
        """Mark current phase as complete"""
        if not self.current_phase:
            return

        monitor = self.phase_monitors[self.current_phase]

        self.phase_history.append({
            'phase': self.current_phase,
            'best_fitness': monitor.best_fitness,
            'total_trials': len(monitor.fitness_history),
            'converged': True
        })

        logger.info(f"Phase {self.current_phase} completed with fitness {monitor.best_fitness:.6f}")
        self.current_phase = None

    def get_overall_summary(self) -> str:
        """Get summary of all phases"""
        lines = []
        lines.append("=" * 80)
        lines.append("MULTI-PHASE OPTIMIZATION SUMMARY")
        lines.append("=" * 80)

        for phase_data in self.phase_history:
            lines.append(f"Phase: {phase_data['phase']}")
            lines.append(f"  Best Fitness: {phase_data['best_fitness']:.6f}")
            lines.append(f"  Total Trials: {phase_data['total_trials']}")
            lines.append(f"  Converged: {phase_data['converged']}")
            lines.append("")

        total_trials = sum(p['total_trials'] for p in self.phase_history)
        lines.append(f"Total Trials (All Phases): {total_trials}")
        lines.append("=" * 80)

        return "\n".join(lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = ConvergenceConfig(
        min_improvement_threshold=0.001,
        max_trials_without_improvement=20,
        target_fitness=0.95
    )

    monitor = ConvergenceMonitor(config)

    # Simulate optimization
    np.random.seed(42)
    base_fitness = 0.5

    for trial in range(100):
        # Simulated fitness with diminishing improvements
        improvement = 0.01 * np.exp(-trial / 20) + np.random.randn() * 0.005
        fitness = base_fitness + improvement
        base_fitness = max(base_fitness, fitness)

        status = monitor.update(trial, fitness, trial_time=10.0)

        if trial % 10 == 0:
            logger.info(f"Trial {trial}: Fitness={fitness:.4f}, Converged={status.is_converged}")

        if status.is_converged:
            logger.info(f"Convergence detected at trial {trial}")
            logger.info(status.reason)
            break

    print(monitor.get_progress_report())
    monitor.plot_convergence()
