"""
Real-time Optimization Visualization Dashboard

Displays live plots during optimization for monitoring progress.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import logging
from typing import List, Dict, Optional
from collections import deque
from threading import Lock
import time

logger = logging.getLogger(__name__)


class LiveDashboard:
    """
    Real-time visualization dashboard for optimization progress

    Displays:
    - Fitness evolution over generations
    - Best parameters evolution
    - Population diversity
    - Time per trial
    - Safety violations
    - Parameter distributions
    """

    def __init__(self, parameter_names: List[str],
                 update_interval: int = 2000,
                 max_history: int = 1000,
                 save_path: str = "/tmp/optimization_dashboard.png"):
        """
        Initialize dashboard

        Args:
            parameter_names: Names of parameters being optimized
            update_interval: Update interval in milliseconds
            max_history: Maximum history points to display
            save_path: Path to save dashboard image
        """
        self.parameter_names = parameter_names
        self.update_interval = update_interval
        self.max_history = max_history
        self.save_path = save_path

        # Data storage (thread-safe)
        self.lock = Lock()
        self.generations = deque(maxlen=max_history)
        self.best_fitness = deque(maxlen=max_history)
        self.avg_fitness = deque(maxlen=max_history)
        self.worst_fitness = deque(maxlen=max_history)
        self.diversity = deque(maxlen=max_history)
        self.trial_times = deque(maxlen=max_history)
        self.safety_violations = deque(maxlen=max_history)
        self.best_params_history: Dict[str, deque] = {
            name: deque(maxlen=max_history) for name in parameter_names
        }

        # Dashboard state
        self.current_generation = 0
        self.total_time = 0.0
        self.start_time = time.time()

        # Figure setup
        self.fig = None
        self.axes = None
        self.setup_figure()

    def setup_figure(self):
        """Setup matplotlib figure with subplots"""
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # Create subplots
        self.axes = {
            'fitness': self.fig.add_subplot(gs[0, :2]),      # Fitness evolution (top left, wide)
            'time': self.fig.add_subplot(gs[0, 2]),           # Time per trial (top right)
            'diversity': self.fig.add_subplot(gs[1, 0]),      # Population diversity
            'violations': self.fig.add_subplot(gs[1, 1]),     # Safety violations
            'params_1': self.fig.add_subplot(gs[1, 2]),       # Top parameters (1-3)
            'params_2': self.fig.add_subplot(gs[2, :]),       # All parameters (bottom, wide)
        }

        # Set titles
        self.axes['fitness'].set_title('Fitness Evolution', fontsize=12, fontweight='bold')
        self.axes['time'].set_title('Trial Duration', fontsize=10)
        self.axes['diversity'].set_title('Population Diversity', fontsize=10)
        self.axes['violations'].set_title('Safety Violations', fontsize=10)
        self.axes['params_1'].set_title('Top 3 Parameters', fontsize=10)
        self.axes['params_2'].set_title('All Parameters Evolution', fontsize=10)

        plt.tight_layout()

    def update_data(self, generation: int, best_fit: float, avg_fit: float,
                   worst_fit: float, div: float, trial_time: float,
                   violations: int, best_params: Dict[str, float]):
        """
        Update dashboard data (thread-safe)

        Args:
            generation: Current generation number
            best_fit: Best fitness in generation
            avg_fit: Average fitness
            worst_fit: Worst fitness
            div: Population diversity metric
            trial_time: Time taken for generation
            violations: Number of safety violations
            best_params: Best parameter values
        """
        with self.lock:
            self.generations.append(generation)
            self.best_fitness.append(best_fit)
            self.avg_fitness.append(avg_fit)
            self.worst_fitness.append(worst_fit)
            self.diversity.append(div)
            self.trial_times.append(trial_time)
            self.safety_violations.append(violations)

            for name in self.parameter_names:
                if name in best_params:
                    self.best_params_history[name].append(best_params[name])

            self.current_generation = generation
            self.total_time = time.time() - self.start_time

    def render(self):
        """Render current state of dashboard"""
        with self.lock:
            if len(self.generations) == 0:
                return

            # Clear all axes
            for ax in self.axes.values():
                ax.clear()

            # Plot fitness evolution
            ax = self.axes['fitness']
            generations_list = list(self.generations)
            ax.plot(generations_list, list(self.best_fitness), 'g-', linewidth=2, label='Best')
            ax.plot(generations_list, list(self.avg_fitness), 'b-', linewidth=1, label='Average')
            ax.plot(generations_list, list(self.worst_fitness), 'r--', linewidth=1, label='Worst')
            ax.fill_between(generations_list, list(self.worst_fitness), list(self.best_fitness),
                           alpha=0.2, color='blue')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_title('Fitness Evolution', fontsize=12, fontweight='bold')

            # Plot trial times
            ax = self.axes['time']
            ax.plot(generations_list, list(self.trial_times), 'purple', linewidth=1)
            ax.fill_between(generations_list, list(self.trial_times), alpha=0.3, color='purple')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Time (s)')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Trial Duration (Avg: {np.mean(self.trial_times):.1f}s)', fontsize=10)

            # Plot diversity
            ax = self.axes['diversity']
            ax.plot(generations_list, list(self.diversity), 'orange', linewidth=1.5)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Diversity')
            ax.grid(True, alpha=0.3)
            ax.set_title('Population Diversity', fontsize=10)

            # Plot safety violations
            ax = self.axes['violations']
            ax.bar(generations_list, list(self.safety_violations), color='red', alpha=0.6)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Violations')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(f'Safety Violations (Total: {sum(self.safety_violations)})', fontsize=10)

            # Plot top 3 parameters
            ax = self.axes['params_1']
            for i, name in enumerate(self.parameter_names[:3]):
                if name in self.best_params_history and len(self.best_params_history[name]) > 0:
                    ax.plot(generations_list, list(self.best_params_history[name]),
                           linewidth=1.5, label=name, marker='o', markersize=2)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Parameter Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title('Top 3 Parameters', fontsize=10)

            # Plot all parameters (heatmap style)
            ax = self.axes['params_2']
            if len(self.parameter_names) > 0 and len(generations_list) > 0:
                # Create matrix of parameter values
                param_matrix = []
                for name in self.parameter_names[:10]:  # Show max 10 parameters
                    if name in self.best_params_history:
                        param_matrix.append(list(self.best_params_history[name]))

                if param_matrix:
                    # Normalize each row for better visualization
                    param_matrix_np = np.array(param_matrix)
                    im = ax.imshow(param_matrix_np, aspect='auto', cmap='viridis', interpolation='nearest')
                    ax.set_yticks(range(len(param_matrix)))
                    ax.set_yticklabels(self.parameter_names[:len(param_matrix)], fontsize=8)
                    ax.set_xlabel('Generation')
                    ax.set_title('All Parameters Evolution (Heatmap)', fontsize=10)
                    plt.colorbar(im, ax=ax, label='Value')

            # Add overall statistics as text
            stats_text = (
                f"Generation: {self.current_generation} | "
                f"Best Fitness: {self.best_fitness[-1]:.2f} | "
                f"Elapsed: {self.total_time/60:.1f} min | "
                f"Est. Remaining: {self._estimate_remaining():.1f} min"
            )
            self.fig.suptitle(stats_text, fontsize=14, fontweight='bold')

            # Save figure
            plt.savefig(self.save_path, dpi=120, bbox_inches='tight')
            logger.info(f"Dashboard saved to {self.save_path}")

    def _estimate_remaining(self) -> float:
        """Estimate remaining time based on current progress"""
        if len(self.trial_times) < 5:
            return 0.0

        avg_time_per_gen = np.mean(list(self.trial_times)[-10:])  # Last 10 generations
        # Assume total 100 generations (configurable)
        remaining_gens = max(0, 100 - self.current_generation)
        remaining_time = (remaining_gens * avg_time_per_gen) / 60.0  # Convert to minutes

        return remaining_time

    def save_final_report(self, output_path: str = "/tmp/optimization_final_report.png"):
        """
        Save final comprehensive report

        Args:
            output_path: Path to save final report
        """
        self.render()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Final report saved to {output_path}")

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics

        Returns:
            Dictionary with summary statistics
        """
        with self.lock:
            if len(self.best_fitness) == 0:
                return {}

            return {
                'total_generations': self.current_generation,
                'best_fitness_final': self.best_fitness[-1] if self.best_fitness else 0,
                'best_fitness_ever': max(self.best_fitness) if self.best_fitness else 0,
                'avg_fitness_final': self.avg_fitness[-1] if self.avg_fitness else 0,
                'total_time_minutes': self.total_time / 60.0,
                'avg_time_per_generation': np.mean(self.trial_times) if self.trial_times else 0,
                'total_safety_violations': sum(self.safety_violations) if self.safety_violations else 0,
                'convergence_rate': self._calculate_convergence_rate()
            }

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate from fitness history"""
        if len(self.best_fitness) < 10:
            return 0.0

        # Compare first 10 and last 10 generations
        early_fitness = np.mean(list(self.best_fitness)[:10])
        recent_fitness = np.mean(list(self.best_fitness)[-10:])

        improvement = recent_fitness - early_fitness
        return improvement


# Convenience function for integration with optimizers
def create_dashboard_callback(dashboard: LiveDashboard):
    """
    Create callback function for optimizer integration

    Args:
        dashboard: LiveDashboard instance

    Returns:
        Callback function
    """
    def callback(generation: int, population: List, statistics: Dict,
                best_individual: Dict):
        """
        Callback function called after each generation

        Args:
            generation: Current generation number
            population: Current population
            statistics: Generation statistics
            best_individual: Best individual in generation
        """
        # Extract data from statistics
        best_fit = statistics.get('max', 0)
        avg_fit = statistics.get('avg', 0)
        worst_fit = statistics.get('min', 0)

        # Calculate diversity
        diversity = statistics.get('std', 0)

        # Get best parameters
        best_params = best_individual if isinstance(best_individual, dict) else {}

        # Update dashboard
        dashboard.update_data(
            generation=generation,
            best_fit=best_fit,
            avg_fit=avg_fit,
            worst_fit=worst_fit,
            div=diversity,
            trial_time=statistics.get('time', 0),
            violations=statistics.get('violations', 0),
            best_params=best_params
        )

        # Render every 5 generations to avoid overhead
        if generation % 5 == 0:
            dashboard.render()

    return callback
