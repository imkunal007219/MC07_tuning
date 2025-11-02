"""
Visualization Tools for Optimization Progress
Comprehensive plotting and analysis visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OptimizationVisualizer:
    """
    Comprehensive visualization for optimization progress
    Creates plots, dashboards, and analysis visualizations
    """

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")

    def plot_convergence(self, fitness_history: List[float],
                        best_fitness_history: List[float],
                        title: str = "Optimization Convergence",
                        save_name: str = "convergence.png"):
        """Plot convergence history"""
        fig, ax = plt.subplots(figsize=(12, 6))

        trials = np.arange(len(fitness_history))

        # Plot all trials
        ax.plot(trials, fitness_history, 'o', alpha=0.3, markersize=4,
               label='Trial Fitness', color='#3498db')

        # Plot best fitness
        ax.plot(trials, best_fitness_history, '-', linewidth=2.5,
               label='Best Fitness', color='#e74c3c')

        # Highlight best
        best_idx = np.argmax(best_fitness_history)
        ax.plot(best_idx, best_fitness_history[best_idx], '*',
               markersize=20, color='#f39c12', label='Best Trial',
               markeredgecolor='black', markeredgewidth=1.5)

        ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Convergence plot saved: {save_path}")

    def plot_parameter_evolution(self, parameter_history: Dict[str, List[float]],
                                 save_name: str = "parameter_evolution.png"):
        """Plot how parameters evolved during optimization"""
        n_params = len(parameter_history)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, (param_name, values) in enumerate(parameter_history.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            trials = np.arange(len(values))
            ax.plot(trials, values, '-o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Trial', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_title(param_name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Parameter evolution plot saved: {save_path}")

    def plot_phase_comparison(self, phase_results: Dict[str, Dict],
                              save_name: str = "phase_comparison.png"):
        """Compare results across optimization phases"""
        phases = list(phase_results.keys())
        best_fitnesses = [phase_results[p]['best_fitness'] for p in phases]
        n_trials = [phase_results[p]['n_trials'] for p in phases]
        durations = [phase_results[p]['duration'] for p in phases]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Best fitness by phase
        colors = plt.cm.viridis(np.linspace(0, 1, len(phases)))
        axes[0].bar(phases, best_fitnesses, color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        axes[0].set_title('Best Fitness by Phase', fontsize=13, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Trials per phase
        axes[1].bar(phases, n_trials, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Number of Trials', fontsize=12, fontweight='bold')
        axes[1].set_title('Trials per Phase', fontsize=13, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Duration per phase
        axes[2].bar(phases, np.array(durations)/60, color=colors, edgecolor='black', linewidth=1.5)
        axes[2].set_ylabel('Duration (minutes)', fontsize=12, fontweight='bold')
        axes[2].set_title('Time per Phase', fontsize=13, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Phase comparison plot saved: {save_path}")

    def plot_flight_data(self, flight_data: Dict[str, np.ndarray],
                        save_name: str = "flight_analysis.png"):
        """Plot flight telemetry data"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        time = flight_data['timestamp']

        # Altitude
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, flight_data['altitude'], linewidth=2, color='#2ecc71')
        ax1.set_ylabel('Altitude (m)', fontsize=11, fontweight='bold')
        ax1.set_title('Altitude Profile', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Attitude (Roll, Pitch, Yaw)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time, flight_data['roll'], label='Roll', linewidth=1.5)
        ax2.plot(time, flight_data['pitch'], label='Pitch', linewidth=1.5)
        ax2.plot(time, flight_data['yaw'], label='Yaw', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Angle (deg)', fontsize=11, fontweight='bold')
        ax2.set_title('Attitude', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Rates
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time, flight_data['roll_rate'], label='Roll Rate', linewidth=1.5)
        ax3.plot(time, flight_data['pitch_rate'], label='Pitch Rate', linewidth=1.5)
        ax3.plot(time, flight_data['yaw_rate'], label='Yaw Rate', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Rate (deg/s)', fontsize=11, fontweight='bold')
        ax3.set_title('Angular Rates', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Velocity
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(time, flight_data['vx'], label='Vx', linewidth=1.5)
        ax4.plot(time, flight_data['vy'], label='Vy', linewidth=1.5)
        ax4.plot(time, flight_data['vz'], label='Vz', linewidth=1.5)
        ax4.set_ylabel('Velocity (m/s)', fontsize=11, fontweight='bold')
        ax4.set_title('Velocity', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # Motors
        ax5 = fig.add_subplot(gs[2, 1])
        if 'motors' in flight_data and len(flight_data['motors'].shape) > 1:
            for i in range(4):
                ax5.plot(time, flight_data['motors'][:, i], label=f'Motor {i+1}', linewidth=1.5)
        ax5.set_ylabel('Motor Output', fontsize=11, fontweight='bold')
        ax5.set_title('Motor Outputs', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # Battery
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.plot(time, flight_data['battery_voltage'], linewidth=2, color='#e74c3c')
        ax6.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax6.set_title('Battery Voltage', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 3D Trajectory (if position data available)
        if 'latitude' in flight_data and 'longitude' in flight_data:
            ax7 = fig.add_subplot(gs[3, 1], projection='3d')
            # Convert lat/lon to local coordinates (simplified)
            x = (flight_data['longitude'] - flight_data['longitude'][0]) * 111000
            y = (flight_data['latitude'] - flight_data['latitude'][0]) * 111000
            z = flight_data['altitude']
            ax7.plot(x, y, z, linewidth=2, color='#9b59b6')
            ax7.set_xlabel('X (m)', fontsize=9, fontweight='bold')
            ax7.set_ylabel('Y (m)', fontsize=9, fontweight='bold')
            ax7.set_zlabel('Z (m)', fontsize=9, fontweight='bold')
            ax7.set_title('3D Trajectory', fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Flight data plot saved: {save_path}")

    def plot_step_response(self, time: np.ndarray, response: np.ndarray,
                          target: float, title: str = "Step Response",
                          save_name: str = "step_response.png"):
        """Plot step response analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Response plot
        ax1.plot(time, response, linewidth=2, label='Response', color='#3498db')
        ax1.axhline(y=target, color='#e74c3c', linestyle='--', linewidth=2, label='Target')
        ax1.axhline(y=target*1.02, color='gray', linestyle=':', alpha=0.5, label='±2% band')
        ax1.axhline(y=target*0.98, color='gray', linestyle=':', alpha=0.5)
        ax1.fill_between(time, target*0.98, target*1.02, alpha=0.1, color='green')

        ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Error plot
        error = response - target
        ax2.plot(time, error, linewidth=2, color='#e74c3c')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.fill_between(time, 0, error, alpha=0.3, color='#e74c3c')

        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
        ax2.set_title('Tracking Error', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Step response plot saved: {save_path}")

    def plot_parameter_correlation(self, parameters: Dict[str, List[float]],
                                   fitness: List[float],
                                   save_name: str = "parameter_correlation.png"):
        """Plot correlation between parameters and fitness"""
        # Create correlation matrix
        param_names = list(parameters.keys())
        n_params = len(param_names)

        if n_params == 0:
            logger.warning("No parameters to plot")
            return

        # Build data matrix
        data = np.column_stack([parameters[name] for name in param_names] + [fitness])
        correlation_matrix = np.corrcoef(data.T)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-1, vmax=1)

        # Labels
        labels = param_names + ['Fitness']
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=11, fontweight='bold')

        # Annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=8)

        ax.set_title('Parameter-Fitness Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation plot saved: {save_path}")

    def create_optimization_dashboard(self, optimization_data: Dict,
                                     save_name: str = "optimization_dashboard.png"):
        """Create comprehensive optimization dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Convergence plot
        ax1 = fig.add_subplot(gs[0, :2])
        fitness_history = optimization_data['fitness_history']
        best_fitness_history = optimization_data['best_fitness_history']
        trials = np.arange(len(fitness_history))

        ax1.plot(trials, fitness_history, 'o', alpha=0.3, markersize=3, label='Trial Fitness')
        ax1.plot(trials, best_fitness_history, '-', linewidth=2.5, label='Best Fitness', color='red')
        ax1.set_xlabel('Trial', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Fitness', fontsize=11, fontweight='bold')
        ax1.set_title('Optimization Convergence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Fitness distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(fitness_history, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(np.mean(fitness_history), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(fitness_history):.3f}')
        ax2.set_xlabel('Fitness', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title('Fitness Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Improvement rate
        ax3 = fig.add_subplot(gs[1, 0])
        if len(best_fitness_history) > 1:
            improvements = np.diff(best_fitness_history)
            ax3.plot(trials[1:], improvements, linewidth=2, color='green')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax3.fill_between(trials[1:], 0, improvements, alpha=0.3, color='green')
            ax3.set_xlabel('Trial', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Improvement', fontsize=11, fontweight='bold')
            ax3.set_title('Per-Trial Improvement', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Statistics summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        stats_text = f"""
        OPTIMIZATION STATISTICS

        Total Trials: {len(fitness_history)}
        Best Fitness: {max(fitness_history):.6f}
        Mean Fitness: {np.mean(fitness_history):.6f}
        Std Dev: {np.std(fitness_history):.6f}

        Best Trial: {np.argmax(fitness_history)}
        Improvement: {(max(fitness_history) - fitness_history[0]):.6f}

        Success Rate: {(np.array(fitness_history) > 0).sum() / len(fitness_history) * 100:.1f}%
        """

        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

        # 5. Crash analysis (if available)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'crashes' in optimization_data:
            crashes = optimization_data['crashes']
            labels = ['Successful', 'Crashed']
            sizes = [len(fitness_history) - sum(crashes), sum(crashes)]
            colors = ['#2ecc71', '#e74c3c']
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                   startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax5.set_title('Trial Success Rate', fontsize=13, fontweight='bold')

        # 6. Phase progress (if available)
        ax6 = fig.add_subplot(gs[2, :])
        if 'phases' in optimization_data:
            phases = optimization_data['phases']
            phase_names = list(phases.keys())
            phase_fitness = [phases[p]['best_fitness'] for p in phase_names]

            x_pos = np.arange(len(phase_names))
            colors_phase = plt.cm.viridis(np.linspace(0, 1, len(phase_names)))

            ax6.bar(x_pos, phase_fitness, color=colors_phase, edgecolor='black', linewidth=1.5)
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(phase_names, rotation=45, ha='right')
            ax6.set_ylabel('Best Fitness', fontsize=11, fontweight='bold')
            ax6.set_title('Performance by Optimization Phase', fontsize=13, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')

        fig.suptitle('OPTIMIZATION DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Dashboard saved: {save_path}")

    def plot_safety_violations(self, violations: List[Dict],
                               save_name: str = "safety_violations.png"):
        """Plot safety violations timeline"""
        if not violations:
            logger.info("No safety violations to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        # Count by type
        violation_types = {}
        for v in violations:
            vtype = v['type']
            if vtype not in violation_types:
                violation_types[vtype] = 0
            violation_types[vtype] += 1

        # Bar plot
        types = list(violation_types.keys())
        counts = list(violation_types.values())

        ax1.barh(types, counts, color='#e74c3c', edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Safety Violations by Type', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Timeline
        timestamps = [v['timestamp'] for v in violations]
        severity_colors = {'critical': '#e74c3c', 'warning': '#f39c12', 'info': '#3498db'}
        colors = [severity_colors.get(v.get('severity', 'warning'), '#95a5a6') for v in violations]

        ax2.scatter(timestamps, range(len(timestamps)), c=colors, s=100, edgecolors='black', linewidth=1.5)
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Violation Index', fontsize=12, fontweight='bold')
        ax2.set_title('Violations Timeline', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Safety violations plot saved: {save_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    viz = OptimizationVisualizer("./test_visualizations")

    # Generate sample data
    np.random.seed(42)
    n_trials = 100

    fitness_history = []
    best_fitness = 0.3
    for i in range(n_trials):
        improvement = 0.01 * np.exp(-i/20) + np.random.randn() * 0.02
        fitness = best_fitness + improvement
        best_fitness = max(best_fitness, fitness)
        fitness_history.append(fitness)

    best_fitness_history = np.maximum.accumulate(fitness_history)

    # Test plots
    viz.plot_convergence(fitness_history, best_fitness_history)

    # Dashboard
    optimization_data = {
        'fitness_history': fitness_history,
        'best_fitness_history': best_fitness_history,
        'crashes': [0] * n_trials,
        'phases': {
            'rate_controllers': {'best_fitness': 0.75, 'n_trials': 30, 'duration': 300},
            'attitude_controllers': {'best_fitness': 0.85, 'n_trials': 30, 'duration': 350},
            'position_controllers': {'best_fitness': 0.92, 'n_trials': 40, 'duration': 400}
        }
    }

    viz.create_optimization_dashboard(optimization_data)
    logger.info("Visualization tests complete")
