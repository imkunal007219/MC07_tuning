"""
Hierarchical Parameter Optimization
Implements sequential optimization: Rate Controllers → Attitude Controllers → Position Controllers
This approach is critical for PID tuning as inner loops must be stable before outer loops
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


@dataclass
class ParameterGroup:
    """Group of related parameters to optimize together"""
    name: str
    parameters: Dict[str, Tuple[float, float]]  # param_name: (min, max)
    dependencies: List[str]  # Names of groups that must be optimized first
    priority: int  # Lower = higher priority


class HierarchicalOptimizer:
    """
    Hierarchical optimization for ArduPilot PID parameters
    Follows control theory best practices: tune from innermost to outermost loop
    """

    def __init__(self, sitl_manager, performance_evaluator, n_trials_per_phase: int = 50):
        self.sitl_manager = sitl_manager
        self.evaluator = performance_evaluator
        self.n_trials_per_phase = n_trials_per_phase
        self.optimized_params = {}
        self.optimization_history = []

        # Define parameter groups in hierarchical order
        self.parameter_groups = self._define_parameter_groups()

    def _define_parameter_groups(self) -> List[ParameterGroup]:
        """
        Define hierarchical parameter groups
        Order: Rate → Attitude → Position → Advanced
        """
        groups = []

        # ============================================================
        # PHASE 1: RATE CONTROLLERS (MOST CRITICAL - INNER LOOP)
        # ============================================================

        # Roll Rate Controller
        groups.append(ParameterGroup(
            name="rate_roll",
            parameters={
                'ATC_RAT_RLL_P': (0.05, 0.30),      # P gain
                'ATC_RAT_RLL_I': (0.05, 0.30),      # I gain
                'ATC_RAT_RLL_D': (0.001, 0.020),    # D gain
                'ATC_RAT_RLL_FLTD': (10.0, 50.0),   # D term filter
                'ATC_RAT_RLL_FLTE': (0.0, 5.0),     # Error filter
                'ATC_RAT_RLL_FLTT': (10.0, 50.0),   # Target filter
            },
            dependencies=[],
            priority=1
        ))

        # Pitch Rate Controller
        groups.append(ParameterGroup(
            name="rate_pitch",
            parameters={
                'ATC_RAT_PIT_P': (0.05, 0.30),
                'ATC_RAT_PIT_I': (0.05, 0.30),
                'ATC_RAT_PIT_D': (0.001, 0.020),
                'ATC_RAT_PIT_FLTD': (10.0, 50.0),
                'ATC_RAT_PIT_FLTE': (0.0, 5.0),
                'ATC_RAT_PIT_FLTT': (10.0, 50.0),
            },
            dependencies=[],
            priority=1
        ))

        # Yaw Rate Controller
        groups.append(ParameterGroup(
            name="rate_yaw",
            parameters={
                'ATC_RAT_YAW_P': (0.10, 0.80),      # Yaw typically needs higher P
                'ATC_RAT_YAW_I': (0.01, 0.10),
                'ATC_RAT_YAW_D': (0.0, 0.005),      # D rarely used for yaw
                'ATC_RAT_YAW_FLTD': (0.0, 10.0),
                'ATC_RAT_YAW_FLTE': (0.0, 5.0),
                'ATC_RAT_YAW_FLTT': (10.0, 50.0),
            },
            dependencies=[],
            priority=1
        ))

        # Rate Controller Limits
        groups.append(ParameterGroup(
            name="rate_limits",
            parameters={
                'ATC_RATE_P_MAX': (90.0, 360.0),    # Max roll rate deg/s
                'ATC_RATE_R_MAX': (90.0, 360.0),    # Max pitch rate deg/s
                'ATC_RATE_Y_MAX': (45.0, 180.0),    # Max yaw rate deg/s
            },
            dependencies=['rate_roll', 'rate_pitch', 'rate_yaw'],
            priority=2
        ))

        # ============================================================
        # PHASE 2: ATTITUDE CONTROLLERS (MIDDLE LOOP)
        # ============================================================

        # Attitude Stabilization
        groups.append(ParameterGroup(
            name="attitude_control",
            parameters={
                'ATC_ANG_RLL_P': (3.0, 12.0),       # Roll angle P
                'ATC_ANG_PIT_P': (3.0, 12.0),       # Pitch angle P
                'ATC_ANG_YAW_P': (3.0, 12.0),       # Yaw angle P
            },
            dependencies=['rate_roll', 'rate_pitch', 'rate_yaw'],
            priority=3
        ))

        # Attitude Acceleration Limits
        groups.append(ParameterGroup(
            name="attitude_limits",
            parameters={
                'ATC_ACCEL_P_MAX': (50000.0, 180000.0),  # Pitch accel deg/s²
                'ATC_ACCEL_R_MAX': (50000.0, 180000.0),  # Roll accel deg/s²
                'ATC_ACCEL_Y_MAX': (10000.0, 72000.0),   # Yaw accel deg/s²
                'ANGLE_MAX': (1000.0, 4500.0),           # Max lean angle (cdeg)
            },
            dependencies=['attitude_control'],
            priority=4
        ))

        # Input Shaping
        groups.append(ParameterGroup(
            name="input_shaping",
            parameters={
                'ATC_INPUT_TC': (0.10, 0.50),       # Input time constant
                'ATC_SLEW_YAW': (1000.0, 6000.0),   # Yaw slew rate
            },
            dependencies=['attitude_control'],
            priority=4
        ))

        # ============================================================
        # PHASE 3: POSITION CONTROLLERS (OUTER LOOP)
        # ============================================================

        # Horizontal Position
        groups.append(ParameterGroup(
            name="position_xy",
            parameters={
                'PSC_POSXY_P': (0.5, 3.0),          # XY position P
                'PSC_VELXY_P': (0.5, 5.0),          # XY velocity P
                'PSC_VELXY_I': (0.1, 2.0),          # XY velocity I
                'PSC_VELXY_D': (0.0, 1.0),          # XY velocity D
            },
            dependencies=['attitude_control'],
            priority=5
        ))

        # Vertical Position
        groups.append(ParameterGroup(
            name="position_z",
            parameters={
                'PSC_POSZ_P': (0.5, 3.0),           # Z position P
                'PSC_VELZ_P': (1.0, 10.0),          # Z velocity P
                'PSC_VELZ_I': (0.0, 3.0),           # Z velocity I (optional)
                'PSC_VELZ_D': (0.0, 1.0),           # Z velocity D (optional)
            },
            dependencies=['rate_yaw'],  # Altitude control depends on stable yaw
            priority=5
        ))

        # Altitude Acceleration
        groups.append(ParameterGroup(
            name="altitude_accel",
            parameters={
                'PSC_ACCZ_P': (0.1, 1.0),           # Z accel P
                'PSC_ACCZ_I': (0.1, 3.0),           # Z accel I
                'PSC_ACCZ_D': (0.0, 0.1),           # Z accel D
            },
            dependencies=['position_z'],
            priority=6
        ))

        # Position Limits
        groups.append(ParameterGroup(
            name="position_limits",
            parameters={
                'PSC_VELXY_MAX': (500.0, 2000.0),   # Max XY velocity cm/s
                'PSC_VELXY_JERK': (5.0, 50.0),      # XY jerk m/s³
                'PILOT_SPEED_UP': (50.0, 500.0),    # Max climb rate cm/s
                'PILOT_SPEED_DN': (50.0, 500.0),    # Max descend rate cm/s
                'PILOT_ACCEL_Z': (100.0, 500.0),    # Pilot vertical accel cm/s²
            },
            dependencies=['position_xy', 'position_z'],
            priority=7
        ))

        # ============================================================
        # PHASE 4: ADVANCED PARAMETERS
        # ============================================================

        # Motor Parameters
        groups.append(ParameterGroup(
            name="motor_params",
            parameters={
                'MOT_THST_HOVER': (0.20, 0.50),     # Hover throttle
                'MOT_SPIN_MIN': (0.05, 0.20),       # Min motor spin
                'MOT_SPIN_MAX': (0.90, 0.98),       # Max motor spin
                'MOT_THST_EXPO': (0.50, 0.80),      # Thrust curve expo
                'MOT_BAT_VOLT_MAX': (48.0, 52.0),   # Max battery voltage (12S)
                'MOT_BAT_VOLT_MIN': (39.6, 42.0),   # Min battery voltage
            },
            dependencies=['rate_roll', 'rate_pitch', 'rate_yaw'],
            priority=8
        ))

        # Sensor Filters
        groups.append(ParameterGroup(
            name="sensor_filters",
            parameters={
                'INS_GYRO_FILTER': (10.0, 80.0),    # Gyro filter Hz
                'INS_ACCEL_FILTER': (10.0, 40.0),   # Accel filter Hz
            },
            dependencies=[],
            priority=9
        ))

        # Feedforward
        groups.append(ParameterGroup(
            name="feedforward",
            parameters={
                'ATC_RAT_RLL_FF': (0.0, 0.5),       # Roll feedforward
                'ATC_RAT_PIT_FF': (0.0, 0.5),       # Pitch feedforward
                'ATC_RAT_YAW_FF': (0.0, 0.5),       # Yaw feedforward
            },
            dependencies=['rate_roll', 'rate_pitch', 'rate_yaw'],
            priority=10
        ))

        # Navigation
        groups.append(ParameterGroup(
            name="navigation",
            parameters={
                'WPNAV_SPEED': (500.0, 2000.0),     # Waypoint speed cm/s
                'WPNAV_RADIUS': (50.0, 500.0),      # Waypoint radius cm
                'WPNAV_ACCEL': (100.0, 500.0),      # Waypoint accel cm/s²
            },
            dependencies=['position_xy'],
            priority=11
        ))

        return sorted(groups, key=lambda g: g.priority)

    def optimize_all_phases(self) -> Dict[str, float]:
        """
        Execute hierarchical optimization through all phases
        Returns final optimized parameter set
        """
        logger.info("=" * 80)
        logger.info("STARTING HIERARCHICAL OPTIMIZATION")
        logger.info("=" * 80)

        total_start_time = time.time()

        for group in self.parameter_groups:
            self._optimize_group(group)

        total_duration = time.time() - total_start_time

        logger.info("=" * 80)
        logger.info(f"HIERARCHICAL OPTIMIZATION COMPLETE - Duration: {total_duration/60:.1f} min")
        logger.info(f"Total Parameters Optimized: {len(self.optimized_params)}")
        logger.info("=" * 80)

        return self.optimized_params

    def _optimize_group(self, group: ParameterGroup) -> Dict[str, float]:
        """Optimize a single parameter group"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"OPTIMIZING GROUP: {group.name.upper()}")
        logger.info(f"Priority: {group.priority}")
        logger.info(f"Parameters: {list(group.parameters.keys())}")
        logger.info(f"Dependencies: {group.dependencies if group.dependencies else 'None'}")
        logger.info("=" * 80)

        # Verify dependencies are met
        for dep in group.dependencies:
            if dep not in [g.name for g in self.parameter_groups if g.name in str(self.optimization_history)]:
                logger.warning(f"Dependency {dep} not yet optimized, but proceeding...")

        # Create Optuna study
        study_name = f"hierarchical_{group.name}_{int(time.time())}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # Define objective function for this group
        def objective(trial):
            # Start with already optimized parameters
            params = self.optimized_params.copy()

            # Sample new values for current group
            for param_name, (min_val, max_val) in group.parameters.items():
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

            # Evaluate fitness
            fitness, crashed = self._evaluate_parameters(params, group.name)

            # Log trial
            logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}, Crashed = {crashed}")

            return fitness

        # Run optimization
        logger.info(f"Running {self.n_trials_per_phase} trials for {group.name}...")
        start_time = time.time()

        try:
            study.optimize(objective, n_trials=self.n_trials_per_phase, n_jobs=1)
        except Exception as e:
            logger.error(f"Optimization failed for {group.name}: {e}")
            return {}

        duration = time.time() - start_time

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info("")
        logger.info(f"BEST RESULT for {group.name}:")
        logger.info(f"  Fitness: {best_value:.4f}")
        logger.info(f"  Duration: {duration/60:.1f} min")
        logger.info(f"  Trials: {len(study.trials)}")
        logger.info(f"  Parameters:")
        for param, value in best_params.items():
            logger.info(f"    {param}: {value:.6f}")

        # Update optimized parameters
        self.optimized_params.update(best_params)

        # Store history
        self.optimization_history.append({
            'group': group.name,
            'priority': group.priority,
            'best_fitness': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'duration': duration
        })

        return best_params

    def _evaluate_parameters(self, params: Dict[str, float], group_name: str) -> Tuple[float, bool]:
        """
        Evaluate parameter set with appropriate tests for current optimization phase
        """
        # Determine which tests to run based on optimization phase
        if 'rate' in group_name.lower():
            # Rate controllers: focus on attitude step responses
            test_types = ['step_response_roll', 'step_response_pitch', 'step_response_yaw']
            weights = {'stability': 0.4, 'response_time': 0.3, 'overshoot': 0.2, 'tracking': 0.1}

        elif 'attitude' in group_name.lower():
            # Attitude controllers: broader attitude tests
            test_types = ['step_response_roll', 'step_response_pitch', 'hover_stability']
            weights = {'stability': 0.3, 'response_time': 0.2, 'overshoot': 0.2, 'tracking': 0.3}

        elif 'position' in group_name.lower() or 'altitude' in group_name.lower():
            # Position controllers: altitude and position tests
            test_types = ['step_response_altitude', 'hover_stability', 'trajectory_square']
            weights = {'stability': 0.3, 'response_time': 0.2, 'overshoot': 0.1, 'tracking': 0.4}

        else:
            # Advanced parameters: full test suite
            test_types = ['hover_stability', 'step_response_altitude', 'trajectory_square', 'disturbance']
            weights = {'stability': 0.25, 'response_time': 0.15, 'overshoot': 0.15, 'tracking': 0.3, 'efficiency': 0.15}

        # Run evaluation through performance evaluator
        fitness = self.evaluator.evaluate(params, test_types=test_types, weights=weights)

        # Check for crash (negative fitness indicates crash)
        crashed = fitness < -1000

        return fitness, crashed

    def get_optimization_summary(self) -> str:
        """Generate summary of hierarchical optimization"""
        summary = []
        summary.append("=" * 80)
        summary.append("HIERARCHICAL OPTIMIZATION SUMMARY")
        summary.append("=" * 80)
        summary.append("")

        for i, phase in enumerate(self.optimization_history, 1):
            summary.append(f"Phase {i}: {phase['group'].upper()}")
            summary.append(f"  Priority: {phase['priority']}")
            summary.append(f"  Best Fitness: {phase['best_fitness']:.4f}")
            summary.append(f"  Trials: {phase['n_trials']}")
            summary.append(f"  Duration: {phase['duration']/60:.1f} min")
            summary.append(f"  Parameters Optimized: {len(phase['best_params'])}")
            summary.append("")

        summary.append(f"Total Parameters: {len(self.optimized_params)}")
        summary.append(f"Total Phases: {len(self.optimization_history)}")

        total_duration = sum(p['duration'] for p in self.optimization_history)
        summary.append(f"Total Duration: {total_duration/60:.1f} min ({total_duration/3600:.2f} hours)")
        summary.append("")
        summary.append("=" * 80)

        return "\n".join(summary)

    def save_parameters(self, filepath: str):
        """Save optimized parameters to ArduPilot .parm file"""
        logger.info(f"Saving optimized parameters to {filepath}")

        with open(filepath, 'w') as f:
            f.write("# Hierarchically Optimized Parameters for 30kg Quadcopter\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Parameters: {len(self.optimized_params)}\n")
            f.write(f"# Optimization Phases: {len(self.optimization_history)}\n")
            f.write("\n")

            # Group parameters by phase
            for phase in self.optimization_history:
                f.write(f"# Phase: {phase['group']} (Priority {phase['priority']})\n")
                f.write(f"# Fitness: {phase['best_fitness']:.4f}\n")

                for param_name, value in phase['best_params'].items():
                    f.write(f"{param_name},{value:.6f}\n")

                f.write("\n")

        logger.info(f"Parameters saved successfully")

    def visualize_convergence(self):
        """Generate convergence plots for each phase"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(self.optimization_history), 1, figsize=(12, 3*len(self.optimization_history)))

        if len(self.optimization_history) == 1:
            axes = [axes]

        for i, (ax, phase) in enumerate(zip(axes, self.optimization_history)):
            # Plot would show trial-by-trial fitness improvement
            ax.set_title(f"Phase {i+1}: {phase['group']}")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            ax.axhline(y=phase['best_fitness'], color='r', linestyle='--', label='Best')
            ax.legend()

        plt.tight_layout()
        plt.savefig('optimization_convergence.png', dpi=150)
        logger.info("Convergence plot saved to optimization_convergence.png")


class QuickTuneOptimizer(HierarchicalOptimizer):
    """
    Faster version for initial rough tuning
    Uses fewer trials and simplified parameter groups
    """

    def __init__(self, sitl_manager, performance_evaluator):
        super().__init__(sitl_manager, performance_evaluator, n_trials_per_phase=20)

    def _define_parameter_groups(self) -> List[ParameterGroup]:
        """Simplified parameter groups for quick tuning"""
        groups = []

        # Quick rate tuning (combined roll/pitch)
        groups.append(ParameterGroup(
            name="rate_controllers",
            parameters={
                'ATC_RAT_RLL_P': (0.05, 0.30),
                'ATC_RAT_RLL_I': (0.05, 0.30),
                'ATC_RAT_RLL_D': (0.001, 0.020),
                'ATC_RAT_PIT_P': (0.05, 0.30),
                'ATC_RAT_PIT_I': (0.05, 0.30),
                'ATC_RAT_PIT_D': (0.001, 0.020),
                'ATC_RAT_YAW_P': (0.10, 0.80),
                'ATC_RAT_YAW_I': (0.01, 0.10),
            },
            dependencies=[],
            priority=1
        ))

        # Quick attitude tuning
        groups.append(ParameterGroup(
            name="attitude_controllers",
            parameters={
                'ATC_ANG_RLL_P': (3.0, 12.0),
                'ATC_ANG_PIT_P': (3.0, 12.0),
                'ATC_ANG_YAW_P': (3.0, 12.0),
            },
            dependencies=['rate_controllers'],
            priority=2
        ))

        # Quick position tuning
        groups.append(ParameterGroup(
            name="position_controllers",
            parameters={
                'PSC_POSXY_P': (0.5, 3.0),
                'PSC_VELXY_P': (0.5, 5.0),
                'PSC_POSZ_P': (0.5, 3.0),
                'PSC_VELZ_P': (1.0, 10.0),
            },
            dependencies=['attitude_controllers'],
            priority=3
        ))

        return sorted(groups, key=lambda g: g.priority)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Hierarchical Optimizer initialized")
