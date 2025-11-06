"""
Multi-Objective Optimization for Drone Tuning using NSGA-II

Implements Non-dominated Sorting Genetic Algorithm II (NSGA-II) to find
Pareto optimal solutions that balance multiple competing objectives:
- Stability (minimize overshoot, oscillations)
- Performance (minimize settling time)
- Robustness (maximize phase/gain margins)
- Efficiency (minimize control effort)

This provides a set of optimal trade-off solutions rather than a single
"best" solution, allowing users to choose based on mission requirements.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pymoo not installed. Multi-objective optimization unavailable.")
    logger.warning("Install with: pip install pymoo")


logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveResult:
    """Container for multi-objective optimization results"""
    pareto_front: np.ndarray          # Objective values on Pareto front
    pareto_set: np.ndarray            # Parameter values on Pareto front
    best_compromise: Dict[str, float]  # Best compromise solution
    objectives_names: List[str]        # Names of objectives
    n_solutions: int                   # Number of Pareto optimal solutions


class DroneTuningProblem(Problem):
    """
    Multi-objective optimization problem for drone tuning
    """

    def __init__(self, parameters: List[str],
                 bounds: Dict[str, Tuple[float, float]],
                 sitl_manager,
                 evaluator,
                 test_sequence_func):
        """
        Initialize drone tuning problem

        Args:
            parameters: List of parameter names to optimize
            bounds: Dictionary of (min, max) bounds for each parameter
            sitl_manager: SITL manager for running simulations
            evaluator: Performance evaluator
            test_sequence_func: Function to run test sequence
        """
        # Convert bounds to arrays
        self.parameters = parameters
        self.bounds_dict = bounds
        xl = np.array([bounds[p][0] for p in parameters])
        xu = np.array([bounds[p][1] for p in parameters])

        # 4 objectives to minimize
        n_obj = 4

        # Initialize problem
        super().__init__(n_var=len(parameters), n_obj=n_obj,
                        xl=xl, xu=xu)

        self.sitl_manager = sitl_manager
        self.evaluator = evaluator
        self.test_sequence_func = test_sequence_func

        # Objective names for reporting
        self.objective_names = [
            'Settling Time (s)',
            'Overshoot (%)',
            'Negative Phase Margin (deg)',  # Negative to minimize
            'Control Effort'
        ]

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for population X

        Args:
            X: Population array (n_individuals x n_variables)
            out: Output dictionary to store objectives
        """
        objectives = []

        for individual in X:
            # Convert individual to parameter dictionary
            params = {}
            for i, param_name in enumerate(self.parameters):
                params[param_name] = individual[i]

            # Run simulation
            instance_id = self.sitl_manager.get_instance(timeout=300)
            if instance_id is None:
                logger.error("No SITL instance available")
                # Return worst possible objectives
                objectives.append([999.0, 999.0, 999.0, 999.0])
                continue

            try:
                success, telemetry = self.sitl_manager.run_simulation(
                    instance_id=instance_id,
                    parameters=params,
                    test_sequence=self.test_sequence_func,
                    duration=30.0
                )

                if success and telemetry:
                    # Evaluate metrics
                    metrics = self.evaluator.evaluate_telemetry(telemetry)

                    # Objective 1: Minimize settling time
                    obj1_settling = metrics.settling_time

                    # Objective 2: Minimize overshoot
                    obj2_overshoot = metrics.overshoot

                    # Objective 3: Maximize phase margin (minimize negative PM)
                    # Convert to minimization: use -phase_margin or (60 - phase_margin)
                    phase_margin = metrics.phase_margin_deg
                    obj3_phase = 60.0 - phase_margin  # Lower is better (closer to 60°)

                    # Objective 4: Minimize control effort (motor saturation)
                    obj4_effort = metrics.motor_saturation_duration

                    objectives.append([obj1_settling, obj2_overshoot, obj3_phase, obj4_effort])

                else:
                    # Failed simulation - worst objectives
                    objectives.append([999.0, 999.0, 999.0, 999.0])

            except Exception as e:
                logger.error(f"Simulation error: {e}")
                objectives.append([999.0, 999.0, 999.0, 999.0])

            finally:
                self.sitl_manager.release_instance(instance_id)

        out["F"] = np.array(objectives)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using NSGA-II
    """

    def __init__(self, sitl_manager, evaluator, population_size: int = 100):
        """
        Initialize multi-objective optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            population_size: Population size for NSGA-II
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo library required for multi-objective optimization. "
                            "Install with: pip install pymoo")

        self.sitl_manager = sitl_manager
        self.evaluator = evaluator
        self.population_size = population_size

        logger.info("Multi-objective optimizer (NSGA-II) initialized")
        logger.info(f"  Population size: {population_size}")

    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                n_generations: int = 100) -> MultiObjectiveResult:
        """
        Run multi-objective optimization

        Args:
            phase_name: Name of optimization phase
            parameters: List of parameter names to optimize
            bounds: Dictionary of (min, max) bounds for each parameter
            n_generations: Number of generations to run

        Returns:
            MultiObjectiveResult with Pareto front and solutions
        """
        logger.info(f"\nStarting multi-objective optimization for {phase_name}")
        logger.info(f"Parameters: {len(parameters)}")
        logger.info(f"Objectives: 4 (settling time, overshoot, phase margin, control effort)")
        logger.info(f"Generations: {n_generations}")

        # Define problem
        problem = DroneTuningProblem(
            parameters=parameters,
            bounds=bounds,
            sitl_manager=self.sitl_manager,
            evaluator=self.evaluator,
            test_sequence_func=self._run_test_sequence
        )

        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        # Set termination criterion
        termination = get_termination("n_gen", n_generations)

        # Run optimization
        logger.info("\nRunning NSGA-II optimization...")
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=True
        )

        # Extract Pareto front
        pareto_front = res.F
        pareto_set = res.X

        logger.info(f"\n✓ Optimization complete!")
        logger.info(f"  Pareto optimal solutions found: {len(pareto_front)}")

        # Find best compromise solution (closest to ideal point)
        ideal_point = np.min(pareto_front, axis=0)
        distances = np.linalg.norm(pareto_front - ideal_point, axis=1)
        best_idx = np.argmin(distances)
        best_compromise_values = pareto_set[best_idx]

        # Convert to dictionary
        best_compromise = {}
        for i, param_name in enumerate(parameters):
            best_compromise[param_name] = best_compromise_values[i]

        logger.info(f"\nBest compromise solution (closest to ideal point):")
        logger.info(f"  Settling time: {pareto_front[best_idx, 0]:.3f} s")
        logger.info(f"  Overshoot: {pareto_front[best_idx, 1]:.1f} %")
        logger.info(f"  Phase margin: {60 - pareto_front[best_idx, 2]:.1f}°")
        logger.info(f"  Control effort: {pareto_front[best_idx, 3]:.2f}")

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            pareto_set=pareto_set,
            best_compromise=best_compromise,
            objectives_names=problem.objective_names,
            n_solutions=len(pareto_front)
        )

    def _run_test_sequence(self, connection, duration: float) -> Tuple[bool, Dict]:
        """Run test sequence and collect telemetry"""
        import os
        from .mission_executor import run_mission_test

        script_dir = os.path.dirname(os.path.abspath(__file__))
        mission_file = os.path.join(script_dir, "missions", "simple_hover.waypoints")
        timeout = 120.0

        return run_mission_test(connection, mission_file, timeout)

    def visualize_pareto_front(self, result: MultiObjectiveResult,
                               output_file: str = "pareto_front.png"):
        """
        Visualize Pareto front (2D projections)

        Args:
            result: MultiObjectiveResult from optimization
            output_file: Output file path for plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pareto Front - Multi-Objective Drone Tuning', fontsize=16)

        pareto_front = result.pareto_front
        obj_names = result.objectives_names

        # Plot all pairwise combinations
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for idx, (i, j) in enumerate(pairs):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            ax.scatter(pareto_front[:, i], pareto_front[:, j], alpha=0.6)
            ax.set_xlabel(obj_names[i])
            ax.set_ylabel(obj_names[j])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        logger.info(f"✓ Pareto front visualization saved: {output_file}")
