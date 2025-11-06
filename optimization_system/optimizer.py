"""
Optimization Algorithms for Automated Drone Tuning

Implements both Genetic Algorithm and Bayesian Optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import random
import pickle
from deap import base, creator, tools, algorithms
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import copy

from .flight_logger import FlightDataLogger
from .flight_analyzer import FlightAnalyzer
from .report_generator import ReportGenerator
from .physics_based_seeding import PhysicsBasedSeeder
from .hierarchical_constraints import HierarchicalConstraintValidator
from .intelligent_test_sequencing import IntelligentTestSequencer


logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base class for optimization algorithms"""

    def __init__(self, sitl_manager, evaluator, max_iterations: int = 100,
                 log_dir: str = "flight_logs"):
        """
        Initialize optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_iterations: Maximum optimization iterations
            log_dir: Directory for flight logs and analysis
        """
        self.sitl_manager = sitl_manager
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.best_params = None
        self.best_fitness = -np.inf
        self.convergence_history = []

        # Initialize logging and analysis system
        self.flight_logger = FlightDataLogger(log_dir=log_dir)
        self.flight_analyzer = FlightAnalyzer(self.flight_logger)
        self.report_generator = ReportGenerator(self.flight_logger, self.flight_analyzer)

        logger.info(f"Flight logging initialized: {log_dir}")

    @abstractmethod
    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                resume_from: Optional[str] = None) -> Tuple[Dict, float, List]:
        """
        Run optimization

        Args:
            phase_name: Name of optimization phase
            parameters: List of parameter names to optimize
            bounds: Dictionary of (min, max) bounds for each parameter
            resume_from: Path to checkpoint file to resume from

        Returns:
            (best_params, best_fitness, convergence_history)
        """
        pass


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm optimizer using DEAP"""

    def __init__(self, sitl_manager, evaluator, max_generations: int = 100,
                 population_size: int = 50, mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7, drone_params: Dict = None,
                 use_physics_seeding: bool = True,
                 enforce_hierarchical_constraints: bool = True,
                 use_intelligent_sequencing: bool = True):
        """
        Initialize Genetic Algorithm optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_generations: Maximum number of generations
            population_size: Population size
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            drone_params: Dictionary of drone physical parameters
            use_physics_seeding: Whether to use physics-based population seeding
            enforce_hierarchical_constraints: Whether to enforce bandwidth separation constraints
            use_intelligent_sequencing: Whether to use progressive test sequencing
        """
        super().__init__(sitl_manager, evaluator, max_generations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.toolbox = None
        self.checkpoint_freq = 10  # Save checkpoint every N generations

        # Initialize physics-based seeding
        self.use_physics_seeding = use_physics_seeding
        self.physics_seeder = None
        if use_physics_seeding and drone_params:
            self.physics_seeder = PhysicsBasedSeeder(drone_params)
            logger.info("Physics-based population seeding enabled")
        else:
            logger.info("Using random population initialization")

        # Initialize hierarchical constraint validator
        self.hierarchical_validator = None
        if enforce_hierarchical_constraints and drone_params:
            self.hierarchical_validator = HierarchicalConstraintValidator(
                drone_params=drone_params,
                enforce_constraints=True,
                penalty_multiplier=1000.0
            )
            logger.info("Hierarchical bandwidth constraints enabled")
        else:
            logger.info("Hierarchical constraints disabled")

        # Initialize intelligent test sequencing
        self.use_intelligent_sequencing = use_intelligent_sequencing
        self.test_sequencer = None
        if use_intelligent_sequencing:
            from config import INTELLIGENT_TEST_SEQUENCING_CONFIG
            self.test_sequencer = IntelligentTestSequencer(
                min_pass_score=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('min_pass_score', 60.0),
                enable_optional_tests=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('enable_optional_tests', True),
                early_termination=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('early_termination', True)
            )
            logger.info("Intelligent test sequencing enabled (saves ~40% evaluation time)")
        else:
            logger.info("Using single mission test mode")

        # Adaptive bounds tracking
        self.use_adaptive_bounds = True
        self.adaptive_bounds_interval = 10  # Update every N generations
        self.min_successful_samples = 20   # Minimum samples before adapting
        self.successful_params = []         # Track successful parameter sets
        self.failed_params = []             # Track failed parameter sets
        self.current_bounds = None          # Will be set during optimization
        self.hard_bounds = None             # Original bounds (hard limits)
        logger.info(f"Adaptive bounds enabled (update every {self.adaptive_bounds_interval} gen)")

    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                resume_from: Optional[str] = None) -> Tuple[Dict, float, List]:
        """Run genetic algorithm optimization"""

        logger.info(f"Starting GA optimization for {phase_name}")
        logger.info(f"Parameters to optimize: {parameters}")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Max generations: {self.max_iterations}")

        # Initialize adaptive bounds
        self.current_bounds = copy.deepcopy(bounds)
        self.hard_bounds = copy.deepcopy(bounds)
        self.successful_params = []
        self.failed_params = []

        # Setup DEAP
        self._setup_deap(parameters, self.current_bounds)

        # Initialize or resume population
        if resume_from:
            pop, gen_start, hof = self._load_checkpoint(resume_from)
            logger.info(f"Resumed from generation {gen_start}")
        else:
            # Use physics-based seeding if available
            if self.use_physics_seeding and self.physics_seeder:
                logger.info("Generating physics-based initial population...")
                population_list = self.physics_seeder.generate_population(
                    parameters=parameters,
                    bounds=bounds,
                    population_size=self.population_size,
                    seed_ratio=0.3,  # 30% seeded, 70% random for diversity
                    diversity_sigma=0.15  # 15% variation around seed values
                )
                # Convert to DEAP individuals
                pop = []
                for individual_list in population_list:
                    ind = creator.Individual(individual_list)
                    pop.append(ind)
                logger.info(f"Created population with {len(pop)} individuals (physics-seeded)")
            else:
                # Fallback to random initialization
                pop = self.toolbox.population(n=self.population_size)
                logger.info(f"Created random population with {len(pop)} individuals")

            gen_start = 0
            hof = tools.HallOfFame(1)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Reset convergence history
        self.convergence_history = []

        # Run evolution
        for gen in range(gen_start, self.max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {gen + 1}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            # Evaluate population (with logging and hierarchical constraints)
            fitnesses = self._evaluate_population(pop, parameters, bounds,
                                                 generation=gen+1, phase_name=phase_name)

            # Assign fitness to individuals and track for adaptive bounds
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = (fit,)

                # Track successful/failed parameters for adaptive bounds
                if self.use_adaptive_bounds:
                    params = self._individual_to_params(ind, parameters, self.current_bounds)
                    if fit > 0:  # Successful (positive fitness)
                        self.successful_params.append(params.copy())
                    else:  # Failed (negative fitness)
                        self.failed_params.append(params.copy())

            # Update hall of fame
            hof.update(pop)

            # Record statistics
            record = stats.compile(pop)
            self.convergence_history.append({
                'generation': gen + 1,
                'avg_fitness': record['avg'],
                'std_fitness': record['std'],
                'min_fitness': record['min'],
                'max_fitness': record['max'],
                'best_fitness': hof[0].fitness.values[0]
            })

            logger.info(f"Avg fitness: {record['avg']:.4f}")
            logger.info(f"Max fitness: {record['max']:.4f}")
            logger.info(f"Best overall: {hof[0].fitness.values[0]:.4f}")

            # Log current statistics
            stats_summary = self.flight_logger.get_statistics()
            logger.info(f"Total flights logged: {stats_summary['total_flights']} "
                       f"(Success rate: {stats_summary['success_rate']:.1%})")

            # Generate analysis report every 5 generations
            if (gen + 1) % 5 == 0:
                logger.info("Generating interim analysis report...")
                report_file = f"reports/optimization_gen{gen+1}.html"
                try:
                    import os
                    os.makedirs("reports", exist_ok=True)
                    self.report_generator.generate_html_report(report_file)
                    logger.info(f"✓ Report generated: {report_file}")
                except Exception as e:
                    logger.warning(f"Could not generate report: {e}")

            # Check convergence
            if self._check_convergence():
                logger.info("Convergence criteria met - stopping early")
                break

            # Save checkpoint periodically
            if (gen + 1) % self.checkpoint_freq == 0:
                checkpoint_file = f"/tmp/ga_checkpoint_gen{gen+1}.pkl"
                self._save_checkpoint(checkpoint_file, pop, gen + 1, hof)
                logger.info(f"Checkpoint saved: {checkpoint_file}")

            # Update adaptive bounds periodically
            if (self.use_adaptive_bounds and
                (gen + 1) % self.adaptive_bounds_interval == 0 and
                len(self.successful_params) >= self.min_successful_samples):
                self._update_adaptive_bounds(parameters)
                # Regenerate toolbox with new bounds
                self._setup_deap(parameters, self.current_bounds)

            # Select next generation
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Elite preservation: ensure best individual survives
            # Clone the elite before replacing population
            if len(hof) > 0:
                elite = self.toolbox.clone(hof[0])

                # Replace worst individual with elite to guarantee best solution persists
                # Find the individual with worst fitness (or no fitness yet)
                worst_idx = 0
                worst_fitness = float('inf')

                for idx, ind in enumerate(offspring):
                    if not ind.fitness.valid:
                        # Individual hasn't been evaluated yet - replace it
                        worst_idx = idx
                        break
                    elif ind.fitness.values[0] < worst_fitness:
                        worst_fitness = ind.fitness.values[0]
                        worst_idx = idx

                # Inject elite into population
                offspring[worst_idx] = elite
                logger.debug(f"Elite preserved with fitness: {elite.fitness.values[0]:.4f}")

            # Replace population
            pop[:] = offspring

        # Extract best parameters
        best_individual = hof[0]
        best_params = self._individual_to_params(best_individual, parameters, bounds)
        best_fitness = best_individual.fitness.values[0]

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Store optimized parameters for hierarchical constraint validation
        if self.hierarchical_validator:
            self.hierarchical_validator.set_optimized_phase(phase_name, best_params)
            logger.info(f"✓ Stored optimized parameters for {phase_name}")

        # Generate final comprehensive report
        logger.info("\nGenerating final analysis report...")
        try:
            import os
            os.makedirs("reports", exist_ok=True)

            # HTML report with visualizations
            final_report = "reports/final_optimization_report.html"
            self.report_generator.generate_html_report(final_report)
            logger.info(f"✓ Final HTML report: {final_report}")

            # JSON analysis export
            json_report = "reports/analysis.json"
            self.flight_analyzer.export_analysis_json(json_report)
            logger.info(f"✓ Analysis data exported: {json_report}")

            # CSV data export for external analysis
            csv_export = "reports/all_flights.csv"
            self.flight_logger.export_csv(csv_export)
            logger.info(f"✓ Flight data CSV: {csv_export}")

            # Print key recommendations
            summary = self.flight_analyzer.generate_summary_report()
            logger.info("\n" + "="*60)
            logger.info("KEY RECOMMENDATIONS:")
            logger.info("="*60)
            for rec in summary.get('recommendations', []):
                logger.info(rec)

        except Exception as e:
            logger.warning(f"Could not generate final report: {e}")

        return best_params, best_fitness, self.convergence_history

    def _setup_deap(self, parameters: List[str], bounds: Dict[str, Tuple[float, float]]):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator - normalized [0, 1] for each parameter
        self.toolbox.register("attr_float", random.random)

        # Individual and population generators
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=len(parameters))
        self.toolbox.register("population", tools.initRepeat, list,
                            self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_population(self, population: List, parameters: List[str],
                            bounds: Dict[str, Tuple[float, float]],
                            generation: int = None,
                            phase_name: str = None) -> List[float]:
        """Evaluate fitness of entire population in parallel"""

        logger.info(f"Evaluating population of {len(population)} individuals...")

        # Convert individuals to parameter dictionaries
        param_sets = []
        for individual in population:
            params = self._individual_to_params(individual, parameters, bounds)
            param_sets.append(params)

        # Run simulations in parallel
        results = self.sitl_manager.run_parallel_simulations(
            parameter_sets=param_sets,
            test_sequence=self._run_test_sequence,
            duration=30.0  # 30 second test
        )

        # Calculate fitness for each result AND log each flight
        fitnesses = []
        for idx, (success, telemetry) in enumerate(results):
            if success and telemetry:
                metrics = self.evaluator.evaluate_telemetry(telemetry)
                fitness = metrics.fitness

                # Apply hierarchical constraint penalty if enabled
                if self.hierarchical_validator and phase_name:
                    constraint_penalty = self.hierarchical_validator.get_constraint_penalty(
                        phase_name=phase_name,
                        params=param_sets[idx]
                    )
                    if constraint_penalty < 0:
                        # Constraint violated - apply penalty
                        logger.debug(f"Individual {idx}: constraint violation penalty = {constraint_penalty}")
                        fitness += constraint_penalty

                fitnesses.append(fitness)
            else:
                # Failed simulation - very poor fitness
                fitness = -1000.0
                fitnesses.append(fitness)

            # Log flight with parameters and telemetry for automated analysis
            flight_id = self.flight_logger.log_flight(
                parameters=param_sets[idx],
                telemetry=telemetry if telemetry else {},
                success=success,
                generation=generation,
                individual_id=idx
            )

            if flight_id:
                logger.debug(f"Flight {flight_id} logged (fitness: {fitness:.4f})")

        logger.info(f"Evaluated and logged {len(fitnesses)} individuals")
        return fitnesses

    def _individual_to_params(self, individual: List[float], parameters: List[str],
                             bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Convert normalized individual to parameter dictionary"""
        params = {}
        for i, param_name in enumerate(parameters):
            min_val, max_val = bounds[param_name]
            # Scale from [0, 1] to [min, max]
            normalized_value = max(0.0, min(1.0, individual[i]))
            params[param_name] = min_val + normalized_value * (max_val - min_val)
        return params

    def _run_test_sequence(self, connection, duration: float) -> Tuple[bool, Dict]:
        """
        Run test sequence and collect telemetry

        Uses progressive mission files for intelligent testing when enabled,
        otherwise falls back to single mission test.
        """
        import os
        from mission_executor import MissionExecutor

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Use intelligent test sequencing if enabled
        if self.use_intelligent_sequencing and self.test_sequencer:
            return self._run_progressive_tests(connection, script_dir)
        else:
            # Fallback: single mission test
            from mission_executor import run_mission_test
            mission_file = os.path.join(script_dir, "missions", "simple_hover.waypoints")
            timeout = 120.0
            logger.info(f"Running single mission test: {mission_file}")
            return run_mission_test(connection, mission_file, timeout)

    def _run_progressive_tests(self, connection, script_dir: str) -> Tuple[bool, Dict]:
        """
        Run progressive test sequence using IntelligentTestSequencer

        Args:
            connection: MAVLink connection
            script_dir: Script directory for mission files

        Returns:
            (overall_success, combined_telemetry)
        """
        from mission_executor import MissionExecutor

        executor = MissionExecutor(connection, enable_early_crash_detection=True)

        # Mission files for each test level
        mission_files = {
            'hover': os.path.join(script_dir, "missions", "level1_hover.waypoints"),
            'small_step': os.path.join(script_dir, "missions", "level2_small_step.waypoints"),
            'large_step': os.path.join(script_dir, "missions", "level3_large_step.waypoints"),
            'frequency': os.path.join(script_dir, "missions", "level4_frequency.waypoints"),
            'trajectory': os.path.join(script_dir, "missions", "level5_trajectory.waypoints"),
        }

        # Test timeouts for each level
        timeouts = {
            'hover': 40.0,       # 30s sensor + 10s mission
            'small_step': 50.0,
            'large_step': 60.0,
            'frequency': 80.0,
            'trajectory': 100.0,
        }

        # Run progressive tests
        test_results = []
        combined_telemetry = {'time': [], 'roll': [], 'pitch': [], 'yaw': [],
                            'altitude': [], 'roll_rate': [], 'pitch_rate': [], 'yaw_rate': []}
        overall_success = True

        # Level 1: Hover (REQUIRED)
        logger.info("Running Level 1: Hover test (5s)")
        success, telemetry = executor.run_mission(mission_files['hover'], timeouts['hover'])
        test_results.append(('hover', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success:
            logger.warning("Level 1 failed - aborting progressive tests (saved ~60s)")
            return False, combined_telemetry

        # Level 2: Small step (REQUIRED)
        logger.info("Running Level 2: Small step test (10°)")
        success, telemetry = executor.run_mission(mission_files['small_step'], timeouts['small_step'])
        test_results.append(('small_step', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success:
            logger.warning("Level 2 failed - aborting progressive tests (saved ~40s)")
            return False, combined_telemetry

        # Level 3: Large step (IMPORTANT)
        logger.info("Running Level 3: Large step test (20°)")
        success, telemetry = executor.run_mission(mission_files['large_step'], timeouts['large_step'])
        test_results.append(('large_step', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success and self.test_sequencer.early_termination:
            logger.warning("Level 3 failed - aborting optional tests (saved ~20s)")
            return False, combined_telemetry

        # Level 4 & 5: Optional advanced tests
        if self.test_sequencer.enable_optional_tests and success:
            logger.info("Running Level 4: Frequency sweep")
            success, telemetry = executor.run_mission(mission_files['frequency'], timeouts['frequency'])
            test_results.append(('frequency', success, telemetry))
            self._append_telemetry(combined_telemetry, telemetry)

            if success:
                logger.info("Running Level 5: Trajectory tracking")
                success, telemetry = executor.run_mission(mission_files['trajectory'], timeouts['trajectory'])
                test_results.append(('trajectory', success, telemetry))
                self._append_telemetry(combined_telemetry, telemetry)

        # Calculate overall success
        overall_success = all(result[1] for result in test_results)
        logger.info(f"Progressive tests complete: {len(test_results)} levels, success={overall_success}")

        return overall_success, combined_telemetry

    def _append_telemetry(self, combined: Dict, new: Dict):
        """Append new telemetry to combined telemetry dict"""
        if not new:
            return

        for key in ['time', 'roll', 'pitch', 'yaw', 'altitude', 'roll_rate', 'pitch_rate', 'yaw_rate']:
            if key in new and isinstance(new[key], (list, np.ndarray)):
                if isinstance(new[key], np.ndarray):
                    combined[key].extend(new[key].tolist())
                else:
                    combined[key].extend(new[key])

    def _update_adaptive_bounds(self, parameters: List[str]):
        """
        Update parameter bounds based on successful parameter statistics

        Uses mean ± 2σ of successful parameters to narrow search space,
        while respecting hard limits.
        """
        logger.info("\n" + "="*60)
        logger.info("UPDATING ADAPTIVE BOUNDS")
        logger.info("="*60)
        logger.info(f"Successful samples: {len(self.successful_params)}")
        logger.info(f"Failed samples: {len(self.failed_params)}")

        bounds_updated = 0

        for param_name in parameters:
            # Extract successful values for this parameter
            successful_values = [p[param_name] for p in self.successful_params]

            if len(successful_values) < self.min_successful_samples:
                continue

            # Calculate statistics
            mean_val = np.mean(successful_values)
            std_val = np.std(successful_values)

            # Get hard bounds
            hard_min, hard_max = self.hard_bounds[param_name]

            # Calculate new bounds: mean ± 2σ (95% confidence interval)
            new_min = max(hard_min, mean_val - 2 * std_val)
            new_max = min(hard_max, mean_val + 2 * std_val)

            # Only update if bounds actually narrow (avoid expansion)
            current_min, current_max = self.current_bounds[param_name]
            if new_min > current_min or new_max < current_max:
                old_range = current_max - current_min
                new_range = new_max - new_min
                reduction = (1 - new_range / old_range) * 100 if old_range > 0 else 0

                self.current_bounds[param_name] = (new_min, new_max)
                bounds_updated += 1

                logger.info(f"  {param_name}:")
                logger.info(f"    Old: [{current_min:.4f}, {current_max:.4f}] (range: {old_range:.4f})")
                logger.info(f"    New: [{new_min:.4f}, {new_max:.4f}] (range: {new_range:.4f})")
                logger.info(f"    Mean ± 2σ: {mean_val:.4f} ± {2*std_val:.4f}")
                logger.info(f"    Reduction: {reduction:.1f}%")

        logger.info(f"\n✓ Updated {bounds_updated}/{len(parameters)} parameter bounds")
        logger.info("="*60 + "\n")

    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if len(self.convergence_history) < 10:
            return False

        # Check if fitness improvement is minimal over last 10 generations
        recent_best = [h['best_fitness'] for h in self.convergence_history[-10:]]
        improvement = recent_best[-1] - recent_best[0]

        if improvement < 0.1:  # Less than 0.1 improvement
            return True

        return False

    def _save_checkpoint(self, filename: str, population: List,
                        generation: int, hall_of_fame):
        """Save optimization checkpoint"""
        checkpoint = {
            'population': population,
            'generation': generation,
            'hall_of_fame': hall_of_fame,
            'convergence_history': self.convergence_history,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state()
        }
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self, filename: str) -> Tuple[List, int, tools.HallOfFame]:
        """Load optimization checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)

        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        self.convergence_history = checkpoint['convergence_history']

        return (checkpoint['population'],
                checkpoint['generation'],
                checkpoint['hall_of_fame'])


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using Optuna"""

    def __init__(self, sitl_manager, evaluator, max_iterations: int = 200,
                 n_startup_trials: int = 20,
                 use_intelligent_sequencing: bool = True):
        """
        Initialize Bayesian optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_iterations: Maximum number of trials
            n_startup_trials: Number of random trials before using TPE
            use_intelligent_sequencing: Whether to use progressive test sequencing
        """
        super().__init__(sitl_manager, evaluator, max_iterations)
        self.n_startup_trials = n_startup_trials
        self.study = None

        # Initialize intelligent test sequencing
        self.use_intelligent_sequencing = use_intelligent_sequencing
        self.test_sequencer = None
        if use_intelligent_sequencing:
            from config import INTELLIGENT_TEST_SEQUENCING_CONFIG
            self.test_sequencer = IntelligentTestSequencer(
                min_pass_score=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('min_pass_score', 60.0),
                enable_optional_tests=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('enable_optional_tests', True),
                early_termination=INTELLIGENT_TEST_SEQUENCING_CONFIG.get('early_termination', True)
            )
            logger.info("Intelligent test sequencing enabled (saves ~40% evaluation time)")
        else:
            logger.info("Using single mission test mode")

    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                resume_from: Optional[str] = None) -> Tuple[Dict, float, List]:
        """Run Bayesian optimization using Optuna"""

        logger.info(f"Starting Bayesian optimization for {phase_name}")
        logger.info(f"Parameters to optimize: {parameters}")
        logger.info(f"Max trials: {self.max_iterations}")

        # Create study
        study_name = f"{phase_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=42
        )

        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler
        )

        # Define objective function
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name in parameters:
                min_val, max_val = bounds[param_name]
                params[param_name] = trial.suggest_float(
                    param_name, min_val, max_val
                )

            # Run simulation
            instance_id = self.sitl_manager.get_instance(timeout=300)
            if instance_id is None:
                logger.error("No SITL instance available")
                return -1000.0

            try:
                success, telemetry = self.sitl_manager.run_simulation(
                    instance_id=instance_id,
                    parameters=params,
                    test_sequence=self._run_test_sequence,
                    duration=30.0
                )

                if success and telemetry:
                    metrics = self.evaluator.evaluate_telemetry(telemetry)
                    fitness = metrics.fitness
                else:
                    fitness = -1000.0

                # Log trial
                logger.info(f"Trial {trial.number}: fitness = {fitness:.4f}")

                # Update convergence history
                self.convergence_history.append({
                    'trial': trial.number,
                    'fitness': fitness,
                    'params': params.copy()
                })

                return fitness

            finally:
                self.sitl_manager.release_instance(instance_id)

        # Run optimization
        logger.info("Starting optimization trials...")

        try:
            self.study.optimize(
                objective,
                n_trials=self.max_iterations,
                callbacks=[self._optuna_callback]
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")

        # Extract best parameters
        best_params = self.study.best_params
        best_fitness = self.study.best_value

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Total trials: {len(self.study.trials)}")

        return best_params, best_fitness, self.convergence_history

    def _run_test_sequence(self, connection, duration: float) -> Tuple[bool, Dict]:
        """
        Run test sequence and collect telemetry

        Uses progressive mission files for intelligent testing when enabled,
        otherwise falls back to single mission test.
        """
        import os
        from mission_executor import MissionExecutor

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Use intelligent test sequencing if enabled
        if self.use_intelligent_sequencing and self.test_sequencer:
            return self._run_progressive_tests(connection, script_dir)
        else:
            # Fallback: single mission test
            from mission_executor import run_mission_test
            mission_file = os.path.join(script_dir, "missions", "simple_hover.waypoints")
            timeout = 120.0
            logger.info(f"Running single mission test: {mission_file}")
            return run_mission_test(connection, mission_file, timeout)

    def _run_progressive_tests(self, connection, script_dir: str) -> Tuple[bool, Dict]:
        """
        Run progressive test sequence using IntelligentTestSequencer

        Args:
            connection: MAVLink connection
            script_dir: Script directory for mission files

        Returns:
            (overall_success, combined_telemetry)
        """
        from mission_executor import MissionExecutor

        executor = MissionExecutor(connection, enable_early_crash_detection=True)

        # Mission files for each test level
        mission_files = {
            'hover': os.path.join(script_dir, "missions", "level1_hover.waypoints"),
            'small_step': os.path.join(script_dir, "missions", "level2_small_step.waypoints"),
            'large_step': os.path.join(script_dir, "missions", "level3_large_step.waypoints"),
            'frequency': os.path.join(script_dir, "missions", "level4_frequency.waypoints"),
            'trajectory': os.path.join(script_dir, "missions", "level5_trajectory.waypoints"),
        }

        # Test timeouts for each level
        timeouts = {
            'hover': 40.0,       # 30s sensor + 10s mission
            'small_step': 50.0,
            'large_step': 60.0,
            'frequency': 80.0,
            'trajectory': 100.0,
        }

        # Run progressive tests
        test_results = []
        combined_telemetry = {'time': [], 'roll': [], 'pitch': [], 'yaw': [],
                            'altitude': [], 'roll_rate': [], 'pitch_rate': [], 'yaw_rate': []}
        overall_success = True

        # Level 1: Hover (REQUIRED)
        logger.info("Running Level 1: Hover test (5s)")
        success, telemetry = executor.run_mission(mission_files['hover'], timeouts['hover'])
        test_results.append(('hover', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success:
            logger.warning("Level 1 failed - aborting progressive tests (saved ~60s)")
            return False, combined_telemetry

        # Level 2: Small step (REQUIRED)
        logger.info("Running Level 2: Small step test (10°)")
        success, telemetry = executor.run_mission(mission_files['small_step'], timeouts['small_step'])
        test_results.append(('small_step', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success:
            logger.warning("Level 2 failed - aborting progressive tests (saved ~40s)")
            return False, combined_telemetry

        # Level 3: Large step (IMPORTANT)
        logger.info("Running Level 3: Large step test (20°)")
        success, telemetry = executor.run_mission(mission_files['large_step'], timeouts['large_step'])
        test_results.append(('large_step', success, telemetry))
        self._append_telemetry(combined_telemetry, telemetry)

        if not success and self.test_sequencer.early_termination:
            logger.warning("Level 3 failed - aborting optional tests (saved ~20s)")
            return False, combined_telemetry

        # Level 4 & 5: Optional advanced tests
        if self.test_sequencer.enable_optional_tests and success:
            logger.info("Running Level 4: Frequency sweep")
            success, telemetry = executor.run_mission(mission_files['frequency'], timeouts['frequency'])
            test_results.append(('frequency', success, telemetry))
            self._append_telemetry(combined_telemetry, telemetry)

            if success:
                logger.info("Running Level 5: Trajectory tracking")
                success, telemetry = executor.run_mission(mission_files['trajectory'], timeouts['trajectory'])
                test_results.append(('trajectory', success, telemetry))
                self._append_telemetry(combined_telemetry, telemetry)

        # Calculate overall success
        overall_success = all(result[1] for result in test_results)
        logger.info(f"Progressive tests complete: {len(test_results)} levels, success={overall_success}")

        return overall_success, combined_telemetry

    def _append_telemetry(self, combined: Dict, new: Dict):
        """Append new telemetry to combined telemetry dict"""
        if not new:
            return

        for key in ['time', 'roll', 'pitch', 'yaw', 'altitude', 'roll_rate', 'pitch_rate', 'yaw_rate']:
            if key in new and isinstance(new[key], (list, np.ndarray)):
                if isinstance(new[key], np.ndarray):
                    combined[key].extend(new[key].tolist())
                else:
                    combined[key].extend(new[key])

    def _optuna_callback(self, study, trial):
        """Callback for Optuna trials"""
        if trial.number % 10 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial.number} completed")
            logger.info(f"Current best fitness: {study.best_value:.4f}")
            logger.info(f"{'='*60}\n")


class HybridOptimizer(BaseOptimizer):
    """Hybrid optimizer combining GA and Bayesian approaches"""

    def __init__(self, sitl_manager, evaluator, max_iterations: int = 150,
                 ga_iterations: int = 50, bayesian_iterations: int = 100):
        """
        Initialize hybrid optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_iterations: Maximum total iterations
            ga_iterations: Number of GA iterations for exploration
            bayesian_iterations: Number of Bayesian iterations for exploitation
        """
        super().__init__(sitl_manager, evaluator, max_iterations)
        self.ga_iterations = ga_iterations
        self.bayesian_iterations = bayesian_iterations

    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                resume_from: Optional[str] = None) -> Tuple[Dict, float, List]:
        """Run hybrid optimization"""

        logger.info(f"Starting Hybrid optimization for {phase_name}")
        logger.info(f"Phase 1: GA exploration ({self.ga_iterations} generations)")
        logger.info(f"Phase 2: Bayesian exploitation ({self.bayesian_iterations} trials)")

        # Phase 1: Genetic Algorithm for exploration
        ga_optimizer = GeneticOptimizer(
            self.sitl_manager,
            self.evaluator,
            max_generations=self.ga_iterations,
            population_size=30
        )

        ga_params, ga_fitness, ga_history = ga_optimizer.optimize(
            phase_name=f"{phase_name}_GA",
            parameters=parameters,
            bounds=bounds
        )

        logger.info(f"\nGA Phase complete - Best fitness: {ga_fitness:.4f}")

        # Phase 2: Bayesian Optimization for exploitation
        # Use GA results to inform initial trials
        bayesian_optimizer = BayesianOptimizer(
            self.sitl_manager,
            self.evaluator,
            max_iterations=self.bayesian_iterations,
            n_startup_trials=10
        )

        bayes_params, bayes_fitness, bayes_history = bayesian_optimizer.optimize(
            phase_name=f"{phase_name}_Bayes",
            parameters=parameters,
            bounds=bounds
        )

        logger.info(f"\nBayesian Phase complete - Best fitness: {bayes_fitness:.4f}")

        # Return best of both
        if bayes_fitness > ga_fitness:
            best_params = bayes_params
            best_fitness = bayes_fitness
        else:
            best_params = ga_params
            best_fitness = ga_fitness

        # Combine convergence histories
        self.convergence_history = ga_history + bayes_history

        logger.info(f"\nHybrid Optimization complete!")
        logger.info(f"Final best fitness: {best_fitness:.4f}")

        return best_params, best_fitness, self.convergence_history
