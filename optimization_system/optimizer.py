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


logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base class for optimization algorithms"""

    def __init__(self, sitl_manager, evaluator, max_iterations: int = 100):
        """
        Initialize optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_iterations: Maximum optimization iterations
        """
        self.sitl_manager = sitl_manager
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.best_params = None
        self.best_fitness = -np.inf
        self.convergence_history = []

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
                 crossover_rate: float = 0.7):
        """
        Initialize Genetic Algorithm optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_generations: Maximum number of generations
            population_size: Population size
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
        """
        super().__init__(sitl_manager, evaluator, max_generations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.toolbox = None
        self.checkpoint_freq = 10  # Save checkpoint every N generations

    def optimize(self, phase_name: str, parameters: List[str],
                bounds: Dict[str, Tuple[float, float]],
                resume_from: Optional[str] = None) -> Tuple[Dict, float, List]:
        """Run genetic algorithm optimization"""

        logger.info(f"Starting GA optimization for {phase_name}")
        logger.info(f"Parameters to optimize: {parameters}")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Max generations: {self.max_iterations}")

        # Setup DEAP
        self._setup_deap(parameters, bounds)

        # Initialize or resume population
        if resume_from:
            pop, gen_start, hof = self._load_checkpoint(resume_from)
            logger.info(f"Resumed from generation {gen_start}")
        else:
            pop = self.toolbox.population(n=self.population_size)
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

            # Evaluate population
            fitnesses = self._evaluate_population(pop, parameters, bounds)

            # Assign fitness to individuals
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = (fit,)

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

            # Check convergence
            if self._check_convergence():
                logger.info("Convergence criteria met - stopping early")
                break

            # Save checkpoint periodically
            if (gen + 1) % self.checkpoint_freq == 0:
                checkpoint_file = f"/tmp/ga_checkpoint_gen{gen+1}.pkl"
                self._save_checkpoint(checkpoint_file, pop, gen + 1, hof)
                logger.info(f"Checkpoint saved: {checkpoint_file}")

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
                            bounds: Dict[str, Tuple[float, float]]) -> List[float]:
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

        # Calculate fitness for each result
        fitnesses = []
        for success, telemetry in results:
            if success and telemetry:
                metrics = self.evaluator.evaluate_telemetry(telemetry)
                fitnesses.append(metrics.fitness)
            else:
                # Failed simulation - very poor fitness
                fitnesses.append(-1000.0)

        logger.info(f"Evaluated {len(fitnesses)} individuals")
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

        This is a placeholder - actual implementation in test_sequences.py
        """
        # Import here to avoid circular dependency
        from test_sequences import HoverStabilityTest

        test = HoverStabilityTest(connection, duration=duration)
        return test.run()

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
                 n_startup_trials: int = 20):
        """
        Initialize Bayesian optimizer

        Args:
            sitl_manager: SITL manager instance
            evaluator: Performance evaluator instance
            max_iterations: Maximum number of trials
            n_startup_trials: Number of random trials before using TPE
        """
        super().__init__(sitl_manager, evaluator, max_iterations)
        self.n_startup_trials = n_startup_trials
        self.study = None

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

        This is a placeholder - actual implementation in test_sequences.py
        """
        # Import here to avoid circular dependency
        from test_sequences import HoverStabilityTest

        test = HoverStabilityTest(connection, duration=duration)
        return test.run()

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
