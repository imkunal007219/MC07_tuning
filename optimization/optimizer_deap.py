#!/usr/bin/env python3
"""
DEAP-based Genetic Algorithm for Drone PID Tuning

Uses Distributed Evolutionary Algorithms in Python (DEAP) library
for robust global optimization with parallel evaluation support.

Advantages:
- Excellent parallelization (processes entire population at once)
- Robust global search (avoids local optima)
- Works well with noisy evaluations
- Easy to understand and debug

Author: MC07 Tuning System
Date: 2025-11-05
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import Dict, List, Callable, Optional, Any, Tuple
import logging
import json
from pathlib import Path
import time
import multiprocessing

from parameter_search_spaces import ParameterSearchSpace
from fitness_evaluator import TestStage

logger = logging.getLogger(__name__)


class DEAPOptimizer:
    """
    Genetic Algorithm optimization using DEAP
    """

    def __init__(self,
                 search_space: Dict[str, Dict[str, Any]],
                 stage: TestStage = TestStage.RATE,
                 population_size: int = 20,
                 seed: Optional[int] = 42):
        """
        Initialize DEAP genetic algorithm optimizer

        Args:
            search_space: Parameter search space dictionary
            stage: Optimization stage
            population_size: Number of individuals in population
            seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.stage = stage
        self.population_size = population_size
        self.param_names = list(search_space.keys())
        self.n_params = len(self.param_names)

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Parameter bounds
        self.bounds_low = [search_space[p]['min'] for p in self.param_names]
        self.bounds_high = [search_space[p]['max'] for p in self.param_names]

        # Create DEAP types (singleton pattern)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize toolbox
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

        # Statistics tracking
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # History
        self.logbook = tools.Logbook()
        self.best_individuals = []

        logger.info(f"DEAPOptimizer initialized for {stage.name}")
        logger.info(f"Population size: {population_size}")
        logger.info(f"Parameters to optimize: {self.n_params}")

    def _setup_toolbox(self):
        """Setup DEAP toolbox with genetic operators"""

        # Attribute generator: random float in parameter bounds
        def random_param(param_idx):
            return random.uniform(self.bounds_low[param_idx], self.bounds_high[param_idx])

        # Individual generator
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                            (lambda i=i: random_param(i) for i in range(self.n_params)),
                            n=1)

        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                            low=self.bounds_low, up=self.bounds_high, eta=20.0)

        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                            low=self.bounds_low, up=self.bounds_high,
                            eta=20.0, indpb=0.2)

        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def individual_to_params(self, individual: List[float]) -> Dict[str, float]:
        """
        Convert DEAP individual to parameter dictionary

        Args:
            individual: DEAP individual (list of floats)

        Returns:
            Parameter dictionary
        """
        return {param_name: individual[i] for i, param_name in enumerate(self.param_names)}

    def params_to_individual(self, params: Dict[str, float]) -> List[float]:
        """
        Convert parameter dictionary to DEAP individual

        Args:
            params: Parameter dictionary

        Returns:
            DEAP individual (list of floats)
        """
        return [params[param_name] for param_name in self.param_names]

    def evaluate_individual(self,
                          individual: List[float],
                          evaluation_function: Callable[[Dict[str, float]], float]) -> Tuple[float,]:
        """
        Evaluate fitness of an individual

        Args:
            individual: DEAP individual
            evaluation_function: Function that evaluates parameter fitness

        Returns:
            Tuple containing fitness score (DEAP convention)
        """
        params = self.individual_to_params(individual)

        try:
            fitness = evaluation_function(params)
            return (fitness,)  # DEAP requires tuple

        except Exception as e:
            logger.error(f"Individual evaluation failed: {e}")
            return (0.0,)  # Worst fitness on failure

    def optimize(self,
                evaluation_function: Callable[[Dict[str, float]], float],
                n_generations: int = 30,
                cx_prob: float = 0.7,
                mut_prob: float = 0.2,
                n_jobs: int = 1,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization

        Args:
            evaluation_function: Function that takes parameters and returns fitness
            n_generations: Number of generations to evolve
            cx_prob: Crossover probability (0.7 = 70% chance)
            mut_prob: Mutation probability (0.2 = 20% chance)
            n_jobs: Number of parallel jobs (1 = sequential, -1 = all cores)
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting DEAP genetic algorithm: {n_generations} generations")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Crossover prob: {cx_prob}, Mutation prob: {mut_prob}")

        start_time = time.time()

        # Setup parallel evaluation if requested
        if n_jobs != 1:
            pool = multiprocessing.Pool(processes=n_jobs if n_jobs > 0 else None)
            self.toolbox.register("map", pool.map)

        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual,
                            evaluation_function=evaluation_function)

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        logger.info("Evaluating initial population...")
        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Record statistics
        record = self.stats.compile(population)
        self.logbook.record(gen=0, **record)

        if verbose:
            print(f"Gen 0: Best={record['max']:.2f}, Avg={record['avg']:.2f}, "
                  f"Std={record['std']:.2f}")

        # Evolution loop
        for gen in range(1, n_generations + 1):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Record statistics
            record = self.stats.compile(population)
            self.logbook.record(gen=gen, **record)

            if verbose:
                print(f"Gen {gen}: Best={record['max']:.2f}, Avg={record['avg']:.2f}, "
                      f"Std={record['std']:.2f}")

            # Track best individual
            best_ind = tools.selBest(population, 1)[0]
            self.best_individuals.append(best_ind)

        # Cleanup parallel processing
        if n_jobs != 1:
            pool.close()
            pool.join()

        elapsed_time = time.time() - start_time

        # Get final best solution
        best_individual = tools.selBest(population, 1)[0]
        best_params = self.individual_to_params(best_individual)
        best_fitness = best_individual.fitness.values[0]

        # Calculate total evaluations
        total_evaluations = self.population_size * (n_generations + 1)

        results = {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'n_generations': n_generations,
            'total_evaluations': total_evaluations,
            'elapsed_time': elapsed_time,
            'stage': self.stage.name,
            'final_population_stats': record,
        }

        logger.info(f"Optimization complete in {elapsed_time:.1f}s")
        logger.info(f"Total evaluations: {total_evaluations}")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return results

    def save_results(self, output_path: str):
        """
        Save optimization results to file

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get best individual
        best_ind = tools.selBest([ind for ind in self.best_individuals], 1)[0]
        best_params = self.individual_to_params(best_ind)

        # Compile results
        results = {
            'stage': self.stage.name,
            'population_size': self.population_size,
            'n_generations': len(self.logbook) - 1,
            'best_fitness': best_ind.fitness.values[0],
            'best_parameters': best_params,
            'evolution_history': [
                {
                    'generation': record['gen'],
                    'max': record['max'],
                    'avg': record['avg'],
                    'min': record['min'],
                    'std': record['std'],
                }
                for record in self.logbook
            ],
        }

        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Also save best parameters as .parm file
        parm_path = output_path.with_suffix('.parm')
        self._save_parameter_file(parm_path, best_params, best_ind.fitness.values[0])

    def _save_parameter_file(self, file_path: Path, parameters: Dict[str, float], fitness: float):
        """
        Save parameters in ArduPilot .parm format

        Args:
            file_path: Output file path
            parameters: Parameter dictionary
            fitness: Fitness score
        """
        with open(file_path, 'w') as f:
            f.write(f"# Optimized parameters - {self.stage.name}\n")
            f.write(f"# Genetic Algorithm (DEAP)\n")
            f.write(f"# Best fitness: {fitness:.4f}\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for param_name, value in parameters.items():
                f.write(f"{param_name}    {value}\n")

        logger.info(f"Parameter file saved to {file_path}")

    def plot_evolution(self, output_path: Optional[str] = None):
        """
        Plot evolution history

        Args:
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            gen = self.logbook.select("gen")
            max_fit = self.logbook.select("max")
            avg_fit = self.logbook.select("avg")
            min_fit = self.logbook.select("min")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(gen, max_fit, 'g-', label='Best', linewidth=2)
            ax.plot(gen, avg_fit, 'b-', label='Average')
            ax.plot(gen, min_fit, 'r-', label='Worst')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Evolution History - {self.stage.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Evolution plot saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available - cannot plot results")


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("DEAP GENETIC ALGORITHM OPTIMIZER DEMONSTRATION")
    print("="*70 + "\n")

    # Setup search space
    search_space_manager = ParameterSearchSpace()
    search_space = search_space_manager.get_stage_1_rate_controller_space()

    print(f"Optimizing {len(search_space)} parameters for RATE controller\n")

    # Mock evaluation function
    def mock_evaluation(params: Dict[str, float]) -> float:
        """Mock fitness function for demonstration"""
        time.sleep(0.05)  # Simulate test duration

        # Arbitrary fitness calculation
        fitness = 50.0
        fitness += (params['ATC_RAT_RLL_P'] - 0.12) ** 2 * -100
        fitness += (params['ATC_RAT_PIT_P'] - 0.10) ** 2 * -100
        fitness += random.gauss(0, 3)

        return max(0, min(100, fitness))

    # Run optimization
    optimizer = DEAPOptimizer(
        search_space=search_space,
        stage=TestStage.RATE,
        population_size=10  # Small for demo
    )

    results = optimizer.optimize(
        evaluation_function=mock_evaluation,
        n_generations=5,  # Few generations for demo
        n_jobs=1
    )

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Generations: {results['n_generations']}")
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Time elapsed: {results['elapsed_time']:.1f}s")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"\nBest parameters:")
    for param_name, value in results['best_parameters'].items():
        print(f"  {param_name}: {value:.6f}")

    # Save results
    optimizer.save_results("results/deap_demo_results.json")

    print("\n" + "="*70 + "\n")
