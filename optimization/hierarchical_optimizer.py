#!/usr/bin/env python3
"""
Hierarchical Optimization Framework

Orchestrates the complete 3-stage optimization process:
Stage 1: Rate Controllers (Inner Loop)
Stage 2: Attitude Controllers (Middle Loop)
Stage 3: Position Controllers (Outer Loop)

Each stage builds upon the optimized parameters from previous stages.

Author: MC07 Tuning System
Date: 2025-11-05
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum

from parameter_search_spaces import ParameterSearchSpace
from fitness_evaluator import FitnessEvaluator, TestStage
from parallel_sitl_manager import ParallelSITLManager
from optimizer_optuna import OptunaOptimizer
from optimizer_deap import DEAPOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Available optimizer algorithms"""
    OPTUNA = "optuna"   # Bayesian Optimization
    DEAP = "deap"       # Genetic Algorithm


class HierarchicalOptimizer:
    """
    Manages hierarchical PID optimization across 3 stages
    """

    def __init__(self,
                 ardupilot_root: str,
                 optimizer_type: OptimizerType = OptimizerType.OPTUNA,
                 num_parallel_instances: int = 4,
                 output_dir: str = "optimization_results"):
        """
        Initialize hierarchical optimizer

        Args:
            ardupilot_root: Path to ArduPilot root directory
            optimizer_type: Which optimization algorithm to use
            num_parallel_instances: Number of parallel SITL instances
            output_dir: Directory to save results
        """
        self.ardupilot_root = Path(ardupilot_root)
        self.optimizer_type = optimizer_type
        self.num_parallel = num_parallel_instances
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.search_space = ParameterSearchSpace()

        # Storage for optimized parameters from each stage
        self.optimized_params = {
            TestStage.RATE: {},
            TestStage.ATTITUDE: {},
            TestStage.POSITION: {},
        }

        logger.info(f"HierarchicalOptimizer initialized")
        logger.info(f"Optimizer: {optimizer_type.value}")
        logger.info(f"Parallel instances: {num_parallel}")
        logger.info(f"Output directory: {output_dir}")

    def optimize_stage(self,
                      stage: TestStage,
                      n_trials: int = 50,
                      population_size: int = 20,
                      n_generations: int = 30) -> Dict[str, float]:
        """
        Optimize a single stage

        Args:
            stage: Which stage to optimize
            n_trials: Number of trials (for Optuna)
            population_size: Population size (for DEAP)
            n_generations: Number of generations (for DEAP)

        Returns:
            Dictionary of optimized parameters
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"OPTIMIZING STAGE {stage.value}: {stage.name}")
        logger.info(f"{'='*70}\n")

        # Get search space for this stage
        if stage == TestStage.RATE:
            search_space = self.search_space.get_stage_1_rate_controller_space()
        elif stage == TestStage.ATTITUDE:
            search_space = self.search_space.get_stage_2_attitude_controller_space()
        else:
            search_space = self.search_space.get_stage_3_position_controller_space()

        logger.info(f"Optimizing {len(search_space)} parameters")

        # Create fitness evaluator for this stage
        fitness_evaluator = FitnessEvaluator(stage=stage)

        # Define evaluation function
        def evaluate_parameters(params: Dict[str, float]) -> float:
            """
            Evaluate a parameter set by running SITL simulation

            Args:
                params: Parameter dictionary

            Returns:
                Fitness score
            """
            # TODO: Implement actual SITL evaluation
            # For now, return mock fitness
            logger.info(f"Evaluating parameter set...")
            time.sleep(0.1)  # Simulate test duration
            return 75.0  # Mock fitness

        # Run optimization with selected algorithm
        if self.optimizer_type == OptimizerType.OPTUNA:
            optimizer = OptunaOptimizer(search_space, stage)
            results = optimizer.optimize(
                evaluation_function=evaluate_parameters,
                n_trials=n_trials,
                n_jobs=1  # Parallel not yet implemented for SITL
            )
            # Save results
            output_file = self.output_dir / f"stage{stage.value}_{self.optimizer_type.value}_results.json"
            optimizer.save_results(str(output_file))

        else:  # DEAP
            optimizer = DEAPOptimizer(search_space, stage, population_size)
            results = optimizer.optimize(
                evaluation_function=evaluate_parameters,
                n_generations=n_generations,
                n_jobs=1
            )
            # Save results
            output_file = self.output_dir / f"stage{stage.value}_{self.optimizer_type.value}_results.json"
            optimizer.save_results(str(output_file))

        # Store optimized parameters
        self.optimized_params[stage] = results['best_parameters']

        logger.info(f"\nStage {stage.value} complete!")
        logger.info(f"Best fitness: {results['best_fitness']:.4f}")

        return results['best_parameters']

    def run_full_optimization(self,
                            n_trials_per_stage: Dict[TestStage, int] = None,
                            skip_stages: Optional[list] = None) -> Dict[str, Any]:
        """
        Run complete hierarchical optimization (all 3 stages)

        Args:
            n_trials_per_stage: Optional custom number of trials per stage
            skip_stages: Optional list of stages to skip

        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING HIERARCHICAL OPTIMIZATION")
        logger.info("="*70 + "\n")

        start_time = time.time()
        skip_stages = skip_stages or []

        # Default number of trials
        if n_trials_per_stage is None:
            if self.optimizer_type == OptimizerType.OPTUNA:
                n_trials_per_stage = {
                    TestStage.RATE: 30,
                    TestStage.ATTITUDE: 20,
                    TestStage.POSITION: 20,
                }
            else:  # DEAP
                n_trials_per_stage = {
                    TestStage.RATE: 30,  # Generations
                    TestStage.ATTITUDE: 20,
                    TestStage.POSITION: 20,
                }

        # Stage 1: Rate Controllers
        if TestStage.RATE not in skip_stages:
            self.optimize_stage(TestStage.RATE, n_trials=n_trials_per_stage[TestStage.RATE])
        else:
            logger.info("Skipping Stage 1 (RATE)")

        # Stage 2: Attitude Controllers (uses Stage 1 results as baseline)
        if TestStage.ATTITUDE not in skip_stages:
            self.optimize_stage(TestStage.ATTITUDE, n_trials=n_trials_per_stage[TestStage.ATTITUDE])
        else:
            logger.info("Skipping Stage 2 (ATTITUDE)")

        # Stage 3: Position Controllers (uses Stage 1 & 2 results as baseline)
        if TestStage.POSITION not in skip_stages:
            self.optimize_stage(TestStage.POSITION, n_trials=n_trials_per_stage[TestStage.POSITION])
        else:
            logger.info("Skipping Stage 3 (POSITION)")

        elapsed_time = time.time() - start_time

        # Combine all optimized parameters
        final_params = {}
        for stage in [TestStage.RATE, TestStage.ATTITUDE, TestStage.POSITION]:
            final_params.update(self.optimized_params[stage])

        # Save final combined parameters
        self._save_final_parameters(final_params, elapsed_time)

        logger.info("\n" + "="*70)
        logger.info("HIERARCHICAL OPTIMIZATION COMPLETE!")
        logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"Total parameters optimized: {len(final_params)}")
        logger.info("="*70 + "\n")

        return {
            'final_parameters': final_params,
            'elapsed_time': elapsed_time,
            'stage_results': self.optimized_params,
            'optimizer_type': self.optimizer_type.value,
        }

    def _save_final_parameters(self, parameters: Dict[str, float], elapsed_time: float):
        """Save final optimized parameters"""
        # Save as .parm file
        parm_file = self.output_dir / "final_optimized_parameters.parm"
        with open(parm_file, 'w') as f:
            f.write("# Final Optimized Parameters - 30kg Drone\n")
            f.write(f"# Hierarchical Optimization ({self.optimizer_type.value})\n")
            f.write(f"# Optimization time: {elapsed_time/60:.1f} minutes\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Group by stage
            rate_params = self.optimized_params[TestStage.RATE]
            attitude_params = self.optimized_params[TestStage.ATTITUDE]
            position_params = self.optimized_params[TestStage.POSITION]

            f.write("# ========================================\n")
            f.write("# STAGE 1: RATE CONTROLLERS\n")
            f.write("# ========================================\n")
            for param, value in rate_params.items():
                f.write(f"{param}    {value}\n")

            f.write("\n# ========================================\n")
            f.write("# STAGE 2: ATTITUDE CONTROLLERS\n")
            f.write("# ========================================\n")
            for param, value in attitude_params.items():
                f.write(f"{param}    {value}\n")

            f.write("\n# ========================================\n")
            f.write("# STAGE 3: POSITION CONTROLLERS\n")
            f.write("# ========================================\n")
            for param, value in position_params.items():
                f.write(f"{param}    {value}\n")

        logger.info(f"Final parameters saved to {parm_file}")

        # Save as JSON
        json_file = self.output_dir / "final_results.json"
        with open(json_file, 'w') as f:
            json.dump({
                'parameters': parameters,
                'elapsed_time': elapsed_time,
                'optimizer': self.optimizer_type.value,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)

        logger.info(f"Results saved to {json_file}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HIERARCHICAL OPTIMIZATION FRAMEWORK DEMONSTRATION")
    print("="*70 + "\n")

    # Configuration
    config = {
        'ardupilot_root': "/home/user/MC07_tuning/ardupilot",
        'optimizer_type': OptimizerType.OPTUNA,
        'num_parallel_instances': 4,
        'output_dir': "results/hierarchical_demo"
    }

    # Initialize optimizer
    hierarchical_opt = HierarchicalOptimizer(**config)

    # Run single stage for demo (Stage 1 only)
    print("Running Stage 1 optimization (demo with few trials)...\n")
    stage1_results = hierarchical_opt.optimize_stage(
        TestStage.RATE,
        n_trials=5  # Small number for demo
    )

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print(f"\nOptimized {len(stage1_results)} parameters for Stage 1")
    print(f"Results saved to: {hierarchical_opt.output_dir}")
    print("\n" + "="*70 + "\n")
