#!/usr/bin/env python3
"""
Optuna-based Bayesian Optimization for Drone PID Tuning

Uses Optuna's Tree-structured Parzen Estimator (TPE) algorithm
for sample-efficient optimization of expensive simulation evaluations.

Advantages:
- Sample efficient (fewer iterations needed)
- Handles high-dimensional search spaces well
- Provides uncertainty estimates
- Good for expensive evaluations (SITL simulations)

Author: MC07 Tuning System
Date: 2025-11-05
"""

import optuna
from optuna.trial import Trial
from typing import Dict, List, Callable, Optional, Any
import logging
import json
from pathlib import Path
import time

from parameter_search_spaces import ParameterSearchSpace
from fitness_evaluator import TestStage

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Bayesian optimization using Optuna
    """

    def __init__(self,
                 search_space: Dict[str, Dict[str, Any]],
                 stage: TestStage = TestStage.RATE,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None):
        """
        Initialize Optuna optimizer

        Args:
            search_space: Parameter search space dictionary
            stage: Optimization stage
            study_name: Optional study name for saving/loading
            storage: Optional database URL for persistent storage
        """
        self.search_space = search_space
        self.stage = stage
        self.study_name = study_name or f"drone_tuning_{stage.name.lower()}_{int(time.time())}"
        self.storage = storage

        # Create Optuna study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction='maximize',  # Maximize fitness
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,  # Random trials before TPE kicks in
                n_ei_candidates=24,    # Number of candidates for expected improvement
                multivariate=True,     # Consider parameter interactions
                seed=42                # Reproducibility
            ),
            load_if_exists=True  # Resume if study exists
        )

        logger.info(f"OptunaOptimizer initialized for {stage.name}")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Parameters to optimize: {len(search_space)}")

    def suggest_parameters(self, trial: Trial) -> Dict[str, float]:
        """
        Suggest parameter values for a trial

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for param_name, param_info in self.search_space.items():
            if param_info['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_info['min'],
                    param_info['max'],
                    log=False  # Linear scale (could use log=True for wide ranges)
                )
            elif param_info['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    int(param_info['min']),
                    int(param_info['max'])
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_info['type']}")

        return params

    def objective_function(self,
                          trial: Trial,
                          evaluation_function: Callable[[Dict[str, float]], float]) -> float:
        """
        Objective function for Optuna to optimize

        Args:
            trial: Optuna trial object
            evaluation_function: Function that evaluates parameter fitness

        Returns:
            Fitness score
        """
        # Suggest parameters for this trial
        params = self.suggest_parameters(trial)

        # Log trial info
        logger.info(f"Trial {trial.number}: Testing parameter set")
        logger.debug(f"Parameters: {params}")

        # Evaluate fitness
        try:
            fitness = evaluation_function(params)
            logger.info(f"Trial {trial.number}: Fitness = {fitness:.4f}")

            # Store parameters as user attributes for later analysis
            for param_name, param_value in params.items():
                trial.set_user_attr(param_name, param_value)

            return fitness

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return worst possible fitness on failure
            return 0.0

    def optimize(self,
                evaluation_function: Callable[[Dict[str, float]], float],
                n_trials: int = 50,
                timeout: Optional[int] = None,
                n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run Bayesian optimization

        Args:
            evaluation_function: Function that takes parameters and returns fitness
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            n_jobs: Number of parallel jobs (1 = sequential)

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Optuna optimization: {n_trials} trials")
        logger.info(f"Parallel jobs: {n_jobs}")

        start_time = time.time()

        # Run optimization
        self.study.optimize(
            lambda trial: self.objective_function(trial, evaluation_function),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        elapsed_time = time.time() - start_time

        # Get results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_fitness = best_trial.value

        results = {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'n_trials': len(self.study.trials),
            'elapsed_time': elapsed_time,
            'study_name': self.study_name,
            'stage': self.stage.name,
        }

        logger.info(f"Optimization complete in {elapsed_time:.1f}s")
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

        # Compile results
        results = {
            'study_name': self.study_name,
            'stage': self.stage.name,
            'n_trials': len(self.study.trials),
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params,
            },
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                }
                for trial in self.study.trials
            ],
        }

        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Also save best parameters as .parm file
        parm_path = output_path.with_suffix('.parm')
        self._save_parameter_file(parm_path, self.study.best_trial.params)

    def _save_parameter_file(self, file_path: Path, parameters: Dict[str, float]):
        """
        Save parameters in ArduPilot .parm format

        Args:
            file_path: Output file path
            parameters: Parameter dictionary
        """
        with open(file_path, 'w') as f:
            f.write(f"# Optimized parameters - {self.stage.name}\n")
            f.write(f"# Study: {self.study_name}\n")
            f.write(f"# Best fitness: {self.study.best_trial.value:.4f}\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for param_name, value in parameters.items():
                f.write(f"{param_name}    {value}\n")

        logger.info(f"Parameter file saved to {file_path}")

    def plot_optimization_history(self, output_path: Optional[str] = None):
        """
        Plot optimization history

        Args:
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Optimization history plot saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available - cannot plot results")

    def plot_parameter_importances(self, output_path: Optional[str] = None):
        """
        Plot parameter importance rankings

        Args:
            output_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Parameter importance plot saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib not available - cannot plot results")


if __name__ == "__main__":
    # Demonstration
    import random

    print("\n" + "="*70)
    print("OPTUNA BAYESIAN OPTIMIZER DEMONSTRATION")
    print("="*70 + "\n")

    # Setup search space
    search_space_manager = ParameterSearchSpace()
    search_space = search_space_manager.get_stage_1_rate_controller_space()

    print(f"Optimizing {len(search_space)} parameters for RATE controller\n")

    # Mock evaluation function (in real use, this would run SITL)
    def mock_evaluation(params: Dict[str, float]) -> float:
        """
        Mock fitness function for demonstration
        In reality, this would run SITL simulation
        """
        # Simulate noisy fitness based on parameter values
        # Real function would run SITL test and return fitness
        time.sleep(0.1)  # Simulate test duration

        # Arbitrary fitness calculation for demo
        fitness = 50.0
        fitness += (params['ATC_RAT_RLL_P'] - 0.12) ** 2 * -100  # Prefer P=0.12
        fitness += (params['ATC_RAT_PIT_P'] - 0.10) ** 2 * -100  # Prefer P=0.10
        fitness += random.gauss(0, 5)  # Add noise

        return max(0, min(100, fitness))  # Clamp to [0, 100]

    # Run optimization
    optimizer = OptunaOptimizer(
        search_space=search_space,
        stage=TestStage.RATE
    )

    results = optimizer.optimize(
        evaluation_function=mock_evaluation,
        n_trials=20,  # Small number for demo
        n_jobs=1
    )

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Study name: {results['study_name']}")
    print(f"Trials completed: {results['n_trials']}")
    print(f"Time elapsed: {results['elapsed_time']:.1f}s")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"\nBest parameters:")
    for param_name, value in results['best_parameters'].items():
        print(f"  {param_name}: {value:.6f}")

    # Save results
    optimizer.save_results("results/optuna_demo_results.json")

    print("\n" + "="*70 + "\n")
