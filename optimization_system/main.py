"""
Main entry point for automated drone tuning system
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from sitl_manager import SITLManager
from optimizer import GeneticOptimizer, BayesianOptimizer
from performance_evaluator import PerformanceEvaluator
from utils import setup_logging, save_results


def main():
    """Main optimization loop"""
    parser = argparse.ArgumentParser(description='Automated Drone Tuning System')
    parser.add_argument('--phase', type=str, default='phase1_rate',
                        choices=['phase1_rate', 'phase2_attitude', 'phase3_position', 'phase4_advanced', 'all'],
                        help='Optimization phase to run')
    parser.add_argument('--algorithm', type=str, default='genetic',
                        choices=['genetic', 'bayesian'],
                        help='Optimization algorithm')
    parser.add_argument('--generations', type=int, default=100,
                        help='Maximum number of generations')
    parser.add_argument('--parallel', type=int, default=10,
                        help='Number of parallel SITL instances')
    parser.add_argument('--speedup', type=int, default=1,
                        help='SITL speedup factor')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{LOGGING_CONFIG['log_dir']}/optimization_{timestamp}.log"
    logger = setup_logging(log_file, LOGGING_CONFIG['log_level'])

    logger.info("="*80)
    logger.info("AUTOMATED DRONE TUNING SYSTEM STARTED")
    logger.info("="*80)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Max Generations: {args.generations}")
    logger.info(f"Parallel Instances: {args.parallel}")
    logger.info(f"SITL Speedup: {args.speedup}")

    # Initialize components
    logger.info("\nInitializing components...")

    sitl_manager = SITLManager(
        num_instances=args.parallel,
        speedup=args.speedup
    )

    performance_evaluator = PerformanceEvaluator()

    # Select optimizer
    if args.algorithm == 'genetic':
        optimizer = GeneticOptimizer(
            sitl_manager=sitl_manager,
            evaluator=performance_evaluator,
            max_generations=args.generations,
            population_size=OPTIMIZATION_CONFIG['population_size']
        )
    else:
        optimizer = BayesianOptimizer(
            sitl_manager=sitl_manager,
            evaluator=performance_evaluator,
            max_iterations=args.generations
        )

    # Determine phases to run
    if args.phase == 'all':
        phases = ['phase1_rate', 'phase2_attitude', 'phase3_position', 'phase4_advanced']
    else:
        phases = [args.phase]

    # Run optimization for each phase
    all_results = {}

    for phase_name in phases:
        logger.info("\n" + "="*80)
        logger.info(f"STARTING PHASE: {OPTIMIZATION_PHASES[phase_name]['name']}")
        logger.info("="*80)

        phase_config = OPTIMIZATION_PHASES[phase_name]

        # Run optimization
        best_params, best_fitness, convergence_history = optimizer.optimize(
            phase_name=phase_name,
            parameters=phase_config['parameters'],
            bounds=phase_config['bounds'],
            resume_from=args.resume
        )

        # Store results
        all_results[phase_name] = {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'convergence_history': convergence_history,
        }

        # Save intermediate results
        checkpoint_file = f"{LOGGING_CONFIG['log_dir']}/checkpoint_{phase_name}_{timestamp}.pkl"
        save_results(checkpoint_file, all_results[phase_name])
        logger.info(f"\nCheckpoint saved to: {checkpoint_file}")

        logger.info(f"\nPhase {phase_name} completed!")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best parameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value:.6f}")

    # Final validation
    logger.info("\n" + "="*80)
    logger.info("FINAL VALIDATION")
    logger.info("="*80)

    # Combine all optimized parameters
    final_params = {}
    for phase_name in phases:
        final_params.update(all_results[phase_name]['best_params'])

    # Run comprehensive validation
    logger.info("\nRunning comprehensive validation tests...")
    validation_results = performance_evaluator.validate_parameters(
        sitl_manager=sitl_manager,
        parameters=final_params
    )

    # Save final results
    final_output = {
        'optimized_parameters': final_params,
        'phase_results': all_results,
        'validation_results': validation_results,
        'timestamp': timestamp,
    }

    output_file = f"{LOGGING_CONFIG['log_dir']}/final_results_{timestamp}.pkl"
    save_results(output_file, final_output)

    # Generate parameter file
    param_file = f"{LOGGING_CONFIG['log_dir']}/optimized_params_{timestamp}.param"
    generate_param_file(final_params, param_file)

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nFinal parameter file: {param_file}")
    logger.info(f"Results file: {output_file}")
    logger.info(f"\nValidation Summary:")
    logger.info(f"  Safety checks passed: {validation_results['safety_passed']}")
    logger.info(f"  Performance score: {validation_results['performance_score']:.4f}")

    # Cleanup
    sitl_manager.cleanup()

    return 0


def generate_param_file(parameters, output_file):
    """Generate ArduPilot parameter file"""
    with open(output_file, 'w') as f:
        f.write("# Optimized parameters for 30kg quadcopter\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n\n")

        # Group by category
        for phase_name, phase_config in OPTIMIZATION_PHASES.items():
            f.write(f"# {phase_config['name']}\n")
            for param in phase_config['parameters']:
                if param in parameters:
                    f.write(f"{param},{parameters[param]:.6f}\n")
            f.write("\n")


if __name__ == '__main__':
    sys.exit(main())
