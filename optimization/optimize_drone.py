#!/usr/bin/env python3
"""
Main CLI script for automated drone PID optimization

Usage examples:
    # Run full hierarchical optimization with Optuna (Bayesian)
    python optimize_drone.py --optimizer optuna --trials 30 --all-stages

    # Run full hierarchical optimization with DEAP (Genetic Algorithm)
    python optimize_drone.py --optimizer deap --generations 30 --all-stages

    # Run only Stage 1 (Rate Controllers) with custom trials
    python optimize_drone.py --optimizer optuna --stage 1 --trials 50

    # Run with 4 parallel SITL instances
    python optimize_drone.py --parallel 4 --all-stages

Author: MC07 Tuning System
Date: 2025-11-05
"""

import argparse
import sys
import logging
from pathlib import Path

from hierarchical_optimizer import HierarchicalOptimizer, OptimizerType
from fitness_evaluator import TestStage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automated PID Optimization for 30kg Drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization with Bayesian Optimization (Optuna)
  %(prog)s --optimizer optuna --trials 30 --all-stages

  # Full optimization with Genetic Algorithm (DEAP)
  %(prog)s --optimizer deap --generations 30 --all-stages

  # Single stage optimization
  %(prog)s --optimizer optuna --stage 1 --trials 50

  # With parallel SITL instances
  %(prog)s --optimizer deap --parallel 4 --all-stages
        """
    )

    # Required arguments
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['optuna', 'deap'],
        default='optuna',
        help='Optimization algorithm (default: optuna)'
    )

    # Stage selection
    stage_group = parser.add_mutually_exclusive_group(required=True)
    stage_group.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3],
        help='Optimize specific stage only (1=Rate, 2=Attitude, 3=Position)'
    )
    stage_group.add_argument(
        '--all-stages',
        action='store_true',
        help='Run complete hierarchical optimization (all 3 stages)'
    )

    # Optimizer-specific parameters
    optuna_group = parser.add_argument_group('Optuna (Bayesian) options')
    optuna_group.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of trials for Optuna (default: 30)'
    )

    deap_group = parser.add_argument_group('DEAP (Genetic Algorithm) options')
    deap_group.add_argument(
        '--generations',
        type=int,
        default=30,
        help='Number of generations for DEAP (default: 30)'
    )
    deap_group.add_argument(
        '--population',
        type=int,
        default=20,
        help='Population size for DEAP (default: 20)'
    )

    # System configuration
    parser.add_argument(
        '--parallel',
        type=int,
        default=4,
        help='Number of parallel SITL instances (default: 4, max depends on CPU cores)'
    )
    parser.add_argument(
        '--ardupilot-root',
        type=str,
        default='/home/user/MC07_tuning/ardupilot',
        help='Path to ArduPilot root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optimization_results',
        help='Output directory for results (default: optimization_results)'
    )

    # Debugging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print("\n" + "="*70)
    print("AUTOMATED DRONE PID OPTIMIZATION SYSTEM")
    print("30kg Heavy-lift Quadcopter - MELARD 1026 Motors")
    print("="*70 + "\n")

    # Print configuration
    optimizer_name = "Optuna (Bayesian Optimization)" if args.optimizer == "optuna" else "DEAP (Genetic Algorithm)"
    print(f"Configuration:")
    print(f"  Optimizer: {optimizer_name}")

    if args.all_stages:
        print(f"  Mode: Full hierarchical optimization (3 stages)")
    else:
        stage_names = {1: "Rate Controllers", 2: "Attitude Controllers", 3: "Position Controllers"}
        print(f"  Mode: Single stage - Stage {args.stage} ({stage_names[args.stage]})")

    if args.optimizer == "optuna":
        print(f"  Trials: {args.trials}")
    else:
        print(f"  Generations: {args.generations}")
        print(f"  Population: {args.population}")

    print(f"  Parallel instances: {args.parallel}")
    print(f"  ArduPilot root: {args.ardupilot_root}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Dry run check
    if args.dry_run:
        print("Dry run mode - exiting without optimization")
        return 0

    # Verify ArduPilot directory exists
    ardupilot_path = Path(args.ardupilot_root)
    if not ardupilot_path.exists():
        logger.error(f"ArduPilot directory not found: {args.ardupilot_root}")
        logger.error("Please specify correct path with --ardupilot-root")
        return 1

    # Initialize hierarchical optimizer
    logger.info("Initializing optimizer...")
    optimizer_type = OptimizerType.OPTUNA if args.optimizer == "optuna" else OptimizerType.DEAP

    try:
        hierarchical_opt = HierarchicalOptimizer(
            ardupilot_root=str(ardupilot_path),
            optimizer_type=optimizer_type,
            num_parallel_instances=args.parallel,
            output_dir=args.output_dir
        )

        # Run optimization
        if args.all_stages:
            # Full hierarchical optimization
            logger.info("Starting full hierarchical optimization...")

            # Prepare trial/generation counts per stage
            if args.optimizer == "optuna":
                n_trials_per_stage = {
                    TestStage.RATE: args.trials,
                    TestStage.ATTITUDE: max(20, args.trials // 2),
                    TestStage.POSITION: max(20, args.trials // 2),
                }
            else:
                n_trials_per_stage = {
                    TestStage.RATE: args.generations,
                    TestStage.ATTITUDE: max(20, args.generations // 2),
                    TestStage.POSITION: max(20, args.generations // 2),
                }

            results = hierarchical_opt.run_full_optimization(
                n_trials_per_stage=n_trials_per_stage
            )

        else:
            # Single stage optimization
            stage_map = {1: TestStage.RATE, 2: TestStage.ATTITUDE, 3: TestStage.POSITION}
            stage = stage_map[args.stage]

            logger.info(f"Starting Stage {args.stage} optimization...")

            if args.optimizer == "optuna":
                results = hierarchical_opt.optimize_stage(stage, n_trials=args.trials)
            else:
                results = hierarchical_opt.optimize_stage(
                    stage,
                    population_size=args.population,
                    n_generations=args.generations
                )

        # Print final summary
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Optimized parameter file: {args.output_dir}/final_optimized_parameters.parm")
        print("\nNext steps:")
        print("1. Review the optimization results in the output directory")
        print("2. Load the optimized parameters into Mission Planner or QGroundControl")
        print("3. Test in SITL before flying real drone")
        print("4. Perform flight tests and fine-tune if necessary")
        print("\n" + "="*70 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
