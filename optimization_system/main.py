"""
Main entry point for automated drone tuning system
"""

import argparse
import sys
import os
import logging
import signal
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from .sitl_manager import SITLManager
from optimizer import GeneticOptimizer, BayesianOptimizer
from .performance_evaluator import PerformanceEvaluator
from .utils import setup_logging, save_results

# Try to import multi-objective optimizer (requires pymoo)
try:
    from multi_objective_optimizer import MultiObjectiveOptimizer
    MULTI_OBJECTIVE_AVAILABLE = True
except ImportError:
    MULTI_OBJECTIVE_AVAILABLE = False

# Global SITL manager for signal handler
sitl_manager_global = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*80)
    print("⚠️  CTRL+C DETECTED - CLEANING UP...")
    print("="*80)

    global sitl_manager_global
    if sitl_manager_global:
        print("🔄 Stopping all SITL instances...")
        sitl_manager_global.cleanup()
        print("✓ Cleanup complete")

    print("👋 Exiting...")
    sys.exit(0)


def main():
    """Main optimization loop"""
    parser = argparse.ArgumentParser(description='Automated Drone Tuning System')
    parser.add_argument('--phase', type=str, default='phase1_rate',
                        choices=['phase1_rate', 'phase2_attitude', 'phase3_position', 'phase4_advanced', 'all'],
                        help='Optimization phase to run')
    parser.add_argument('--algorithm', type=str, default='genetic',
                        choices=['genetic', 'bayesian', 'multi-objective'],
                        help='Optimization algorithm (multi-objective requires pymoo)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Maximum number of generations')
    parser.add_argument('--parallel', type=int, default=10,
                        help='Number of parallel SITL instances')
    parser.add_argument('--speedup', type=int, default=1,
                        help='SITL speedup factor')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint file')
    parser.add_argument('--bounds-mode', type=str, default='narrow',
                        choices=['wide', 'narrow', 'adaptive'],
                        help='Parameter bounds mode: wide (comprehensive search), '
                             'narrow (±30%% around physics values), adaptive (expand as needed)')
    parser.add_argument('--hierarchical-constraints', action='store_true', default=True,
                        help='Enforce bandwidth separation constraints for cascade control (default: True)')
    parser.add_argument('--no-hierarchical-constraints', dest='hierarchical_constraints',
                        action='store_false',
                        help='Disable hierarchical bandwidth constraints')
    parser.add_argument('--intelligent-sequencing', action='store_true', default=True,
                        help='Use progressive test sequencing for faster evaluation (default: True)')
    parser.add_argument('--no-intelligent-sequencing', dest='intelligent_sequencing',
                        action='store_false',
                        help='Disable intelligent test sequencing (use single mission tests)')
    parser.add_argument('--early-crash-detection', action='store_true', default=True,
                        help='Enable early crash detection using Lyapunov stability (default: True)')
    parser.add_argument('--no-early-crash-detection', dest='early_crash_detection',
                        action='store_false',
                        help='Disable early crash detection')

    args = parser.parse_args()

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{config.LOGGING_CONFIG['log_dir']}/optimization_{timestamp}.log"
    logger = setup_logging(log_file, config.LOGGING_CONFIG['log_level'])

    logger.info("="*80)
    logger.info("AUTOMATED DRONE TUNING SYSTEM STARTED")
    logger.info("="*80)
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Max Generations: {args.generations}")
    logger.info(f"Parallel Instances: {args.parallel}")
    logger.info(f"SITL Speedup: {args.speedup}")
    logger.info(f"Bounds Mode: {args.bounds_mode}")
    if args.bounds_mode in config.BOUNDS_MODE_CONFIG:
        logger.info(f"  {config.BOUNDS_MODE_CONFIG[args.bounds_mode]['description']}")
    logger.info(f"Hierarchical Constraints: {'Enabled' if args.hierarchical_constraints else 'Disabled'}")
    if args.hierarchical_constraints:
        logger.info("  Enforcing bandwidth separation: Rate > 4-15x Attitude > 3-8x Position")
    logger.info(f"Intelligent Test Sequencing: {'Enabled' if args.intelligent_sequencing else 'Disabled'}")
    if args.intelligent_sequencing:
        logger.info("  Progressive testing: Hover → Small Step → Large Step → Frequency → Trajectory")
        logger.info("  Expected time savings: ~40%")
    logger.info(f"Early Crash Detection: {'Enabled' if args.early_crash_detection else 'Disabled'}")
    if args.early_crash_detection:
        logger.info("  Lyapunov stability criteria for early abort")
        logger.info("  Expected time savings: ~75% for crash cases")

    # Initialize components
    logger.info("\nInitializing components...")

    global sitl_manager_global
    sitl_manager = SITLManager(
        num_instances=args.parallel,
        speedup=args.speedup
    )
    sitl_manager_global = sitl_manager  # Store for signal handler

    performance_evaluator = PerformanceEvaluator()

    # Get optimization phases with selected bounds mode
    logger.info(f"\nLoading parameter bounds (mode: {args.bounds_mode})...")
    optimization_phases = config.get_optimization_bounds(args.bounds_mode)
    logger.info(f"✓ Loaded {len(optimization_phases)} optimization phases")

    # Select optimizer
    if args.algorithm == 'genetic':
        optimizer = GeneticOptimizer(
            sitl_manager=sitl_manager,
            evaluator=performance_evaluator,
            max_generations=args.generations,
            population_size=config.OPTIMIZATION_CONFIG['population_size'],
            drone_params=config.DRONE_PARAMS,  # Enable physics-based seeding
            use_physics_seeding=True,  # Use control theory to seed population
            enforce_hierarchical_constraints=args.hierarchical_constraints,  # Bandwidth separation constraints
            use_intelligent_sequencing=args.intelligent_sequencing  # Progressive test sequencing
        )
    elif args.algorithm == 'bayesian':
        optimizer = BayesianOptimizer(
            sitl_manager=sitl_manager,
            evaluator=performance_evaluator,
            max_iterations=args.generations,
            use_intelligent_sequencing=args.intelligent_sequencing  # Progressive test sequencing
        )
    elif args.algorithm == 'multi-objective':
        if not MULTI_OBJECTIVE_AVAILABLE:
            logger.error("Multi-objective optimization requires pymoo library")
            logger.error("Install with: pip install pymoo")
            sys.exit(1)
        optimizer = MultiObjectiveOptimizer(
            sitl_manager=sitl_manager,
            evaluator=performance_evaluator,
            population_size=config.OPTIMIZATION_CONFIG['population_size']
        )
        logger.info("Multi-objective optimization selected (NSGA-II)")
        logger.info("Will generate Pareto front of trade-off solutions")
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        sys.exit(1)

    # Determine phases to run
    if args.phase == 'all':
        phases = ['phase1_rate', 'phase2_attitude', 'phase3_position', 'phase4_advanced']
    else:
        phases = [args.phase]

    # Run optimization for each phase
    all_results = {}

    for phase_name in phases:
        logger.info("\n" + "="*80)
        logger.info(f"STARTING PHASE: {optimization_phases[phase_name]['name']}")
        logger.info("="*80)

        phase_config = optimization_phases[phase_name]

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
        checkpoint_file = f"{config.LOGGING_CONFIG['log_dir']}/checkpoint_{phase_name}_{timestamp}.pkl"
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

    # Validate hierarchical bandwidth separation
    if args.hierarchical_constraints and args.algorithm == 'genetic':
        logger.info("\nValidating cascade control bandwidth separation...")
        if optimizer.hierarchical_validator:
            cascade_validation = optimizer.hierarchical_validator.get_full_system_validation()
            validation_results['cascade_bandwidth'] = cascade_validation

            if cascade_validation.get('valid', False):
                logger.info("✓ All cascade bandwidth constraints satisfied")
                bw = cascade_validation.get('bandwidths', {})
                ratios = cascade_validation.get('ratios', {})
                logger.info(f"  Rate loop:     {bw.get('rate', 0):.2f} Hz")
                logger.info(f"  Attitude loop: {bw.get('attitude', 0):.2f} Hz (separation: {ratios.get('rate_to_attitude', 0):.2f}x)")
                logger.info(f"  Position loop: {bw.get('position', 0):.2f} Hz (separation: {ratios.get('attitude_to_position', 0):.2f}x)")
            else:
                logger.warning("✗ Cascade bandwidth validation failed")
                for violation in cascade_validation.get('violations', []):
                    logger.warning(f"  {violation}")

    # Save final results
    final_output = {
        'optimized_parameters': final_params,
        'phase_results': all_results,
        'validation_results': validation_results,
        'timestamp': timestamp,
    }

    output_file = f"{config.LOGGING_CONFIG['log_dir']}/final_results_{timestamp}.pkl"
    save_results(output_file, final_output)

    # Generate parameter file
    param_file = f"{config.LOGGING_CONFIG['log_dir']}/optimized_params_{timestamp}.param"
    generate_param_file(final_params, param_file, optimization_phases)

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


def generate_param_file(parameters, output_file, optimization_phases):
    """Generate ArduPilot parameter file"""
    with open(output_file, 'w') as f:
        f.write("# Optimized parameters for 30kg quadcopter\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n\n")

        # Group by category
        for phase_name, phase_config in optimization_phases.items():
            f.write(f"# {phase_config['name']}\n")
            for param in phase_config['parameters']:
                if param in parameters:
                    f.write(f"{param},{parameters[param]:.6f}\n")
            f.write("\n")


if __name__ == '__main__':
    sys.exit(main())
