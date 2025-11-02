# Automated Drone Tuning System

## Overview
Automated PID tuning and parameter optimization system for 30kg quadcopter using ArduPilot SITL.

## Architecture

```
optimization_system/
├── main.py                      # Main entry point
├── config.py                    # Configuration and parameters
├── sitl_manager.py              # SITL instance management
├── optimizer.py                 # Optimization algorithms (GA/Bayesian)
├── performance_evaluator.py     # Fitness evaluation
├── test_sequences.py            # Automated test missions
├── telemetry_logger.py          # Data logging
├── visualizer.py                # Results visualization
├── utils.py                     # Utility functions
└── requirements.txt             # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Single Phase Optimization

```bash
# Rate controller tuning (Phase 1)
python main.py --phase phase1_rate --algorithm genetic --generations 100

# Attitude controller tuning (Phase 2)
python main.py --phase phase2_attitude --algorithm genetic --generations 100

# Position controller tuning (Phase 3)
python main.py --phase phase3_position --algorithm genetic --generations 100

# Advanced parameters (Phase 4)
python main.py --phase phase4_advanced --algorithm genetic --generations 100
```

### Run Complete Optimization

```bash
# All phases sequentially
python main.py --phase all --algorithm genetic --generations 100 --parallel 10
```

### Algorithm Options

```bash
# Genetic Algorithm (recommended for global search)
python main.py --algorithm genetic

# Bayesian Optimization (sample efficient)
python main.py --algorithm bayesian
```

### Parallel Processing

```bash
# Run 20 parallel SITL instances
python main.py --parallel 20

# With SITL speedup
python main.py --parallel 10 --speedup 5
```

### Resume from Checkpoint

```bash
python main.py --resume logs/checkpoint_phase1_rate_20231101_120000.pkl
```

## Output Files

- `optimized_params_TIMESTAMP.param` - ArduPilot parameter file
- `final_results_TIMESTAMP.pkl` - Complete results (pickle)
- `optimization_TIMESTAMP.log` - Detailed log
- `convergence_plot_TIMESTAMP.png` - Fitness convergence
- `performance_report_TIMESTAMP.pdf` - Full report

## Optimization Phases

1. **Phase 1: Rate Controllers** - Inner loop (most critical)
2. **Phase 2: Attitude Controllers** - Middle loop
3. **Phase 3: Position Controllers** - Outer loop
4. **Phase 4: Advanced Parameters** - Fine-tuning

## Configuration

Edit `config.py` to modify:
- Drone physical parameters
- Optimization bounds
- Fitness weights
- Safety constraints
- Test sequences

## Performance Metrics

- Rise time
- Settling time
- Overshoot
- Steady-state error
- Phase margin
- Disturbance rejection
- Power efficiency

## Safety Features

- Crash detection
- Motor saturation monitoring
- Attitude limit enforcement
- Altitude bounds checking
- Oscillation detection
