# MC07 - Automated Drone PID Tuning System

**Automated parameter optimization for 30kg heavy-lift quadcopter using ArduPilot SITL**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ArduPilot](https://img.shields.io/badge/ArduPilot-SITL-green.svg)](https://ardupilot.org/)

---

## 📋 Project Overview

This system provides **fully automated PID tuning** for a 30kg quadcopter X-frame drone with **no manual tuning required**. It uses ArduPilot SITL (Software In The Loop) simulation with hierarchical optimization to tune 50+ parameters across rate, attitude, and position controllers.

### Drone Specifications
- **Mass:** 30 kg
- **Motors:** MELARD 1026 (Kv=100, 14S)
- **Propellers:** 36×19 inch
- **ESC:** E150 14S (150A, PWM 400Hz)
- **Battery:** 14S LiPo, 38.5Ah
- **Thrust-to-Weight Ratio:** 3.2:1
- **Inertia:** I_xx=2.78, I_yy=4.88, I_zz=7.18 kg·m²

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/imkunal007219/MC07_tuning.git
cd MC07_tuning

# Install Python dependencies
pip install -r optimization/requirements.txt

# Install ArduPilot SITL (if not already done)
cd ardupilot
./waf configure --board sitl
./waf copter
```

### 2. Run Optimization

**Option A: Full Hierarchical Optimization (Bayesian)**
```bash
cd optimization
python optimize_drone.py --optimizer optuna --trials 30 --all-stages
```

**Option B: Full Hierarchical Optimization (Genetic Algorithm)**
```bash
python optimize_drone.py --optimizer deap --generations 30 --all-stages
```

**Option C: Single Stage Only**
```bash
# Stage 1: Rate Controllers (most critical)
python optimize_drone.py --optimizer optuna --stage 1 --trials 50
```

### 3. Load Optimized Parameters

```bash
# Parameters saved to: optimization_results/final_optimized_parameters.parm
# Load into Mission Planner or QGroundControl
# Test in SITL before real flight!
```

---

## 📚 System Architecture

### Hierarchical Optimization (3 Stages)

```
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: Rate Controllers (Inner Loop - 400Hz)         │
│ Parameters: ATC_RAT_RLL_P/I/D, ATC_RAT_PIT_P/I/D, etc │
│ Test: Gyro rate tracking, stability                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Attitude Controllers (Middle Loop - 100Hz)    │
│ Parameters: ATC_ANG_RLL_P, ATC_ANG_PIT_P, etc         │
│ Test: Angle tracking, smooth transitions               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 3: Position Controllers (Outer Loop - 10Hz)      │
│ Parameters: PSC_VELXY_P/I/D, PSC_ACCZ_P/I, etc        │
│ Test: Position hold, waypoint navigation               │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description | File |
|-----------|-------------|------|
| **Parameter Search Spaces** | Defines optimization bounds for 50+ parameters | `parameter_search_spaces.py` |
| **Parallel SITL Manager** | Manages 4 simultaneous SITL instances | `parallel_sitl_manager.py` |
| **Fitness Evaluator** | Calculates performance metrics | `fitness_evaluator.py` |
| **Optuna Optimizer** | Bayesian optimization (sample-efficient) | `optimizer_optuna.py` |
| **DEAP Optimizer** | Genetic algorithm (robust, parallel-friendly) | `optimizer_deap.py` |
| **Hierarchical Framework** | Orchestrates 3-stage optimization | `hierarchical_optimizer.py` |
| **Main CLI** | Command-line interface | `optimize_drone.py` |

---

## 🎯 Usage Examples

### Basic Usage

```bash
# Full optimization with default settings (Optuna, 30 trials per stage)
python optimize_drone.py --optimizer optuna --all-stages

# Full optimization with Genetic Algorithm
python optimize_drone.py --optimizer deap --generations 30 --all-stages
```

### Advanced Usage

```bash
# High-quality optimization (more trials = better results but slower)
python optimize_drone.py --optimizer optuna --trials 100 --all-stages

# Optimize only rate controllers (Stage 1) with many trials
python optimize_drone.py --optimizer optuna --stage 1 --trials 200

# Use 4 parallel SITL instances (faster on multi-core systems)
python optimize_drone.py --optimizer deap --parallel 4 --all-stages

# Verbose logging for debugging
python optimize_drone.py --optimizer optuna --all-stages --verbose

# Dry run (check configuration without running)
python optimize_drone.py --optimizer optuna --all-stages --dry-run
```

---

## 📊 Optimization Algorithms

### Optuna (Bayesian Optimization)

**Advantages:**
- ✅ Sample-efficient (fewer evaluations needed)
- ✅ Good for expensive simulations
- ✅ Provides uncertainty estimates
- ✅ Handles high-dimensional spaces well

**When to use:** Default choice for most cases, especially when simulation time is long.

```bash
python optimize_drone.py --optimizer optuna --trials 30 --all-stages
```

### DEAP (Genetic Algorithm)

**Advantages:**
- ✅ Robust global search (avoids local optima)
- ✅ Excellent parallelization
- ✅ Works well with noisy evaluations
- ✅ Easy to understand and debug

**When to use:** When you want more exploration or have many CPU cores for parallel evaluation.

```bash
python optimize_drone.py --optimizer deap --generations 30 --population 20 --all-stages
```

---

## 📈 Expected Timeline

### With 4 Parallel Instances

| Stage | Parameters | Trials/Generations | Time Estimate |
|-------|------------|-------------------|---------------|
| **Stage 1 (Rate)** | 18 params | 30-50 | 45-60 min |
| **Stage 2 (Attitude)** | 5 params | 20-30 | 30-40 min |
| **Stage 3 (Position)** | 12 params | 20-30 | 40-50 min |
| **TOTAL** | **35+ params** | **70-110** | **~2-3 hours** |

*Without parallelization: 20-30 hours!*

---

## 📁 Project Structure

```
MC07_tuning/
├── ardupilot/                          # ArduPilot SITL
│   ├── Tools/autotest/
│   │   ├── models/drone_30kg.json      # 30kg drone physics model
│   │   └── default_params/copter-30kg.parm  # Baseline parameters
│   └── ...
├── optimization/                       # Optimization framework
│   ├── optimize_drone.py              # Main CLI script ⭐
│   ├── hierarchical_optimizer.py      # 3-stage orchestration
│   ├── parameter_search_spaces.py     # Search space definitions
│   ├── parallel_sitl_manager.py       # Multi-instance SITL
│   ├── fitness_evaluator.py           # Performance metrics
│   ├── optimizer_optuna.py            # Bayesian optimization
│   ├── optimizer_deap.py              # Genetic algorithm
│   └── requirements.txt               # Python dependencies
├── DRONE_MODEL_UPDATE_SUMMARY.md      # Model specifications
├── DRONE_PARAMETERS_REQUIRED.md       # Parameter documentation
└── README.md                          # This file
```

---

## 🔧 Parameter Search Spaces

### Stage 1: Rate Controllers (18 parameters)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `ATC_RAT_RLL_P` | 0.05 - 0.25 | 0.12 | Roll rate P gain |
| `ATC_RAT_RLL_I` | 0.04 - 0.20 | 0.10 | Roll rate I gain |
| `ATC_RAT_RLL_D` | 0.001 - 0.012 | 0.005 | Roll rate D gain |
| `ATC_RAT_PIT_P` | 0.04 - 0.20 | 0.10 | Pitch rate P gain |
| `ATC_RAT_YAW_P` | 0.10 - 0.40 | 0.20 | Yaw rate P gain |
| `INS_GYRO_FILTER` | 40 - 120 | 80 | Gyro filter (Hz) |
| ... | ... | ... | ... |

### Stage 2: Attitude Controllers (5 parameters)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `ATC_ANG_RLL_P` | 3.0 - 8.0 | 4.5 | Roll angle P gain |
| `ATC_ANG_PIT_P` | 3.0 - 8.0 | 4.5 | Pitch angle P gain |
| `ATC_ANG_YAW_P` | 3.0 - 8.0 | 4.5 | Yaw angle P gain |
| ... | ... | ... | ... |

### Stage 3: Position Controllers (12 parameters)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `PSC_POSXY_P` | 0.5 - 2.0 | 1.0 | Horizontal position P |
| `PSC_VELXY_P` | 0.5 - 2.0 | 0.9 | Horizontal velocity P |
| `PSC_ACCZ_P` | 0.15 - 0.50 | 0.25 | Vertical acceleration P |
| ... | ... | ... | ... |

*Full parameter list: see `parameter_search_spaces.py`*

---

## 📊 Output Files

After optimization completes, you'll find:

```
optimization_results/
├── final_optimized_parameters.parm    # Load this into ArduPilot ⭐
├── final_results.json                 # Complete results
├── stage1_optuna_results.json         # Stage 1 details
├── stage1_optuna_results.parm         # Stage 1 parameters
├── stage2_optuna_results.json         # Stage 2 details
├── stage3_optuna_results.json         # Stage 3 details
└── optimization.log                   # Full log
```

---

## 🧪 Testing Optimized Parameters

### 1. Test in SITL First

```bash
cd ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -f drone-30kg --console --map

# In MAVProxy console:
param load ../../optimization_results/final_optimized_parameters.parm
arm throttle
mode guided
takeoff 10
```

### 2. Verify Stability

- ✅ Hover should be stable (no oscillations)
- ✅ Step responses should be smooth
- ✅ Position hold should be accurate (< 1m drift)
- ✅ No excessive motor saturation

### 3. Flight Test (Real Drone)

**⚠️ IMPORTANT SAFETY:**
- Start with conservative gains (80% of optimized values)
- First flight in calm conditions
- Test hover → small movements → full maneuvers
- Have experienced pilot ready to take over

---

## 🛠️ Troubleshooting

### SITL Won't Start

```bash
# Check ArduPilot build
cd ardupilot
./waf configure --board sitl
./waf copter

# Try running SITL manually
cd ArduCopter
../Tools/autotest/sim_vehicle.py -f drone-30kg
```

### Optimization is Slow

- Reduce number of trials: `--trials 20`
- Use fewer parallel instances on low-spec machines
- Use DEAP with smaller population: `--population 10`

### Results Not Converging

- Increase trials/generations: `--trials 100`
- Try different optimizer: switch between `optuna` and `deap`
- Check SITL logs for crashes or instability

### Parameter Bounds Too Tight

- Edit `parameter_search_spaces.py`
- Adjust `min` and `max` values for specific parameters
- Rerun optimization

---

## 📖 References

- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [ArduCopter PID Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

---

## ⚠️ Disclaimer

**This is an automated tuning system. Always:**
- ✅ Test in SITL before real flights
- ✅ Start conservatively (reduce optimized gains by 20%)
- ✅ Have experienced pilot supervision
- ✅ Follow local regulations and safety guidelines
- ❌ Never skip safety checks
- ❌ Never fly untested parameters on real hardware

**The authors are not responsible for any damage or injury resulting from use of this system.**

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Happy Flying! 🚁**
