# PROJECT MEMORY - Automated Drone Tuning System

## PROJECT OBJECTIVE
Create an **automated PID tuning and parameter optimization system** for a **30kg quadcopter X-frame drone** using ArduPilot SITL (Software In The Loop) with **no manual tuning required**.

---

## CORE REQUIREMENTS

### 1. **Primary Goal**
- Tune a 30kg quad-X frame drone running ArduPilot firmware
- Use actual ArduCopter control loop logic (not simplified simulation)
- **Automated optimization** - NO manual tuning iterations
- No visual simulation needed - only flight data for validation
- Must consider ALL parameters (not just PID gains)

### 2. **Critical Constraints**
- Timeline: Complete today
- No visual simulation required
- Must validate drone is tuned and working fine through flight data
- Cannot accept crashes due to missed parameters
- User has drone data but doesn't know how to create mathematical model

---

## TECHNICAL ARCHITECTURE

### **System Components**

```
┌─────────────────────────────────────────────────────────────┐
│          AUTOMATED OPTIMIZATION LOOP                        │
│                                                             │
│  Optimization Algorithm (GA/Bayesian)                       │
│           ↓                                                 │
│  SITL Simulation (Multiple Parallel Instances)              │
│           ↓                                                 │
│  Performance Metrics Evaluation                             │
│           ↓                                                 │
│  Parameter Adjustment                                       │
│           ↓                                                 │
│  Convergence Check → If not converged, repeat               │
└─────────────────────────────────────────────────────────────┘
```

### **Software Stack**
1. ArduPilot source code (latest stable)
2. MAVProxy / pymavlink for MAVLink communication
3. Python optimization libraries (Optuna, DEAP, scipy)
4. dronekit-python for high-level control
5. Custom SITL physics model for 30kg drone

---

## REQUIRED DRONE DATA

### **Physical Properties**
| Parameter | Description | Unit | Example |
|-----------|-------------|------|---------|
| mass | Total vehicle mass | kg | 30.0 |
| Ixx | Roll axis inertia | kg·m² | 6.0 |
| Iyy | Pitch axis inertia | kg·m² | 6.0 |
| Izz | Yaw axis inertia | kg·m² | 10.0 |
| arm_length | Center to motor distance | m | 0.75 |
| frame_height | Frame height | m | 0.1 |
| disc_area | Total rotor disc area | m² | 1.5 |

### **Motor/Propeller**
| Parameter | Description | Unit | Example |
|-----------|-------------|------|---------|
| thrust_coefficient (kt) | Motor thrust coefficient | N/(rad/s)² | 2.86e-5 |
| torque_coefficient (kq) | Motor torque coefficient | Nm/(rad/s)² | 4.77e-7 |
| propExpo (MOT_THST_EXPO) | Thrust curve shape | 0-1 | 0.75 |
| Motor Kv | Motor Kv rating | RPM/V | TBD |
| Propeller diameter | Prop size | inches | 20-24 |
| Max thrust per motor | Maximum thrust | N | 75+ |

### **ESC/Battery**
| Parameter | Description | Unit | Example |
|-----------|-------------|------|---------|
| ESC update rate | ESC frequency | Hz | 400 |
| MOT_BAT_VOLT_MAX | Max battery voltage | V | 50.4 (12S) |
| MOT_BAT_VOLT_MIN | Min battery voltage | V | 42.0 (12S) |
| battCapacityAh | Battery capacity | Ah | 22 |

---

## CRITICAL FILES TO MODIFY

### **ArduPilot Source Code**
1. `libraries/SITL/SIM_Frame_30kg.cpp` - Custom frame physics
2. `libraries/SITL/SIM_Multicopter.cpp` - Physics calculations
3. `Tools/autotest/pysim/vehicleinfo.py` - Frame registration
4. `Tools/autotest/default_params/30kg_quad.parm` - Parameter file

---

## PARAMETERS TO OPTIMIZE

### **Hierarchical Optimization Order**
1. **Phase 1: Rate Controllers** (Inner Loop - Most Critical)
   - ATC_RAT_RLL_P, ATC_RAT_RLL_I, ATC_RAT_RLL_D
   - ATC_RAT_PIT_P, ATC_RAT_PIT_I, ATC_RAT_PIT_D
   - ATC_RAT_YAW_P, ATC_RAT_YAW_I
   - Filters: INS_GYRO_FILTER, ATC_RAT_*_FLTD/FLTE/FLTT

2. **Phase 2: Attitude Controllers** (Middle Loop)
   - ATC_ANG_RLL_P, ATC_ANG_PIT_P, ATC_ANG_YAW_P
   - ATC_ACCEL_R_MAX, ATC_ACCEL_P_MAX, ATC_ACCEL_Y_MAX

3. **Phase 3: Position Controllers** (Outer Loop)
   - PSC_POSXY_P
   - PSC_VELXY_P, PSC_VELXY_I, PSC_VELXY_D
   - PSC_POSZ_P, PSC_VELZ_P
   - PSC_ACCZ_P, PSC_ACCZ_I, PSC_ACCZ_D

4. **Phase 4: Advanced Parameters**
   - MOT_THST_HOVER, MOT_SPIN_MIN/MAX
   - Input shaping, feed-forward terms
   - Navigation parameters

**Total Parameters to Optimize: 50+**

---

## FITNESS FUNCTION (Performance Metrics)

```python
fitness = w1*stability + w2*response_time + w3*overshoot +
          w4*steady_state_error + w5*power_efficiency
```

### **Metrics to Calculate**
- Rise time (time to 90% setpoint)
- Settling time (within 2% of setpoint)
- Overshoot percentage
- Steady-state error
- Phase/gain margins
- Disturbance rejection
- Motor saturation frequency
- Oscillation detection

### **Crash Penalties**
- Infinite negative fitness for crashes
- High penalty for sustained oscillations
- Penalty for motor saturation

---

## AUTOMATED TEST SEQUENCES

Each parameter set runs through:
1. **Basic Stability** - 5 second hover test
2. **Step Responses** - Roll, pitch, yaw, altitude steps
3. **Frequency Sweep** - Sine waves 0.1-10 Hz
4. **Trajectory Tracking** - Figure-8, square patterns
5. **Disturbance Rejection** - Simulated wind gusts
6. **Emergency Maneuvers** - Rapid stops from velocity

---

## OPTIMIZATION ALGORITHM RECOMMENDATION

**Primary: Bayesian Optimization (Optuna)**
- Sample efficient (minimizes simulation runs)
- Good for expensive simulations
- Provides uncertainty estimates
- Handles hierarchical optimization well

**Alternative: Genetic Algorithm (DEAP)**
- Good for global search
- Handles multiple parameters simultaneously
- Robust to noise

---

## SAFETY CONSTRAINTS

```python
safety_constraints = {
    'max_angle': 45,           # degrees
    'max_rate': 360,           # deg/s
    'max_altitude_error': 2,   # meters
    'max_position_error': 5,   # meters
    'min_phase_margin': 45,    # degrees
    'max_oscillation_amp': 5,  # degrees
}
```

---

## EXPECTED TIMELINE

1. **Setup** (2-3 hours)
   - ArduPilot environment
   - Custom physics model
   - Optimization framework

2. **Initial Rough Tune** (30 min - 100 iterations)
   - Parallel SITL instances
   - Basic stability validation

3. **Fine Tuning** (2-3 hours - 1000+ iterations)
   - Hierarchical optimization
   - All parameter phases

4. **Validation** (30 min)
   - Final test missions
   - Performance reports

**Total: 5-6 hours** (vs days of manual tuning)

---

## PARALLEL PROCESSING

- Run 10-20 SITL instances simultaneously
- Linear speedup with CPU cores
- Recommended: 16+ cores, 32GB RAM, SSD

---

## OUTPUT DELIVERABLES

1. **Optimized parameter file** (`.param`)
2. **Convergence plots** (fitness vs generation)
3. **Performance report** (all metrics)
4. **Safety validation results**
5. **Frequency response plots**
6. **Step response characteristics**
7. **Confidence intervals on parameters**

---

## CRITICAL REMINDERS

⚠️ **NEVER FORGET:**
1. This is **automated tuning** - NO manual iterations
2. Must optimize **ALL 50+ parameters**, not just PIDs
3. Use **hierarchical optimization** (Rate → Attitude → Position)
4. **Safety constraints** must be enforced at all times
5. User needs this **completed today**
6. No visual simulation - **data logging only**
7. Must use **actual ArduCopter firmware logic**
8. **30kg quad X-frame** - not default drone
9. Validate with comprehensive test missions before finalizing

---

## IMPLEMENTATION APPROACH

### **Step-by-Step Execution**
1. Setup ArduPilot SITL environment
2. Create custom 30kg drone physics model
3. Implement automated optimization framework
4. Run hierarchical parameter optimization
5. Validate results with test missions
6. Generate final parameter file and reports

### **Python Libraries Needed**
- `optuna` - Bayesian optimization
- `pymavlink` - MAVLink communication
- `dronekit` - Drone control
- `scipy` - Signal processing
- `numpy`, `pandas` - Data handling
- `matplotlib` - Visualization

---

## KEY SUCCESS CRITERIA

✅ Drone achieves stable hover
✅ Step responses meet rise/settling time requirements
✅ No oscillations or instability
✅ Position hold accuracy < 1 meter
✅ Passes all safety constraint checks
✅ All 50+ parameters optimized
✅ Complete validation test suite passed

---
TODO list
     ❎ Analyze the project requirements from the chat history
     ❎ Create comprehensive todo list and memory file
     ❎ Set up ArduPilot development environment
     ❎ Clone and build ArduPilot SITL for copter
     ❎ Create custom 30kg drone physics model in SITL
     ❎ Implement custom frame definition (SIM_Frame_30kg.cpp)
     ❎ Create parameter file for 30kg quad (copter-30kg.parm)
     ❎ Build automated optimization system architecture
     ❎ Implement SITL manager for parallel instances
     ❎ Create fitness evaluation system with performance metrics
     ❎ Implement optimization algorithm (Genetic Algorithm/Bayesian)
     ❎ Create automated test sequences (hover, step response, trajectory)
     ❎ Implement hierarchical optimization (Rate → Attitude → Position)
     ❎ Add safety constraints and crash detection
     ❎ Create data logging and telemetry collection system
     ❎ Implement convergence criteria and stopping conditions
     ❎ Create visualization tools for optimization progress
     ☐ Run initial optimization with conservative parameters
     ☐ Validate optimized parameters with test missions
     ☐ Generate final parameter file and performance reports


**notes 
1.if you want to run any ardupilot command or simulation , use this ". ~/.profile" command first 

2. do not create any summary after task completion unless asked to as we have to save token and work efficiently 

3. if you want to run arducopter sitl you go to this folder /Documents/WORK/mc07 sitl/ardupilot/ArduCopter and run this command "../Tools/autotest/sim_vehicle.py --map --console" , this is a standard command you can modify yourself according to our needs as we have made a custom drone model with custom parameters 

