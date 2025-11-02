# ArduPilot Modifications for 30kg Quadcopter

This directory contains all modifications made to the ArduPilot SITL (Software In The Loop) environment to support the custom 30kg heavy-lift quadcopter X-frame drone.

## Overview

These modifications enable the automated PID tuning system to simulate a realistic 30kg quadcopter with appropriate physics, parameters, and motor characteristics.

---

## Modified Files

### 1. **Tools/autotest/pysim/vehicleinfo.py**

**Purpose:** Registers the custom "drone-30kg" vehicle model in SITL

**Modification:** Added new vehicle definition

```python
"drone-30kg": {
    "model": "x:@ROMFS/models/drone_30kg.json",
    "waf_target": "bin/arducopter",
    "default_params_filename": [
        "default_params/copter.parm",
        "default_params/copter-30kg.parm",
    ],
},
```

**Location in ArduPilot:** Around line 224 (in the Copter models section)

---

### 2. **Tools/autotest/default_params/copter-30kg.parm** (NEW FILE)

**Purpose:** Parameter file specifically tuned for 30kg quadcopter

**Key Parameters:**

#### Frame Configuration
- `FRAME_CLASS = 1` (Quad)
- `FRAME_TYPE = 1` (X configuration)

#### Motor Configuration (Heavy-lift)
- `MOT_THST_EXPO = 0.55` - Thrust curve exponential
- `MOT_THST_HOVER = 0.5` - 50% throttle for hover (high due to weight)
- `MOT_BAT_VOLT_MIN = 36.0V` - 12S LiPo minimum (3.0V × 12)
- `MOT_BAT_VOLT_MAX = 50.4V` - 12S LiPo maximum (4.2V × 12)
- `MOT_SPIN_MIN = 0.15`
- `MOT_SPIN_MAX = 0.95`
- `MOT_SPOOL_TIME = 0.5` - Longer spool time for large motors

#### Battery
- `BATT_CAPACITY = 30000` mAh (30Ah for extended flight time)

#### Rate Controllers (Conservative Initial Tuning)
- `ATC_RAT_RLL_P = 0.10` (lower for heavier inertia)
- `ATC_RAT_RLL_I = 0.08`
- `ATC_RAT_RLL_D = 0.004`
- `ATC_RAT_PIT_P = 0.10`
- `ATC_RAT_PIT_I = 0.08`
- `ATC_RAT_PIT_D = 0.004`
- `ATC_RAT_YAW_P = 0.25`
- `ATC_RAT_YAW_I = 0.02`

#### Position Controllers
- `PSC_VELXY_P = 0.8`
- `PSC_VELXY_I = 0.4`
- `PSC_VELXY_D = 0.15`
- `PSC_ACCZ_P = 0.20`
- `PSC_ACCZ_I = 0.40`

#### Safety
- `FENCE_RADIUS = 200m` - Safety geofence
- `FS_THR_ENABLE = 1` - Throttle failsafe enabled
- `LAND_SPEED = 50 cm/s` - Slower landing for heavy drone

**Note:** These parameters serve as conservative starting points. The automated optimization system will tune 50+ parameters to achieve optimal performance.

---

### 3. **Tools/autotest/models/drone_30kg.json** (NEW FILE)

**Purpose:** Physical model definition for SITL physics simulation

**Key Physical Parameters:**

#### Mass & Dimensions
- `mass = 30.0 kg` - Total takeoff weight
- `diagonal_size = 1.5 m` - Motor-to-motor diagonal
- `disc_area = 0.85 m²` - Total rotor disc area (4 motors with ~20-22" props)

#### Reference Flight Conditions
- `refSpd = 15.0 m/s` - Cruise speed
- `refAngle = 25.0°` - Pitch angle at cruise
- `refVoltage = 44.4V` - 12S nominal (3.7V × 12)
- `refCurrent = 45.0A` - Cruise current draw
- `refAlt = 50m` - Reference altitude MSL
- `refTempC = 25°C` - Standard temperature
- `refBatRes = 0.02Ω` - Battery internal resistance

#### Battery
- `maxVoltage = 50.4V` - Fully charged 12S
- `battCapacityAh = 30Ah` - ~20-25 min flight time

#### Motor/Propulsion
- `propExpo = 0.55` - MOT_THST_EXPO
- `hoverThrOut = 0.5` - 50% throttle for hover
- `pwmMin = 1000`
- `pwmMax = 2000`
- `spin_min = 0.15`
- `spin_max = 0.95`
- `slew_max = 60` - Motor slew rate %/sec

#### Aerodynamics
- `num_motors = 4` - Quadcopter configuration
- `mdrag_coef = 0.12` - Momentum drag coefficient
- `refRotRate = 90°/s` - Maximum yaw rate

**Note:** These values are engineering estimates. For production use, calibrate with actual flight test data.

---

## How to Apply These Modifications

### Option 1: Manual Copy (Recommended)

1. **Clone ArduPilot** (if not already done):
   ```bash
   git clone https://github.com/ArduPilot/ardupilot.git
   cd ardupilot
   git submodule update --init --recursive
   ```

2. **Copy modified files:**
   ```bash
   # From your project root
   cp ardupilot_modifications/Tools/autotest/default_params/copter-30kg.parm \
      ardupilot/Tools/autotest/default_params/

   cp ardupilot_modifications/Tools/autotest/models/drone_30kg.json \
      ardupilot/Tools/autotest/models/
   ```

3. **Apply vehicleinfo.py modification:**
   ```bash
   # Edit ardupilot/Tools/autotest/pysim/vehicleinfo.py
   # Add the drone-30kg definition shown above around line 224
   ```

4. **Build ArduCopter:**
   ```bash
   cd ardupilot/ArduCopter
   ../Tools/autotest/sim_vehicle.py --build-only
   ```

### Option 2: Git Patch (Alternative)

If you prefer, you can create a patch file:

```bash
cd ardupilot
git diff Tools/autotest/pysim/vehicleinfo.py > ../30kg-modifications.patch
git apply ../30kg-modifications.patch
```

---

## Running SITL with 30kg Drone

Once modifications are applied:

```bash
cd ardupilot/ArduCopter

# Standard launch
../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg --map --console

# With speedup (for optimization)
../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg --speedup 2

# Custom home location
../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg \
  --location CMAC --map --console
```

**Note:** Remember to source profile first:
```bash
. ~/.profile
```

---

## Integration with Optimization System

The optimization system in `optimization_system/` automatically:

1. Launches multiple SITL instances with `-f drone-30kg`
2. Loads `copter-30kg.parm` parameters
3. Uses physics model from `drone_30kg.json`
4. Tests and optimizes all 50+ parameters
5. Validates with comprehensive flight tests

---

## Verification

To verify modifications are applied correctly:

```bash
cd ardupilot/ArduCopter

# Check if vehicle is registered
../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg --help

# Launch and check parameters
../Tools/autotest/sim_vehicle.py -v ArduCopter -f drone-30kg

# In MAVProxy console:
# param show FRAME_*
# param show MOT_*
```

Expected output should show:
- `FRAME_CLASS = 1`
- `MOT_THST_HOVER = 0.5`
- Other 30kg-specific parameters loaded

---

## File Structure

```
ardupilot_modifications/
├── README.md (this file)
└── Tools/
    └── autotest/
        ├── pysim/
        │   └── vehicleinfo.py (modified)
        ├── default_params/
        │   └── copter-30kg.parm (new)
        └── models/
            └── drone_30kg.json (new)
```

---

## Important Notes

### 1. **Conservative Starting Parameters**
The initial PID gains are deliberately conservative (low) to ensure:
- No oscillations during initial testing
- Safe starting point for optimization
- Stable baseline for improvement

### 2. **Optimization Will Override**
The automated tuning system will:
- Test these baseline parameters
- Systematically optimize all 50+ parameters
- Generate final optimized `.param` file
- Conservative values ensure safe optimization start

### 3. **Physical Model Estimates**
The `drone_30kg.json` values are engineering estimates based on:
- Typical 30kg quadcopter characteristics
- 12S battery configuration
- Large motor/propeller combinations
- Industry standard power-to-weight ratios

**For production use:** Update with actual measured values from:
- Motor thrust tests
- Battery discharge curves
- Flight test data
- Moment of inertia measurements

### 4. **ArduPilot Version**
These modifications are compatible with:
- ArduPilot master branch (tested Nov 2025)
- Copter 4.6+ firmware
- May require minor adjustments for older versions

---

## Troubleshooting

### Vehicle not found
```
Error: Vehicle model 'drone-30kg' not found
```
**Solution:** Verify `vehicleinfo.py` modification was applied correctly

### Parameters not loading
```
Warning: Using default parameters
```
**Solution:** Check `copter-30kg.parm` is in correct directory and has proper syntax

### Physics simulation errors
```
Error loading model file
```
**Solution:** Verify `drone_30kg.json` syntax (valid JSON format)

### Build errors
```
Failed to build ArduCopter
```
**Solution:** Run `git submodule update --init --recursive` and rebuild

---

## References

- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [Parameter Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)
- [Custom Models in SITL](https://ardupilot.org/dev/docs/sitl-with-JSON.html)

---

## License

These modifications are intended for use with ArduPilot, which is licensed under GPL v3.
