# Required Drone Parameters for SITL Modeling and Optimization

**Document Version:** 1.0
**Date:** November 3, 2025
**Project:** MC07 - 30kg Quadcopter X-Frame Automated Tuning System

---

## Executive Summary

This document specifies all physical, mechanical, and electrical parameters required from the **Aerodynamic Design Team** and **Integration Team** to create an accurate Software-In-The-Loop (SITL) simulation model for automated PID tuning and parameter optimization.

**Critical:** Without accurate parameters, the optimization results will not transfer to the real drone, potentially causing crashes or poor performance.

---

## Table of Contents

1. [Airframe & Structural Parameters](#1-airframe--structural-parameters)
2. [Motor & Propeller Parameters](#2-motor--propeller-parameters)
3. [Electronic Speed Controller (ESC) Parameters](#3-electronic-speed-controller-esc-parameters)
4. [Battery & Power System Parameters](#4-battery--power-system-parameters)
5. [Sensor & IMU Parameters](#5-sensor--imu-parameters)
6. [Aerodynamic Parameters](#6-aerodynamic-parameters)
7. [Mass Distribution & Inertia](#7-mass-distribution--inertia)
8. [Environmental & Operational Parameters](#8-environmental--operational-parameters)
9. [Control System Baseline](#9-control-system-baseline)
10. [Data Collection Guidelines](#10-data-collection-guidelines)

---

## 1. Airframe & Structural Parameters

**Responsible Team:** Integration Team + Structural Design

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Measure |
|-----------|--------|-------------|------|---------------|----------|----------------|
| **Total Vehicle Mass** | `m_total` | Complete assembled drone mass including battery | kg | 30.0 | CRITICAL | Calibrated scale, weigh fully assembled |
| **Frame Type** | - | Configuration type | - | X-frame quad | CRITICAL | Design specification |
| **Motor-to-Motor Distance** | `d_motor` | Center-to-center distance between diagonal motors | m | 1.50 | CRITICAL | Measuring tape or CAD |
| **Arm Length** | `L_arm` | Distance from center of gravity to motor axis | m | 0.75 | CRITICAL | CAD model or physical measurement |
| **Frame Height** | `h_frame` | Vertical distance from bottom to top | m | 0.10 | HIGH | Caliper or CAD |
| **Frame Width** | `w_frame` | Maximum width of frame | m | 1.50 | MEDIUM | Measuring tape |
| **Frame Material** | - | Primary structural material | - | Carbon fiber | LOW | Design specification |
| **Arm Thickness** | `t_arm` | Cross-sectional thickness of arms | m | 0.015 | LOW | Caliper |
| **Landing Gear Height** | `h_gear` | Ground clearance at rest | m | 0.20 | MEDIUM | Physical measurement |
| **Motor Mount Angle** | `θ_mount` | Motor tilt angle (if applicable) | degrees | 0 | HIGH | Protractor or CAD |
| **Center of Gravity Height** | `h_cg` | Height of CG above frame bottom | m | 0.05 | HIGH | Balance test + measurement |

**Notes:**
- Measure with battery installed and all payload mounted
- If CG is not centered, provide X, Y, Z offsets from geometric center

---

## 2. Motor & Propeller Parameters

**Responsible Team:** Aerodynamic Design Team + Integration Team

### Motor Specifications

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Motor Model** | - | Manufacturer and model number | - | T-Motor U8 Pro | CRITICAL | Part number |
| **Motor Kv Rating** | `Kv` | RPM per volt (unloaded) | RPM/V | 170 | CRITICAL | Datasheet or test |
| **Motor Resistance** | `R_motor` | Phase-to-phase resistance | Ω | 0.08 | HIGH | Multimeter measurement |
| **Motor Inductance** | `L_motor` | Phase inductance | mH | 0.05 | MEDIUM | LCR meter or datasheet |
| **No-load Current** | `I_0` | Current draw at zero thrust | A | 0.8 | MEDIUM | Motor test stand |
| **Max Current** | `I_max` | Maximum continuous current | A | 40 | CRITICAL | Datasheet |
| **Motor Weight** | `m_motor` | Mass of single motor | kg | 0.280 | MEDIUM | Scale |
| **Number of Poles** | `N_poles` | Magnetic pole count | - | 28 | LOW | Datasheet |
| **Stator Diameter** | `d_stator` | Outer stator diameter | mm | 80 | LOW | Caliper or datasheet |

### Propeller Specifications

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Propeller Model** | - | Manufacturer and model | - | T-Motor 24x8.2 CF | CRITICAL | Part number |
| **Propeller Diameter** | `D_prop` | Tip-to-tip diameter | inches / m | 24" / 0.61 | CRITICAL | Measuring tape |
| **Propeller Pitch** | `P_prop` | Theoretical advance per revolution | inches / m | 8.2" / 0.21 | CRITICAL | Specification |
| **Number of Blades** | `N_blades` | Blades per propeller | - | 2 | HIGH | Visual count |
| **Blade Chord** | `c_blade` | Average blade width | m | 0.040 | MEDIUM | Caliper |
| **Propeller Material** | - | Construction material | - | Carbon fiber | LOW | Specification |
| **Propeller Weight** | `m_prop` | Mass of single propeller | kg | 0.045 | MEDIUM | Scale |
| **Rotation Direction** | - | CW or CCW per motor position | - | Front-right: CW | HIGH | Visual or datasheet |
| **Disc Area** | `A_disc` | Total swept area of all 4 props | m² | 1.17 | HIGH | Calculate: 4 × π(D/2)² |

### Motor-Propeller Performance Data

**CRITICAL - Must be obtained from thrust stand testing:**

| Parameter | Symbol | Description | Unit | Priority | How to Obtain |
|-----------|--------|-------------|------|----------|---------------|
| **Thrust Coefficient** | `C_T` or `k_t` | Thrust per (rad/s)² | N/(rad/s)² | CRITICAL | Thrust stand measurement |
| **Torque Coefficient** | `C_Q` or `k_q` | Torque per (rad/s)² | Nm/(rad/s)² | CRITICAL | Thrust stand measurement |
| **Thrust Curve Data** | - | Thrust vs throttle % (0-100%) | N vs % | CRITICAL | Thrust stand test |
| **Power Curve Data** | - | Power vs throttle % | W vs % | HIGH | Thrust stand test |
| **RPM Curve Data** | - | RPM vs throttle % | RPM vs % | HIGH | Thrust stand + tachometer |
| **Max Static Thrust** | `T_max` | Maximum thrust per motor (static) | N | CRITICAL | Thrust stand maximum |
| **Hover Thrust** | `T_hover` | Estimated hover thrust per motor | N | CRITICAL | Calculate: (m_total × 9.81) / 4 |
| **Thrust-to-Weight Ratio** | `TWR` | Total max thrust / weight | - | HIGH | Calculate: (4 × T_max) / (m_total × 9.81) |

**IMPORTANT:** Thrust stand testing should be conducted with:
- Actual motor + propeller combination to be used
- Multiple data points (0%, 25%, 50%, 75%, 100% throttle)
- Measurements of thrust, current, voltage, RPM
- Minimum 3 runs per data point for averaging

### Thrust Expo Parameter

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Motor Thrust Expo** | `MOT_THST_EXPO` | Thrust curve shape (0=linear, 1=quadratic) | 0-1 | 0.65-0.75 | CRITICAL | Curve fitting from thrust stand data |

**Calculation:**
```
If thrust curve is linear: MOT_THST_EXPO = 0
If thrust curve is quadratic (typical): MOT_THST_EXPO = 0.5
If thrust curve is between: Fit polynomial and determine expo
```

---

## 3. Electronic Speed Controller (ESC) Parameters

**Responsible Team:** Integration Team

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **ESC Model** | - | Manufacturer and model number | - | T-Motor Flame 60A | CRITICAL | Part number |
| **ESC Protocol** | - | Communication protocol | - | DShot600 | CRITICAL | Configuration/datasheet |
| **PWM Frequency** | `f_PWM` | ESC PWM output frequency | Hz | 48000 | HIGH | Datasheet |
| **Update Rate** | `f_update` | Command update frequency | Hz | 400 | CRITICAL | ArduPilot setting |
| **Min Throttle PWM** | `PWM_min` | Minimum PWM value | μs | 1000 | HIGH | ESC calibration |
| **Max Throttle PWM** | `PWM_max` | Maximum PWM value | μs | 2000 | HIGH | ESC calibration |
| **ESC Latency** | `τ_ESC` | Response delay | ms | 5-10 | MEDIUM | Oscilloscope test (optional) |
| **Max Continuous Current** | `I_ESC_max` | Per ESC | A | 60 | HIGH | Datasheet |
| **Burst Current** | `I_ESC_burst` | Short duration max | A | 80 | MEDIUM | Datasheet |
| **ESC Weight** | `m_ESC` | Mass per ESC | kg | 0.025 | LOW | Scale |
| **Number of ESCs** | - | Total ESC count | - | 4 | - | Configuration |

---

## 4. Battery & Power System Parameters

**Responsible Team:** Integration Team

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Battery Type** | - | Chemistry type | - | LiPo | CRITICAL | Specification |
| **Cell Count** | `N_cells` | Number of cells in series | S | 12 (12S) | CRITICAL | Battery label |
| **Nominal Voltage** | `V_nom` | Nominal cell voltage × cells | V | 44.4 (3.7V × 12) | CRITICAL | Specification |
| **Max Voltage** | `V_max` | Fully charged voltage | V | 50.4 (4.2V × 12) | CRITICAL | Specification |
| **Min Safe Voltage** | `V_min` | Minimum safe voltage | V | 42.0 (3.5V × 12) | CRITICAL | Specification |
| **Battery Capacity** | `Q_bat` | Total capacity | mAh / Ah | 22000 / 22 | CRITICAL | Battery label |
| **C-Rating** | `C_rate` | Continuous discharge rating | C | 25 | HIGH | Specification |
| **Max Discharge Current** | `I_bat_max` | Maximum continuous discharge | A | 550 (25C × 22Ah) | HIGH | Calculate or spec |
| **Battery Weight** | `m_bat` | Mass of battery | kg | 3.2 | HIGH | Scale |
| **Battery Internal Resistance** | `R_bat` | Total internal resistance | mΩ | 10 | MEDIUM | Battery tester or datasheet |
| **Battery Dimensions** | `L×W×H` | Physical size | mm | 200×80×60 | LOW | Caliper |
| **Battery Configuration** | - | Series/parallel arrangement | - | 12S1P | MEDIUM | Configuration |

**Notes:**
- Voltage sag under load affects performance - provide measured voltage at hover if available
- Battery placement affects CG - coordinate with airframe measurements

---

## 5. Sensor & IMU Parameters

**Responsible Team:** Integration Team

### Inertial Measurement Unit (IMU)

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **IMU Model** | - | Sensor chip model | - | ICM-42688-P | HIGH | Flight controller spec |
| **Gyro Sample Rate** | `f_gyro` | Gyroscope sampling frequency | Hz | 8000 | HIGH | Datasheet |
| **Accel Sample Rate** | `f_accel` | Accelerometer sampling frequency | Hz | 4000 | MEDIUM | Datasheet |
| **Gyro Noise Density** | `σ_gyro` | Noise specification | deg/s/√Hz | 0.004 | MEDIUM | Datasheet |
| **Accel Noise Density** | `σ_accel` | Noise specification | m/s²/√Hz | 0.15 | MEDIUM | Datasheet |
| **Gyro Range** | - | Maximum measurement range | deg/s | ±2000 | LOW | Configuration |
| **Accel Range** | - | Maximum measurement range | g | ±16 | LOW | Configuration |
| **IMU Location** | `x,y,z_IMU` | Position relative to CG | m | 0, 0, 0.02 | MEDIUM | Measurement from CAD |

### GPS/GNSS (if installed)

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **GPS Model** | - | GPS receiver model | - | Ublox M9N | LOW | Specification |
| **GPS Update Rate** | `f_GPS` | Position update frequency | Hz | 10 | LOW | Configuration |
| **GPS Antenna Location** | `x,y,z_GPS` | Position relative to CG | m | 0, 0, 0.15 | LOW | Measurement |

### Barometer

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Baro Model** | - | Barometer chip model | - | MS5611 | LOW | Flight controller spec |
| **Altitude Resolution** | - | Measurement precision | m | 0.1 | LOW | Datasheet |

---

## 6. Aerodynamic Parameters

**Responsible Team:** Aerodynamic Design Team

### Drag Coefficients

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Drag Coefficient (Body)** | `C_D` | Frontal drag coefficient | - | 0.8-1.2 | HIGH | Wind tunnel or CFD |
| **Frontal Area** | `A_front` | Frontal projected area | m² | 0.15 | HIGH | CAD projection |
| **Side Area** | `A_side` | Side projected area | m² | 0.10 | MEDIUM | CAD projection |
| **Vertical Drag Coefficient** | `C_D_z` | Vertical drag | - | 1.0 | MEDIUM | Estimate or test |
| **Induced Drag Factor** | `k_i` | Induced drag coefficient | - | 0.1 | LOW | Estimate |

### Rotor/Propeller Interactions

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Rotor Downwash Velocity** | `v_i` | Induced velocity in hover | m/s | 8.5 | MEDIUM | Calculate: √(T/(2ρA)) |
| **Blade Flapping Angle** | `β_flap` | Maximum blade flap angle | degrees | 2-5 | LOW | Observation or calculation |
| **Rotor Interference Factor** | `k_int` | Efficiency loss due to rotor interaction | - | 0.95 | MEDIUM | Empirical or CFD |

### Environmental Coefficients

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Obtain |
|-----------|--------|-------------|------|---------------|----------|---------------|
| **Ground Effect Height** | `h_GE` | Height below which ground effect applies | m | 0.5-1.0 | LOW | Estimate: ~1× rotor diameter |
| **Ground Effect Factor** | `k_GE` | Thrust increase near ground | - | 1.1-1.2 | LOW | Literature or test |

---

## 7. Mass Distribution & Inertia

**Responsible Team:** Integration Team + Structural Analysis

### Moments of Inertia (CRITICAL)

| Parameter | Symbol | Description | Unit | Example Value | Priority | How to Measure |
|-----------|--------|-------------|------|---------------|----------|----------------|
| **Roll Inertia** | `I_xx` | Moment of inertia about X-axis (roll) | kg·m² | 0.30 | CRITICAL | Bifilar pendulum test or CAD |
| **Pitch Inertia** | `I_yy` | Moment of inertia about Y-axis (pitch) | kg·m² | 0.30 | CRITICAL | Bifilar pendulum test or CAD |
| **Yaw Inertia** | `I_zz` | Moment of inertia about Z-axis (yaw) | kg·m² | 0.50 | CRITICAL | Bifilar pendulum test or CAD |
| **Cross-Product Inertia** | `I_xy, I_xz, I_yz` | Product of inertia terms | kg·m² | ~0 (if symmetric) | MEDIUM | CAD or assume negligible |

**How to Measure Inertia (Bifilar Pendulum Method):**

1. Suspend drone from two parallel strings of equal length
2. Twist drone and measure oscillation period
3. Calculate inertia: `I = (m × g × L² × T²) / (16 × π² × d)`
   - `m` = mass (kg)
   - `g` = 9.81 m/s²
   - `L` = string length (m)
   - `T` = oscillation period (s)
   - `d` = distance between suspension points (m)

**Alternative:** Use CAD software (SolidWorks, CATIA) with accurate component masses

### Component Mass Breakdown

| Component | Mass (kg) | Priority | Notes |
|-----------|-----------|----------|-------|
| Frame structure | ___ | HIGH | Including arms, center plates |
| Motors (4×) | ___ | HIGH | Should match motor specification |
| Propellers (4×) | ___ | MEDIUM | Should match propeller specification |
| ESCs (4×) | ___ | MEDIUM | Should match ESC specification |
| Flight controller | ___ | LOW | - |
| Battery | ___ | CRITICAL | Should match battery specification |
| Wiring & connectors | ___ | LOW | - |
| Payload (if any) | ___ | MEDIUM | Camera, gimbal, etc. |
| **Total** | **30.0** | CRITICAL | Must sum to total vehicle mass |

---

## 8. Environmental & Operational Parameters

**Responsible Team:** Integration Team / Operations

| Parameter | Symbol | Description | Unit | Example Value | Priority | Notes |
|-----------|--------|-------------|------|---------------|----------|-------|
| **Operating Altitude** | `h_op` | Typical flight altitude AGL | m | 0-100 | MEDIUM | Mission profile |
| **Max Wind Speed** | `v_wind_max` | Maximum operational wind | m/s | 10 | MEDIUM | Safety limits |
| **Operating Temperature** | `T_op` | Temperature range | °C | -10 to 40 | LOW | Environmental spec |
| **Air Density** | `ρ` | Operating air density | kg/m³ | 1.225 (sea level) | MEDIUM | Altitude-dependent |
| **Expected Payload** | `m_payload` | Typical payload mass | kg | 0-5 | MEDIUM | Mission requirement |

---

## 9. Control System Baseline

**Responsible Team:** Integration Team

### Current ArduPilot Configuration

If the drone has been manually tuned or has existing parameters, provide:

| Parameter Group | Description | Priority | File/Method |
|-----------------|-------------|----------|-------------|
| **Current .param file** | Existing ArduPilot parameters | HIGH | Export from Mission Planner/QGC |
| **Flight logs** | Recent flight log files | MEDIUM | .bin or .log files |
| **Tuning notes** | Manual tuning observations | LOW | Documentation if available |

### Rate Controller Limits (if known)

| Parameter | Symbol | Description | Unit | Example Value | Priority |
|-----------|--------|-------------|------|---------------|----------|
| **Max Roll Rate** | `ω_roll_max` | Maximum desired roll rate | deg/s | 360 | HIGH |
| **Max Pitch Rate** | `ω_pitch_max` | Maximum desired pitch rate | deg/s | 360 | HIGH |
| **Max Yaw Rate** | `ω_yaw_max` | Maximum desired yaw rate | deg/s | 180 | HIGH |
| **Max Tilt Angle** | `θ_max` | Maximum allowed tilt | degrees | 45 | HIGH |

---

## 10. Data Collection Guidelines

### Critical Measurements

**Priority 1 - MUST HAVE (Blocks optimization):**
- [ ] Total vehicle mass (fully assembled with battery)
- [ ] Arm length (center to motor axis)
- [ ] Motor Kv rating
- [ ] Propeller diameter and pitch
- [ ] Thrust curve data (thrust vs throttle %, minimum 5 points)
- [ ] Max thrust per motor
- [ ] Battery voltage (max/min)
- [ ] Battery capacity
- [ ] Moments of inertia (Ixx, Iyy, Izz)

**Priority 2 - SHOULD HAVE (Affects accuracy):**
- [ ] Motor resistance and inductance
- [ ] Thrust and torque coefficients (from thrust stand)
- [ ] ESC update rate and protocol
- [ ] Drag coefficients
- [ ] Component mass breakdown
- [ ] IMU specifications
- [ ] RPM vs throttle curve

**Priority 3 - NICE TO HAVE (Refinement):**
- [ ] ESC latency
- [ ] Ground effect measurements
- [ ] Temperature effects on performance
- [ ] Battery internal resistance
- [ ] Vibration analysis data

### Measurement Tools Needed

**Essential:**
- Calibrated scale (0-50 kg, 1g precision)
- Measuring tape / caliper (0-2m range, 0.1mm precision)
- Thrust stand with load cell (0-200N range)
- Multimeter (resistance, continuity)
- Tachometer (optical RPM measurement)

**Recommended:**
- CAD software with mass properties tool
- Bifilar pendulum setup for inertia measurement
- Wind tunnel access (or CFD simulation)
- Current/voltage logger
- Oscilloscope (for ESC timing)

### Data Format Requirements

Please provide data in the following formats:

1. **Specifications:** Excel spreadsheet or CSV with all parameters
2. **Thrust stand data:** CSV with columns: throttle_%, thrust_N, current_A, voltage_V, rpm
3. **CAD models:** STEP or IGES format with accurate masses assigned
4. **Drawings:** PDF with dimensions and CG location marked
5. **Photos:** High-resolution images showing component placement

---

## Data Submission Template

### Team: [Aerodynamic Design / Integration]
### Date: ___________
### Completed by: ___________

**Checklist:**

| Section | Status | Comments |
|---------|--------|----------|
| 1. Airframe & Structural | ☐ Complete ☐ Partial ☐ Not Started | |
| 2. Motor & Propeller | ☐ Complete ☐ Partial ☐ Not Started | |
| 3. ESC Parameters | ☐ Complete ☐ Partial ☐ Not Started | |
| 4. Battery & Power | ☐ Complete ☐ Partial ☐ Not Started | |
| 5. Sensor & IMU | ☐ Complete ☐ Partial ☐ Not Started | |
| 6. Aerodynamic | ☐ Complete ☐ Partial ☐ Not Started | |
| 7. Mass & Inertia | ☐ Complete ☐ Partial ☐ Not Started | |
| 8. Environmental | ☐ Complete ☐ Partial ☐ Not Started | |
| 9. Control Baseline | ☐ Complete ☐ Partial ☐ Not Started | |

**Attached Files:**
- [ ] Specifications spreadsheet
- [ ] Thrust stand data
- [ ] CAD model
- [ ] Dimensional drawings
- [ ] Photos of assembly
- [ ] Current .param file (if available)
- [ ] Flight logs (if available)

**Known Uncertainties / Estimates:**
_List any parameters that are estimated rather than measured, and explain why:_

---

## Contact Information

**For Questions or Clarifications:**

**Optimization Team Lead:** ___________
**Email:** ___________
**Phone:** ___________

**Data Submission:**
- Email to: ___________
- Upload to: ___________
- Deadline: ___________

---

## Appendix A: Quick Reference - Minimum Required Parameters

If time is extremely limited, provide AT MINIMUM these 15 parameters:

1. `m_total` - Total vehicle mass (kg)
2. `L_arm` - Arm length (m)
3. `I_xx` - Roll inertia (kg·m²)
4. `I_yy` - Pitch inertia (kg·m²)
5. `I_zz` - Yaw inertia (kg·m²)
6. Motor Kv rating (RPM/V)
7. `D_prop` - Propeller diameter (m or inches)
8. `P_prop` - Propeller pitch (m or inches)
9. `T_max` - Max thrust per motor (N)
10. `V_max` - Battery max voltage (V)
11. `V_min` - Battery min voltage (V)
12. `Q_bat` - Battery capacity (Ah)
13. `C_D` - Drag coefficient (estimate 1.0 if unknown)
14. `A_front` - Frontal area (m²)
15. Thrust vs throttle curve (minimum 5 data points)

**With these 15 parameters, optimization can proceed with reasonable accuracy. All other parameters will use conservative estimates.**

---

## Appendix B: Equations & Calculations

### Thrust Coefficient Calculation
From thrust stand data:
```
k_t = T / ω²
where:
  T = thrust (N)
  ω = angular velocity (rad/s) = (RPM × 2π) / 60
```

### Torque Coefficient Calculation
From power and thrust data:
```
k_q = P / ω³
where:
  P = power (W) = V × I
  ω = angular velocity (rad/s)
```

### Hover Thrust Calculation
```
T_hover_per_motor = (m_total × g) / 4
T_hover_per_motor = (30 kg × 9.81 m/s²) / 4 = 73.6 N
```

### Thrust-to-Weight Ratio
```
TWR = (4 × T_max) / (m_total × g)
Minimum recommended: TWR > 2.0 for agile flight
```

### Moment of Inertia (Point Mass Approximation)
For quick estimation:
```
I_xx ≈ I_yy ≈ 4 × m_motor × L_arm²
I_zz ≈ 8 × m_motor × L_arm²
```
(More accurate methods required for final model)

---

## Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-03 | Initial release | MC07 Optimization Team |

---

**END OF DOCUMENT**
