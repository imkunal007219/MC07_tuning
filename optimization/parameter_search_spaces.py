#!/usr/bin/env python3
"""
Parameter Search Space Definitions for 30kg Drone PID Optimization

This module defines the search spaces for all ArduPilot parameters to be optimized.
Bounds are calculated based on:
- Measured inertia values (I_xx=2.78, I_yy=4.88, I_zz=7.18 kg·m²)
- Total mass (30 kg)
- Motor/propeller performance characteristics
- Safety constraints

Author: MC07 Tuning System
Date: 2025-11-05
"""

import numpy as np
from typing import Dict, Any, List


class ParameterSearchSpace:
    """
    Defines parameter search spaces for hierarchical optimization
    """

    def __init__(self):
        # Drone physical properties
        self.mass = 30.0  # kg
        self.I_xx = 2.78  # kg·m² (Roll inertia)
        self.I_yy = 4.88  # kg·m² (Pitch inertia)
        self.I_zz = 7.18  # kg·m² (Yaw inertia)
        self.arm_length = 0.75  # m
        self.twr = 3.2  # Thrust-to-weight ratio

        # Calculate baseline P gains using rule of thumb
        self.baseline_roll_p = 0.12
        self.baseline_pitch_p = 0.10
        self.baseline_yaw_p = 0.20

    def get_stage_1_rate_controller_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Stage 1: Rate Controller Parameters (Inner Loop - MOST CRITICAL)

        These control the angular velocity (gyro rate) response.
        Most important for stability.

        Returns:
            Dictionary of parameter search spaces
        """

        return {
            # ========================================
            # ROLL RATE CONTROLLER (I_xx = 2.78 kg·m²)
            # ========================================
            'ATC_RAT_RLL_P': {
                'min': 0.05,
                'max': 0.25,
                'default': self.baseline_roll_p,
                'type': 'float',
                'description': 'Roll rate P gain - inversely proportional to roll inertia',
                'unit': '-',
                'priority': 'CRITICAL'
            },
            'ATC_RAT_RLL_I': {
                'min': 0.04,
                'max': 0.20,
                'default': 0.10,
                'type': 'float',
                'description': 'Roll rate I gain - removes steady-state error',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_RAT_RLL_D': {
                'min': 0.001,
                'max': 0.012,
                'default': 0.005,
                'type': 'float',
                'description': 'Roll rate D gain - damping term',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_RAT_RLL_FLTD': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Roll rate D-term low-pass filter frequency',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },
            'ATC_RAT_RLL_FLTT': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Roll rate target low-pass filter frequency',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },

            # ========================================
            # PITCH RATE CONTROLLER (I_yy = 4.88 kg·m²)
            # Higher inertia than roll - needs more torque
            # ========================================
            'ATC_RAT_PIT_P': {
                'min': 0.04,
                'max': 0.20,
                'default': self.baseline_pitch_p,
                'type': 'float',
                'description': 'Pitch rate P gain - lower than roll due to higher inertia',
                'unit': '-',
                'priority': 'CRITICAL'
            },
            'ATC_RAT_PIT_I': {
                'min': 0.03,
                'max': 0.16,
                'default': 0.08,
                'type': 'float',
                'description': 'Pitch rate I gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_RAT_PIT_D': {
                'min': 0.001,
                'max': 0.010,
                'default': 0.004,
                'type': 'float',
                'description': 'Pitch rate D gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_RAT_PIT_FLTD': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Pitch rate D-term filter frequency',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },
            'ATC_RAT_PIT_FLTT': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Pitch rate target filter frequency',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },

            # ========================================
            # YAW RATE CONTROLLER (I_zz = 7.18 kg·m²)
            # Highest inertia - needs highest gains
            # ========================================
            'ATC_RAT_YAW_P': {
                'min': 0.10,
                'max': 0.40,
                'default': self.baseline_yaw_p,
                'type': 'float',
                'description': 'Yaw rate P gain - highest due to highest inertia',
                'unit': '-',
                'priority': 'CRITICAL'
            },
            'ATC_RAT_YAW_I': {
                'min': 0.01,
                'max': 0.08,
                'default': 0.02,
                'type': 'float',
                'description': 'Yaw rate I gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_RAT_YAW_FLTD': {
                'min': 2.0,
                'max': 10.0,
                'default': 5.0,
                'type': 'float',
                'description': 'Yaw rate D-term filter (usually lower than roll/pitch)',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },
            'ATC_RAT_YAW_FLTE': {
                'min': 1.0,
                'max': 5.0,
                'default': 2.0,
                'type': 'float',
                'description': 'Yaw rate error filter',
                'unit': 'Hz',
                'priority': 'LOW'
            },
            'ATC_RAT_YAW_FLTT': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Yaw rate target filter',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },

            # ========================================
            # SENSOR FILTERING
            # ========================================
            'INS_GYRO_FILTER': {
                'min': 40.0,
                'max': 120.0,
                'default': 80.0,
                'type': 'float',
                'description': 'Gyroscope low-pass filter frequency',
                'unit': 'Hz',
                'priority': 'HIGH'
            },
            'INS_ACCEL_FILTER': {
                'min': 10.0,
                'max': 40.0,
                'default': 20.0,
                'type': 'float',
                'description': 'Accelerometer low-pass filter frequency',
                'unit': 'Hz',
                'priority': 'MEDIUM'
            },

            # ========================================
            # RATE LIMITS
            # ========================================
            'ATC_ACCEL_R_MAX': {
                'min': 80000.0,
                'max': 150000.0,
                'default': 110000.0,
                'type': 'float',
                'description': 'Maximum roll acceleration',
                'unit': 'deg/s²',
                'priority': 'MEDIUM'
            },
            'ATC_ACCEL_P_MAX': {
                'min': 80000.0,
                'max': 150000.0,
                'default': 110000.0,
                'type': 'float',
                'description': 'Maximum pitch acceleration',
                'unit': 'deg/s²',
                'priority': 'MEDIUM'
            },
            'ATC_ACCEL_Y_MAX': {
                'min': 20000.0,
                'max': 40000.0,
                'default': 27000.0,
                'type': 'float',
                'description': 'Maximum yaw acceleration',
                'unit': 'deg/s²',
                'priority': 'MEDIUM'
            },
        }

    def get_stage_2_attitude_controller_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Stage 2: Attitude Controller Parameters (Middle Loop)

        These control the angle (attitude) response.
        Depends on stable rate controllers from Stage 1.

        Returns:
            Dictionary of parameter search spaces
        """

        return {
            # ========================================
            # ANGLE CONTROLLERS
            # ========================================
            'ATC_ANG_RLL_P': {
                'min': 3.0,
                'max': 8.0,
                'default': 4.5,
                'type': 'float',
                'description': 'Roll angle P gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_ANG_PIT_P': {
                'min': 3.0,
                'max': 8.0,
                'default': 4.5,
                'type': 'float',
                'description': 'Pitch angle P gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'ATC_ANG_YAW_P': {
                'min': 3.0,
                'max': 8.0,
                'default': 4.5,
                'type': 'float',
                'description': 'Yaw angle P gain',
                'unit': '-',
                'priority': 'HIGH'
            },

            # ========================================
            # INPUT SHAPING
            # ========================================
            'ATC_INPUT_TC': {
                'min': 0.1,
                'max': 0.3,
                'default': 0.15,
                'type': 'float',
                'description': 'Attitude input time constant (smoothing)',
                'unit': 's',
                'priority': 'MEDIUM'
            },
            'ATC_SLEW_YAW': {
                'min': 4000.0,
                'max': 10000.0,
                'default': 6000.0,
                'type': 'float',
                'description': 'Yaw target slew rate limit',
                'unit': 'deg/s',
                'priority': 'LOW'
            },
        }

    def get_stage_3_position_controller_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Stage 3: Position Controller Parameters (Outer Loop)

        These control position and velocity tracking.
        Depends on stable rate and attitude controllers.

        Returns:
            Dictionary of parameter search spaces
        """

        return {
            # ========================================
            # HORIZONTAL POSITION CONTROLLER
            # ========================================
            'PSC_POSXY_P': {
                'min': 0.5,
                'max': 2.0,
                'default': 1.0,
                'type': 'float',
                'description': 'Horizontal position P gain',
                'unit': '-',
                'priority': 'HIGH'
            },

            # ========================================
            # HORIZONTAL VELOCITY CONTROLLER
            # ========================================
            'PSC_VELXY_P': {
                'min': 0.5,
                'max': 2.0,
                'default': 0.9,
                'type': 'float',
                'description': 'Horizontal velocity P gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'PSC_VELXY_I': {
                'min': 0.2,
                'max': 1.0,
                'default': 0.45,
                'type': 'float',
                'description': 'Horizontal velocity I gain',
                'unit': '-',
                'priority': 'MEDIUM'
            },
            'PSC_VELXY_D': {
                'min': 0.05,
                'max': 0.40,
                'default': 0.18,
                'type': 'float',
                'description': 'Horizontal velocity D gain',
                'unit': '-',
                'priority': 'MEDIUM'
            },
            'PSC_VELXY_FLTD': {
                'min': 2.0,
                'max': 10.0,
                'default': 5.0,
                'type': 'float',
                'description': 'Horizontal velocity D-term filter',
                'unit': 'Hz',
                'priority': 'LOW'
            },
            'PSC_VELXY_FLTE': {
                'min': 2.0,
                'max': 10.0,
                'default': 5.0,
                'type': 'float',
                'description': 'Horizontal velocity error filter',
                'unit': 'Hz',
                'priority': 'LOW'
            },

            # ========================================
            # VERTICAL POSITION CONTROLLER
            # ========================================
            'PSC_POSZ_P': {
                'min': 0.5,
                'max': 2.0,
                'default': 1.0,
                'type': 'float',
                'description': 'Vertical position P gain',
                'unit': '-',
                'priority': 'HIGH'
            },

            # ========================================
            # VERTICAL VELOCITY CONTROLLER
            # ========================================
            'PSC_VELZ_P': {
                'min': 3.0,
                'max': 8.0,
                'default': 5.0,
                'type': 'float',
                'description': 'Vertical velocity P gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'PSC_VELZ_FLTE': {
                'min': 2.0,
                'max': 10.0,
                'default': 5.0,
                'type': 'float',
                'description': 'Vertical velocity error filter',
                'unit': 'Hz',
                'priority': 'LOW'
            },

            # ========================================
            # VERTICAL ACCELERATION CONTROLLER
            # ========================================
            'PSC_ACCZ_P': {
                'min': 0.15,
                'max': 0.50,
                'default': 0.25,
                'type': 'float',
                'description': 'Vertical acceleration P gain',
                'unit': '-',
                'priority': 'HIGH'
            },
            'PSC_ACCZ_I': {
                'min': 0.20,
                'max': 1.00,
                'default': 0.50,
                'type': 'float',
                'description': 'Vertical acceleration I gain',
                'unit': '-',
                'priority': 'MEDIUM'
            },
            'PSC_ACCZ_D': {
                'min': 0.00,
                'max': 0.02,
                'default': 0.00,
                'type': 'float',
                'description': 'Vertical acceleration D gain (usually zero)',
                'unit': '-',
                'priority': 'LOW'
            },
        }

    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all parameters across all stages

        Returns:
            Combined dictionary of all parameters
        """
        all_params = {}
        all_params.update(self.get_stage_1_rate_controller_space())
        all_params.update(self.get_stage_2_attitude_controller_space())
        all_params.update(self.get_stage_3_position_controller_space())
        return all_params

    def get_parameter_names_by_stage(self, stage: int) -> List[str]:
        """
        Get list of parameter names for a specific stage

        Args:
            stage: Stage number (1, 2, or 3)

        Returns:
            List of parameter names
        """
        if stage == 1:
            return list(self.get_stage_1_rate_controller_space().keys())
        elif stage == 2:
            return list(self.get_stage_2_attitude_controller_space().keys())
        elif stage == 3:
            return list(self.get_stage_3_position_controller_space().keys())
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3")

    def print_search_space_summary(self, stage: int = None):
        """
        Print a summary of the search space

        Args:
            stage: Optional stage number (1-3). If None, prints all stages.
        """
        if stage is None:
            stages = [1, 2, 3]
        else:
            stages = [stage]

        for s in stages:
            if s == 1:
                space = self.get_stage_1_rate_controller_space()
                title = "STAGE 1: RATE CONTROLLER PARAMETERS"
            elif s == 2:
                space = self.get_stage_2_attitude_controller_space()
                title = "STAGE 2: ATTITUDE CONTROLLER PARAMETERS"
            else:
                space = self.get_stage_3_position_controller_space()
                title = "STAGE 3: POSITION CONTROLLER PARAMETERS"

            print(f"\n{'='*70}")
            print(f"{title}")
            print(f"{'='*70}")
            print(f"Total parameters: {len(space)}\n")

            for param_name, param_info in space.items():
                print(f"Parameter: {param_name}")
                print(f"  Range: [{param_info['min']:.4f}, {param_info['max']:.4f}]")
                print(f"  Default: {param_info['default']:.4f}")
                print(f"  Priority: {param_info['priority']}")
                print(f"  Description: {param_info['description']}")
                print()


if __name__ == "__main__":
    # Demonstration
    search_space = ParameterSearchSpace()

    print("\n" + "="*70)
    print("PARAMETER SEARCH SPACE DEFINITION FOR 30KG DRONE")
    print("="*70)
    print(f"\nDrone Properties:")
    print(f"  Mass: {search_space.mass} kg")
    print(f"  I_xx (Roll):  {search_space.I_xx} kg·m²")
    print(f"  I_yy (Pitch): {search_space.I_yy} kg·m²")
    print(f"  I_zz (Yaw):   {search_space.I_zz} kg·m²")
    print(f"  Arm length: {search_space.arm_length} m")
    print(f"  Thrust-to-Weight Ratio: {search_space.twr}")

    # Print all stages
    search_space.print_search_space_summary()

    # Summary
    total_params = len(search_space.get_all_parameters())
    stage1_params = len(search_space.get_stage_1_rate_controller_space())
    stage2_params = len(search_space.get_stage_2_attitude_controller_space())
    stage3_params = len(search_space.get_stage_3_position_controller_space())

    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"Total parameters to optimize: {total_params}")
    print(f"  Stage 1 (Rate):      {stage1_params} parameters")
    print(f"  Stage 2 (Attitude):  {stage2_params} parameters")
    print(f"  Stage 3 (Position):  {stage3_params} parameters")
    print("="*70 + "\n")
