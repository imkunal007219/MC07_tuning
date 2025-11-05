#!/usr/bin/env python3
"""
Simple mission test - minimal version for debugging.

Tests just the mission execution without the full optimization framework.
"""

import sys
import os
import time
import logging

# Setup logging first
logging.basicConfig(
    level=logging.DEBUG,  # More verbose
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymavlink import mavutil
from mission_executor import MissionExecutor


def test_simple_mission():
    """Test mission execution with a manually started SITL."""

    logger.info("="*60)
    logger.info("SIMPLE MISSION TEST")
    logger.info("="*60)

    # Instructions for user
    print("\n" + "="*60)
    print("MANUAL SITL SETUP REQUIRED")
    print("="*60)
    print("\nPlease open a separate terminal and run:")
    print("\n  cd ~/Documents/MC07_tuning/ardupilot/ArduCopter")
    print("  ../Tools/autotest/sim_vehicle.py -f drone-30kg --console\n")
    print("Wait for SITL to fully start (you'll see 'EKF2 IMU1 is using GPS')")
    print("="*60)

    input("\nPress ENTER when SITL is ready...")

    # Connect to SITL
    logger.info("\nConnecting to SITL on UDP:127.0.0.1:14550...")

    try:
        connection = mavutil.mavlink_connection('udp:127.0.0.1:14550', timeout=10)
        logger.info("Waiting for heartbeat...")
        connection.wait_heartbeat(timeout=10)
        logger.info(f"✓ Connected - System ID: {connection.target_system}, Component ID: {connection.target_component}")
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.error("\nMake sure SITL is running with:")
        logger.error("  cd ~/Documents/MC07_tuning/ardupilot/ArduCopter")
        logger.error("  ../Tools/autotest/sim_vehicle.py -f drone-30kg --console")
        return False

    # Get mission file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mission_file = os.path.join(script_dir, "missions", "simple_hover.waypoints")

    logger.info(f"\nMission file: {mission_file}")

    if not os.path.exists(mission_file):
        logger.error(f"✗ Mission file not found: {mission_file}")
        return False

    logger.info("✓ Mission file found")

    # Create executor
    logger.info("\nCreating mission executor...")
    executor = MissionExecutor(connection)

    # Run mission
    logger.info("\n" + "="*60)
    logger.info("EXECUTING MISSION")
    logger.info("="*60)
    logger.info("\nWatch the MAVProxy console for:")
    logger.info("  1. AUTO_OPTIONS set to 1")
    logger.info("  2. ARMING_CHECK set to 0")
    logger.info("  3. Mission items loaded")
    logger.info("  4. Mode changed to AUTO")
    logger.info("  5. Vehicle ARMED")
    logger.info("  6. Takeoff and mission execution")
    logger.info("="*60 + "\n")

    try:
        success, telemetry = executor.run_mission(mission_file, timeout=90)

        logger.info("\n" + "="*60)
        if success:
            logger.info("✓✓✓ MISSION COMPLETED SUCCESSFULLY! ✓✓✓")

            if telemetry and 'metrics' in telemetry:
                metrics = telemetry['metrics']
                logger.info(f"\nFlight metrics:")
                logger.info(f"  Duration: {metrics.get('duration', 0):.1f}s")
                logger.info(f"  Max altitude: {metrics.get('max_altitude', 0):.2f}m")
                logger.info(f"  Max roll: {metrics.get('max_roll', 0):.2f}°")
                logger.info(f"  Max pitch: {metrics.get('max_pitch', 0):.2f}°")
        else:
            logger.error("✗✗✗ MISSION FAILED ✗✗✗")
            logger.error("\nCheck the logs above and MAVProxy console for errors")

        logger.info("="*60)

        return success

    except Exception as e:
        logger.error(f"\n✗ Mission execution failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("\n" + "="*60)
    print("SIMPLE MISSION EXECUTION TEST")
    print("="*60)
    print("\nThis test requires you to manually start SITL first.")
    print("This helps isolate mission execution issues from SITL management issues.")
    print("="*60 + "\n")

    try:
        success = test_simple_mission()

        print("\n" + "="*60)
        if success:
            print("TEST RESULT: ✓ SUCCESS")
            print("="*60)
            print("\nThe mission executed successfully!")
            print("\nNext step: Test with automatic SITL management:")
            print("  python3 test_mission_sequence.py")
            return 0
        else:
            print("TEST RESULT: ✗ FAILED")
            print("="*60)
            print("\nPlease review the error messages above.")
            print("\nCommon issues:")
            print("  1. SITL not running - start it manually first")
            print("  2. Wrong port - make sure using 14550")
            print("  3. Mission file not found - check path")
            return 1

    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user (Ctrl+C)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
