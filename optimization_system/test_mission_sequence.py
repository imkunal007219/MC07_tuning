#!/usr/bin/env python3
"""
Test the correct mission execution sequence.

This follows the user's verified working sequence:
1. Load waypoints
2. Wait 20 seconds for sensors
3. Set mode AUTO
4. Arm throttle
5. Send RC 3 1500
"""

import sys
import os
import time
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sitl_manager import SITLManager
from mission_executor import MissionExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_mission_sequence():
    """Test mission execution with correct sequence."""
    logger.info("="*60)
    logger.info("TESTING MISSION EXECUTION SEQUENCE")
    logger.info("="*60)

    # Initialize SITL manager with single instance
    sitl_manager = SITLManager(
        num_instances=1,
        speedup=1,  # Real-time for better observation
        show_console=True  # Show MAVProxy console
    )

    try:
        # Start SITL
        logger.info("\nStarting SITL instance...")
        instance_id = sitl_manager.get_instance(timeout=60)

        if instance_id is None:
            logger.error("Failed to start SITL instance")
            return False

        logger.info(f"✓ SITL instance {instance_id} started")

        # Get connection
        instance = sitl_manager.instances[instance_id]
        connection = instance.connection

        if connection is None:
            logger.error("No connection available")
            return False

        logger.info("✓ MAVLink connection established")

        # Create mission executor
        executor = MissionExecutor(connection)

        # Get mission file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mission_file = os.path.join(script_dir, "missions", "simple_hover.waypoints")

        logger.info(f"\nMission file: {mission_file}")

        if not os.path.exists(mission_file):
            logger.error(f"Mission file not found: {mission_file}")
            return False

        # Run mission with updated sequence
        logger.info("\n" + "="*60)
        logger.info("EXECUTING MISSION (updated sequence)")
        logger.info("="*60)
        logger.info("Sequence:")
        logger.info("  1. Set AUTO_OPTIONS=1, ARMING_CHECK=0")
        logger.info("  2. Load waypoints")
        logger.info("  3. Wait 20 seconds for sensors")
        logger.info("  4. Set mode AUTO")
        logger.info("  5. Arm throttle")
        logger.info("  6. Send RC 3 1500")
        logger.info("="*60 + "\n")

        success, telemetry = executor.run_mission(mission_file, timeout=90)

        logger.info("\n" + "="*60)
        if success:
            logger.info("✓✓✓ MISSION COMPLETED SUCCESSFULLY! ✓✓✓")
            logger.info(f"Flight duration: {telemetry.get('metrics', {}).get('duration', 0):.1f}s")
            logger.info(f"Max altitude: {telemetry.get('metrics', {}).get('max_altitude', 0):.2f}m")
        else:
            logger.error("✗✗✗ MISSION FAILED ✗✗✗")

        logger.info("="*60)

        return success

    except KeyboardInterrupt:
        logger.warning("\n\nTest interrupted by user (Ctrl+C)")
        return False

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        sitl_manager.cleanup()
        logger.info("✓ Cleanup complete")


def main():
    """Run the test."""
    print("\n" + "="*60)
    print("MISSION EXECUTION SEQUENCE TEST")
    print("="*60)
    print("\nThis test verifies the correct mission execution sequence:")
    print("  1. Set AUTO_OPTIONS=1, ARMING_CHECK=0")
    print("  2. Load waypoints")
    print("  3. Wait 20 seconds for sensors")
    print("  4. Set mode AUTO")
    print("  5. Arm throttle")
    print("  6. Send RC 3 1500")
    print("\nPress Ctrl+C to abort at any time")
    print("="*60 + "\n")

    # Source ArduPilot environment
    logger.info("Make sure you've sourced ArduPilot environment:")
    logger.info("  . ~/.profile")
    logger.info("")

    time.sleep(2)

    # Run test
    success = test_mission_sequence()

    print("\n" + "="*60)
    if success:
        print("TEST RESULT: ✓ SUCCESS")
        print("="*60)
        print("\nThe mission executed successfully with the correct sequence!")
        print("You should have seen:")
        print("  1. Parameters set (AUTO_OPTIONS, ARMING_CHECK)")
        print("  2. Waypoints loaded")
        print("  3. 20-second sensor wait")
        print("  4. Mode changed to AUTO")
        print("  5. Vehicle armed")
        print("  6. RC override sent")
        print("  7. Mission executed and completed")
        return 0
    else:
        print("TEST RESULT: ✗ FAILED")
        print("="*60)
        print("\nPlease check the logs above for error details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
