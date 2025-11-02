#!/usr/bin/env python3
"""
Test script to verify SITL can start properly
"""

import sys
import os

# Add optimization_system to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'optimization_system'))

import logging
from sitl_manager import SITLManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sitl_startup():
    """Test starting a single SITL instance"""
    logger.info("Testing SITL startup...")

    try:
        # Create SITL manager with 1 instance
        sitl_manager = SITLManager(
            num_instances=1,
            speedup=1
        )

        logger.info(f"SITL Manager initialized successfully")
        logger.info(f"ArduPilot path: {sitl_manager.ardupilot_path}")

        # Test parameters
        test_params = {
            'ATC_RAT_RLL_P': 0.15,
            'ATC_RAT_PIT_P': 0.15,
            'ATC_RAT_YAW_P': 0.25,
        }

        # Try to start one instance
        logger.info("Attempting to start SITL instance 0...")
        success = sitl_manager.start_instance(0, test_params)

        if success:
            logger.info("✓ SITL instance started successfully!")

            # Clean up
            logger.info("Stopping SITL instance...")
            sitl_manager.stop_instance(0)
            logger.info("✓ SITL instance stopped")

            return True
        else:
            logger.error("✗ Failed to start SITL instance")
            return False

    except Exception as e:
        logger.error(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            sitl_manager.cleanup()
        except:
            pass

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("SITL STARTUP TEST")
    logger.info("="*60)

    success = test_sitl_startup()

    logger.info("="*60)
    if success:
        logger.info("TEST PASSED ✓")
        sys.exit(0)
    else:
        logger.info("TEST FAILED ✗")
        sys.exit(1)
