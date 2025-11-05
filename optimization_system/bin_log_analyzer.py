"""
ArduPilot .bin Log Analyzer

Extracts parameters and flight data from ArduPilot binary log files (.bin)
for verification that parameters were actually applied during flight.
"""

import os
import logging
from typing import Dict, Optional
from pymavlink import mavutil
import glob

logger = logging.getLogger(__name__)


class BinLogAnalyzer:
    """Extract parameters from ArduPilot .bin log files"""

    @staticmethod
    def find_latest_bin_log(search_dir: str) -> Optional[str]:
        """
        Find the most recent .bin log file in directory.

        Args:
            search_dir: Directory to search for .bin files

        Returns:
            Path to latest .bin file, or None if not found
        """
        pattern = os.path.join(search_dir, "*.bin")
        bin_files = glob.glob(pattern)

        if not bin_files:
            logger.warning(f"No .bin files found in {search_dir}")
            return None

        # Get most recent by modification time
        latest = max(bin_files, key=os.path.getmtime)
        logger.debug(f"Found latest .bin log: {latest}")
        return latest

    @staticmethod
    def extract_parameters(bin_file: str) -> Dict[str, float]:
        """
        Extract all parameters from ArduPilot .bin log file.

        ArduPilot logs PARM messages at start of flight containing
        all parameter values. This provides verification that parameters
        were actually applied.

        Args:
            bin_file: Path to .bin log file

        Returns:
            Dictionary of parameter_name -> value
        """
        if not os.path.exists(bin_file):
            logger.error(f".bin file not found: {bin_file}")
            return {}

        parameters = {}

        try:
            # Open .bin file with pymavlink
            mlog = mavutil.mavlink_connection(bin_file)

            logger.debug(f"Extracting parameters from {bin_file}")

            # Read all PARM messages
            # PARM messages contain parameter definitions from the flight
            msg_count = 0
            while True:
                msg = mlog.recv_match(type='PARM', blocking=False)
                if msg is None:
                    break

                msg_count += 1

                # Extract parameter name and value
                # Handle both bytes and string for parameter name
                param_name = msg.Name
                if isinstance(param_name, bytes):
                    param_name = param_name.decode('utf-8')
                param_name = param_name.rstrip('\x00')  # Remove null padding

                param_value = msg.Value

                parameters[param_name] = float(param_value)

            logger.info(f"Extracted {len(parameters)} parameters from {bin_file} ({msg_count} PARM messages)")

            # Also check for PARAM_VALUE messages (alternative message type)
            if len(parameters) == 0:
                logger.debug("No PARM messages found, trying PARAM_VALUE...")
                mlog = mavutil.mavlink_connection(bin_file)

                while True:
                    msg = mlog.recv_match(type='PARAM_VALUE', blocking=False)
                    if msg is None:
                        break

                    param_name = msg.param_id
                    if isinstance(param_name, bytes):
                        param_name = param_name.decode('utf-8')
                    param_name = param_name.rstrip('\x00')

                    parameters[param_name] = float(msg.param_value)

                logger.info(f"Extracted {len(parameters)} parameters from PARAM_VALUE messages")

        except Exception as e:
            logger.error(f"Failed to extract parameters from {bin_file}: {e}")
            return {}

        return parameters

    @staticmethod
    def verify_parameters_match(intended: Dict[str, float],
                               actual: Dict[str, float],
                               tolerance: float = 0.001) -> Dict:
        """
        Verify that intended parameters match actual parameters from log.

        Args:
            intended: Parameters we intended to set
            actual: Parameters extracted from .bin log
            tolerance: Allowed difference for floating point comparison

        Returns:
            Dictionary with verification results:
            {
                'all_match': bool,
                'mismatches': [(param_name, intended_value, actual_value), ...],
                'missing': [param_name, ...],
                'match_count': int,
                'total_count': int
            }
        """
        mismatches = []
        missing = []
        match_count = 0

        for param_name, intended_value in intended.items():
            if param_name not in actual:
                missing.append(param_name)
                continue

            actual_value = actual[param_name]
            diff = abs(actual_value - intended_value)

            if diff > tolerance:
                mismatches.append((param_name, intended_value, actual_value))
            else:
                match_count += 1

        all_match = (len(mismatches) == 0 and len(missing) == 0)

        result = {
            'all_match': all_match,
            'mismatches': mismatches,
            'missing': missing,
            'match_count': match_count,
            'total_count': len(intended),
            'verification_rate': match_count / len(intended) if len(intended) > 0 else 0.0
        }

        if not all_match:
            if mismatches:
                logger.warning(f"Parameter mismatches found: {len(mismatches)}")
                for name, intended_val, actual_val in mismatches[:5]:  # Show first 5
                    logger.warning(f"  {name}: intended={intended_val:.6f}, actual={actual_val:.6f}")
            if missing:
                logger.warning(f"Missing parameters in log: {missing[:5]}")

        return result
