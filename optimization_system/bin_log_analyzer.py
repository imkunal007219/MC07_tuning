"""
ArduPilot .bin Log Analyzer

Extracts parameters and flight data from ArduPilot binary log files (.bin)
for verification that parameters were actually applied during flight.

Uses mavlogdump.py to extract CSV data from .bin logs.
"""

import os
import logging
import subprocess
import glob
import csv
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BinLogAnalyzer:
    """Extract parameters from ArduPilot .bin log files"""

    @staticmethod
    def find_latest_bin_log(ardupilot_path: str, instance_id: int) -> Optional[str]:
        """
        Find the most recent .bin log file for a specific SITL instance.

        ArduPilot SITL writes logs to: ArduCopter/logs/
        Log files are named like: 00000001.BIN, 00000002.BIN, etc.

        Args:
            ardupilot_path: Path to ArduPilot root directory
            instance_id: SITL instance ID

        Returns:
            Path to latest .bin file, or None if not found
        """
        # Check multiple possible log locations
        log_dirs = [
            os.path.join(ardupilot_path, "ArduCopter", "logs"),
            os.path.join(ardupilot_path, "ArduCopter"),
            f"/tmp/sitl_instance_{instance_id}",
        ]

        latest_bin = None
        latest_mtime = 0

        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue

            pattern = os.path.join(log_dir, "*.BIN")
            bin_files = glob.glob(pattern)

            # Also check lowercase .bin
            pattern_lower = os.path.join(log_dir, "*.bin")
            bin_files.extend(glob.glob(pattern_lower))

            for bin_file in bin_files:
                mtime = os.path.getmtime(bin_file)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_bin = bin_file

        if latest_bin:
            logger.debug(f"Found latest .bin log for instance {instance_id}: {latest_bin}")
        else:
            logger.warning(f"No .bin files found for instance {instance_id} in {log_dirs}")

        return latest_bin

    @staticmethod
    def extract_parameters_csv(bin_file: str, venv_python: str = None) -> Dict[str, float]:
        """
        Extract all parameters from ArduPilot .bin log file using CSV method.

        Uses mavlogdump.py to convert PARM messages to CSV, then parses the CSV.

        Args:
            bin_file: Path to .bin log file
            venv_python: Path to venv python (default: looks for venv-ardupilot)

        Returns:
            Dictionary of parameter_name -> value
        """
        if not os.path.exists(bin_file):
            logger.error(f".bin file not found: {bin_file}")
            return {}

        # Find venv python
        if venv_python is None:
            possible_paths = [
                os.path.expanduser("~/venv-ardupilot/bin/python"),
                os.path.expanduser("~/venv-ardupilot/bin/python3"),
                "python3",
                "python",
            ]
            for path in possible_paths:
                if os.path.exists(path) or subprocess.run(["which", path], capture_output=True).returncode == 0:
                    venv_python = path
                    break

        if not venv_python:
            logger.error("Could not find python executable for mavlogdump")
            return {}

        # Find mavlogdump.py
        mavlogdump_paths = [
            os.path.expanduser("~/venv-ardupilot/bin/mavlogdump.py"),
            "mavlogdump.py",
        ]

        mavlogdump = None
        for path in mavlogdump_paths:
            if os.path.exists(path):
                mavlogdump = path
                break

        if not mavlogdump:
            logger.warning("mavlogdump.py not found, falling back to pymavlink method")
            return BinLogAnalyzer.extract_parameters(bin_file)

        parameters = {}

        try:
            logger.debug(f"Extracting PARM messages from {bin_file} using mavlogdump")

            # Run mavlogdump to extract PARM messages as CSV
            result = subprocess.run(
                [venv_python, mavlogdump, "--format", "csv", "--types", "PARM", bin_file],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"mavlogdump failed: {result.stderr}")
                return {}

            # Parse CSV output
            csv_data = result.stdout.strip()
            if not csv_data:
                logger.warning(f"No PARM messages found in {bin_file}")
                return {}

            # Parse CSV
            reader = csv.DictReader(csv_data.split('\n'))
            for row in reader:
                try:
                    # PARM CSV format: timestamp, Name, Value
                    if 'Name' in row and 'Value' in row:
                        param_name = row['Name'].strip()
                        param_value = float(row['Value'])
                        parameters[param_name] = param_value
                except (ValueError, KeyError) as e:
                    logger.debug(f"Could not parse PARM row: {row}, error: {e}")
                    continue

            logger.info(f"Extracted {len(parameters)} parameters from {bin_file} using CSV method")

        except subprocess.TimeoutExpired:
            logger.error(f"mavlogdump timed out processing {bin_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to extract parameters using CSV method: {e}")
            return {}

        return parameters

    @staticmethod
    def extract_parameters(bin_file: str) -> Dict[str, float]:
        """
        Extract all parameters from ArduPilot .bin log file using pymavlink.

        Fallback method if mavlogdump CSV extraction fails.

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
            from pymavlink import mavutil

            # Open .bin file with pymavlink
            mlog = mavutil.mavlink_connection(bin_file)

            logger.debug(f"Extracting parameters from {bin_file} using pymavlink")

            # Read all PARM messages
            msg_count = 0
            while True:
                msg = mlog.recv_match(type='PARM', blocking=False)
                if msg is None:
                    break

                msg_count += 1

                # Extract parameter name and value
                param_name = msg.Name
                if isinstance(param_name, bytes):
                    param_name = param_name.decode('utf-8')
                param_name = param_name.rstrip('\x00')  # Remove null padding

                param_value = msg.Value

                parameters[param_name] = float(param_value)

            logger.info(f"Extracted {len(parameters)} parameters from {bin_file} ({msg_count} PARM messages)")

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
