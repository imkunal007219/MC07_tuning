"""
Mission File Loader

Loads and uploads ArduPilot waypoint mission files to SITL.
Supports QGroundControl WPL 110 format.
"""

import logging
from typing import List, Tuple
from pymavlink import mavutil
import time

logger = logging.getLogger(__name__)


class MissionLoader:
    """Handles loading and uploading mission files to ArduPilot."""

    def __init__(self, connection: mavutil.mavlink_connection):
        """
        Initialize mission loader.

        Args:
            connection: MAVLink connection to vehicle
        """
        self.connection = connection

    def load_mission_file(self, mission_file: str) -> List[Tuple]:
        """
        Load mission from .waypoints file.

        Args:
            mission_file: Path to .waypoints file

        Returns:
            List of waypoint tuples (seq, frame, command, params, x, y, z, autocontinue)
        """
        logger.info(f"Loading mission file: {mission_file}")

        waypoints = []

        try:
            with open(mission_file, 'r') as f:
                lines = f.readlines()

            # First line should be "QGC WPL 110"
            if not lines[0].strip().startswith('QGC WPL'):
                raise ValueError(f"Invalid mission file format: {lines[0]}")

            # Parse waypoints
            for line in lines[1:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 11:
                    logger.warning(f"Skipping invalid line: {line}")
                    continue

                waypoint = {
                    'seq': int(parts[0]),
                    'current': int(parts[1]),
                    'frame': int(parts[2]),
                    'command': int(parts[3]),
                    'param1': float(parts[4]),
                    'param2': float(parts[5]),
                    'param3': float(parts[6]),
                    'param4': float(parts[7]),
                    'x': float(parts[8]),  # Latitude
                    'y': float(parts[9]),  # Longitude
                    'z': float(parts[10]), # Altitude
                    'autocontinue': int(parts[11]) if len(parts) > 11 else 1
                }

                waypoints.append(waypoint)
                logger.debug(f"Loaded waypoint {waypoint['seq']}: CMD={waypoint['command']}")

            logger.info(f"Loaded {len(waypoints)} waypoints from {mission_file}")
            return waypoints

        except Exception as e:
            logger.error(f"Failed to load mission file: {e}")
            return []

    def upload_mission(self, waypoints: List[dict], timeout: float = 30.0) -> bool:
        """
        Upload mission to vehicle.

        Args:
            waypoints: List of waypoint dictionaries
            timeout: Upload timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        if not waypoints:
            logger.error("No waypoints to upload")
            return False

        logger.info(f"Uploading {len(waypoints)} waypoints to vehicle...")

        try:
            # Clear existing mission
            logger.debug("Clearing existing mission...")
            self.connection.waypoint_clear_all_send()
            time.sleep(0.5)

            # Send mission count
            logger.debug(f"Sending mission count: {len(waypoints)}")
            self.connection.waypoint_count_send(len(waypoints))

            # Wait for waypoint request
            start_time = time.time()
            uploaded = 0

            while uploaded < len(waypoints) and time.time() - start_time < timeout:
                msg = self.connection.recv_match(
                    type=['MISSION_REQUEST', 'MISSION_REQUEST_INT'],
                    blocking=True,
                    timeout=5
                )

                if msg is None:
                    logger.warning(f"Timeout waiting for waypoint request (uploaded {uploaded}/{len(waypoints)})")
                    continue

                seq = msg.seq
                logger.debug(f"Vehicle requested waypoint {seq}")

                if seq >= len(waypoints):
                    logger.error(f"Vehicle requested invalid waypoint: {seq}")
                    return False

                wp = waypoints[seq]

                # Send waypoint
                self.connection.mav.mission_item_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    wp['seq'],
                    wp['frame'],
                    wp['command'],
                    wp['current'],
                    wp['autocontinue'],
                    wp['param1'],
                    wp['param2'],
                    wp['param3'],
                    wp['param4'],
                    wp['x'],
                    wp['y'],
                    wp['z']
                )

                logger.debug(f"Sent waypoint {seq}: CMD={wp['command']}")
                uploaded += 1

            # Wait for mission ACK
            ack = self.connection.recv_match(type='MISSION_ACK', blocking=True, timeout=5)

            if ack is None:
                logger.error("No mission ACK received")
                return False

            if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                logger.error(f"Mission rejected: {ack.type}")
                return False

            logger.info(f"✓ Mission upload successful ({uploaded} waypoints)")
            return True

        except Exception as e:
            logger.error(f"Mission upload failed: {e}")
            return False

    def load_and_upload(self, mission_file: str, timeout: float = 30.0) -> bool:
        """
        Load mission file and upload to vehicle.

        Args:
            mission_file: Path to mission file
            timeout: Upload timeout

        Returns:
            True if successful
        """
        waypoints = self.load_mission_file(mission_file)
        if not waypoints:
            return False

        return self.upload_mission(waypoints, timeout)

    def verify_mission(self) -> int:
        """
        Verify uploaded mission.

        Returns:
            Number of waypoints in current mission, or -1 on error
        """
        try:
            # Request mission count
            self.connection.waypoint_request_list_send()

            # Wait for mission count
            msg = self.connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=5)

            if msg is None:
                logger.error("No mission count received")
                return -1

            count = msg.count
            logger.info(f"Current mission has {count} waypoints")
            return count

        except Exception as e:
            logger.error(f"Mission verification failed: {e}")
            return -1


# Helper function for easy use
def upload_mission_file(connection: mavutil.mavlink_connection,
                        mission_file: str) -> bool:
    """
    Convenience function to upload a mission file.

    Args:
        connection: MAVLink connection
        mission_file: Path to .waypoints file

    Returns:
        True if successful
    """
    loader = MissionLoader(connection)
    return loader.load_and_upload(mission_file)
