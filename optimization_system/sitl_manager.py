"""
SITL Manager for Parallel Instance Management

Manages multiple ArduPilot SITL instances for parallel parameter evaluation
"""

import subprocess
import time
import os
import signal
import logging
from typing import List, Dict, Optional, Tuple
import threading
from queue import Queue, Empty
from pymavlink import mavutil
from dataclasses import dataclass
import random
import socket


logger = logging.getLogger(__name__)


@dataclass
class SITLInstance:
    """Represents a single SITL instance"""
    instance_id: int
    process: Optional[subprocess.Popen]
    mavlink_port: int
    sitl_port: int
    connection: Optional[mavutil.mavlink_connection]
    status: str  # 'idle', 'running', 'crashed', 'error'
    current_params: Optional[Dict[str, float]]


class SITLManager:
    """Manages multiple parallel SITL instances"""

    def __init__(self, num_instances: int = 10, speedup: int = 1,
                 ardupilot_path: str = None, frame_type: str = "quad-30kg"):
        """
        Initialize SITL Manager

        Args:
            num_instances: Number of parallel SITL instances
            speedup: SITL speedup factor (1 = real-time)
            ardupilot_path: Path to ArduPilot directory
            frame_type: Frame type to use (quad-30kg for our custom frame)
        """
        self.num_instances = num_instances
        self.speedup = speedup
        self.frame_type = frame_type

        # Determine ArduPilot path
        if ardupilot_path is None:
            # Try to find it in common locations
            possible_paths = [
                os.path.expanduser("~/ardupilot"),
                os.path.join(os.getcwd(), "ardupilot"),
                "/home/kunal/Documents/WORK/mc07 sitl/ardupilot"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.ardupilot_path = path
                    break
            else:
                raise FileNotFoundError("ArduPilot directory not found")
        else:
            self.ardupilot_path = ardupilot_path

        logger.info(f"Using ArduPilot path: {self.ardupilot_path}")

        # Instance management
        self.instances: List[SITLInstance] = []
        self.instance_queue = Queue()
        self.lock = threading.Lock()

        # Port management (starting ports to avoid conflicts)
        self.base_mavlink_port = 14550
        self.base_sitl_port = 5760

        # Initialize instances
        self._initialize_instances()

    def _initialize_instances(self):
        """Initialize all SITL instances"""
        logger.info(f"Initializing {self.num_instances} SITL instances...")

        for i in range(self.num_instances):
            instance = SITLInstance(
                instance_id=i,
                process=None,
                mavlink_port=self.base_mavlink_port + i * 10,
                sitl_port=self.base_sitl_port + i * 10,
                connection=None,
                status='idle',
                current_params=None
            )
            self.instances.append(instance)
            self.instance_queue.put(i)

        logger.info(f"Initialized {len(self.instances)} instances")

    def _find_free_port(self, start_port: int) -> int:
        """Find a free port starting from start_port"""
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    port += 1
                    if port > start_port + 1000:
                        raise RuntimeError("Could not find free port")

    def start_instance(self, instance_id: int, parameters: Dict[str, float]) -> bool:
        """
        Start a SITL instance with given parameters

        Args:
            instance_id: Instance ID to start
            parameters: Parameter dictionary to apply

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            instance = self.instances[instance_id]

            if instance.status == 'running':
                logger.warning(f"Instance {instance_id} already running")
                return False

            logger.debug(f"Starting SITL instance {instance_id}")

            # Use ArduCopter directory as working directory (required for SITL)
            work_dir = os.path.join(self.ardupilot_path, "ArduCopter")

            # Create unique temporary directory for instance data
            temp_dir = f"/tmp/sitl_instance_{instance_id}"
            os.makedirs(temp_dir, exist_ok=True)

            # Ensure free ports
            mavlink_port = self._find_free_port(instance.mavlink_port)
            sitl_port = self._find_free_port(instance.sitl_port)

            instance.mavlink_port = mavlink_port
            instance.sitl_port = sitl_port

            try:
                # Build sim_vehicle command (matching your working command)
                sim_vehicle_path = os.path.join(
                    self.ardupilot_path,
                    "Tools/autotest/sim_vehicle.py"
                )

                cmd = [
                    "python3",
                    sim_vehicle_path,
                    "--model", "quad",
                    "--no-rebuild",
                    "--no-mavproxy",
                    "-w",  # Wipe eeprom
                    "-I", str(instance_id),
                    "--out", f"127.0.0.1:{mavlink_port}",
                    "--speedup", str(self.speedup),
                    f"--add-param-file={os.path.join(self.ardupilot_path, 'Tools/autotest/default_params/copter-30kg.parm')}"
                ]

                # Set environment to prevent xterm and source profile
                env = os.environ.copy()
                env['SITL_RITW'] = '0'  # Disable "run in the window"
                env['DISPLAY'] = ''  # Disable X11

                # Properly quote the command arguments
                cmd_quoted = [f'"{arg}"' if ' ' in arg else arg for arg in cmd]
                profile_cmd = f'. "$HOME/.profile" && {" ".join(cmd_quoted)}'

                # Start SITL process with output logging
                stdout_file = open(f"/tmp/sitl_stdout_{instance_id}.log", "w")
                stderr_file = open(f"/tmp/sitl_stderr_{instance_id}.log", "w")

                instance.process = subprocess.Popen(
                    profile_cmd,
                    shell=True,
                    executable='/bin/bash',  # Use bash instead of sh
                    cwd=work_dir,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    preexec_fn=os.setsid  # Create new process group
                )

                # Wait for SITL to start (increased time for first boot)
                logger.debug(f"Waiting for SITL {instance_id} to boot...")
                time.sleep(15)

                # Check if process is still running
                if instance.process.poll() is not None:
                    logger.error(f"SITL process {instance_id} died during startup")
                    logger.error(f"Check logs: /tmp/sitl_stdout_{instance_id}.log and /tmp/sitl_stderr_{instance_id}.log")
                    return False

                # Connect via MAVLink (SITL uses TCP on base port 5760 + instance_id*10)
                actual_tcp_port = self.base_sitl_port + instance_id * 10
                connection_string = f"tcp:127.0.0.1:{actual_tcp_port}"
                logger.debug(f"Connecting to {connection_string}")

                # Retry connection with exponential backoff
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        instance.connection = mavutil.mavlink_connection(
                            connection_string,
                            timeout=10
                        )
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            wait_time = 2 ** retry  # Exponential backoff
                            logger.debug(f"Connection attempt {retry+1} failed, waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise e

                # Wait for heartbeat with longer timeout
                logger.debug(f"Waiting for heartbeat on instance {instance_id}")
                msg = instance.connection.wait_heartbeat(timeout=90)

                if msg is None:
                    logger.error(f"No heartbeat received from instance {instance_id}")
                    logger.error(f"Check logs: /tmp/sitl_stdout_{instance_id}.log and /tmp/sitl_stderr_{instance_id}.log")
                    self._kill_instance(instance)
                    return False

                logger.debug(f"Heartbeat received from instance {instance_id}")

                # Apply parameters
                if not self._apply_parameters(instance, parameters):
                    logger.error(f"Failed to apply parameters to instance {instance_id}")
                    self._kill_instance(instance)
                    return False

                instance.status = 'running'
                instance.current_params = parameters.copy()

                logger.info(f"Instance {instance_id} started successfully on port {mavlink_port}")
                return True

            except Exception as e:
                logger.error(f"Failed to start instance {instance_id}: {e}")
                self._kill_instance(instance)
                return False

    def _apply_parameters(self, instance: SITLInstance, parameters: Dict[str, float]) -> bool:
        """Apply parameters to SITL instance"""
        try:
            logger.debug(f"Applying {len(parameters)} parameters to instance {instance.instance_id}")

            for param_name, param_value in parameters.items():
                # Send parameter set command
                instance.connection.mav.param_set_send(
                    instance.connection.target_system,
                    instance.connection.target_component,
                    param_name.encode('utf-8'),
                    float(param_value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )

                # Wait for acknowledgment
                time.sleep(0.01)

            # Give time for parameters to settle
            time.sleep(1)

            logger.debug(f"Parameters applied to instance {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply parameters: {e}")
            return False

    def stop_instance(self, instance_id: int):
        """Stop a SITL instance"""
        with self.lock:
            instance = self.instances[instance_id]
            self._kill_instance(instance)
            instance.status = 'idle'
            instance.current_params = None

    def _kill_instance(self, instance: SITLInstance):
        """Kill a SITL instance process"""
        try:
            if instance.connection:
                instance.connection.close()
                instance.connection = None

            if instance.process:
                # Kill entire process group
                os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)

                # Wait for process to terminate
                try:
                    instance.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                    instance.process.wait()

                instance.process = None

        except Exception as e:
            logger.warning(f"Error killing instance {instance.instance_id}: {e}")

    def get_instance(self, timeout: float = None) -> Optional[int]:
        """
        Get an available instance ID from the pool

        Args:
            timeout: Maximum time to wait for an instance

        Returns:
            Instance ID or None if timeout
        """
        try:
            instance_id = self.instance_queue.get(timeout=timeout)
            return instance_id
        except Empty:
            return None

    def release_instance(self, instance_id: int):
        """Release an instance back to the pool"""
        self.instance_queue.put(instance_id)

    def run_simulation(self, instance_id: int, parameters: Dict[str, float],
                      test_sequence: callable, duration: float = 60.0) -> Tuple[bool, Dict]:
        """
        Run a simulation with given parameters

        Args:
            instance_id: Instance ID to use
            parameters: Parameters to test
            test_sequence: Function that runs the test sequence
            duration: Maximum simulation duration

        Returns:
            (success, telemetry_data)
        """
        instance = self.instances[instance_id]

        # Start instance with parameters
        if not self.start_instance(instance_id, parameters):
            return False, {}

        try:
            # Run test sequence and collect telemetry
            success, telemetry = test_sequence(instance.connection, duration)

            return success, telemetry

        except Exception as e:
            logger.error(f"Simulation failed on instance {instance_id}: {e}")
            return False, {}

        finally:
            # Stop instance
            self.stop_instance(instance_id)

    def run_parallel_simulations(self, parameter_sets: List[Dict[str, float]],
                                test_sequence: callable, duration: float = 60.0) -> List[Tuple[bool, Dict]]:
        """
        Run multiple simulations in parallel

        Args:
            parameter_sets: List of parameter dictionaries to test
            test_sequence: Test sequence function
            duration: Simulation duration

        Returns:
            List of (success, telemetry) tuples
        """
        results = [None] * len(parameter_sets)
        threads = []

        def worker(idx: int, params: Dict[str, float]):
            # Get available instance
            instance_id = self.get_instance(timeout=300)
            if instance_id is None:
                logger.error(f"No instance available for parameter set {idx}")
                results[idx] = (False, {})
                return

            try:
                # Run simulation
                success, telemetry = self.run_simulation(
                    instance_id, params, test_sequence, duration
                )
                results[idx] = (success, telemetry)

            finally:
                # Release instance
                self.release_instance(instance_id)

        # Start threads
        for idx, params in enumerate(parameter_sets):
            thread = threading.Thread(target=worker, args=(idx, params))
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        return results

    def cleanup(self):
        """Cleanup all SITL instances"""
        logger.info("Cleaning up SITL instances...")

        for instance in self.instances:
            self._kill_instance(instance)

        # Clean up temporary directories
        for i in range(self.num_instances):
            work_dir = f"/tmp/sitl_instance_{i}"
            if os.path.exists(work_dir):
                try:
                    import shutil
                    shutil.rmtree(work_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove {work_dir}: {e}")

        logger.info("Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
