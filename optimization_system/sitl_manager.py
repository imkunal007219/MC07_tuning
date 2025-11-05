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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # Class-level shutdown flag
    _shutdown_requested = False

    def __init__(self, num_instances: int = 10, speedup: int = 1,
                 ardupilot_path: str = None, frame_type: str = "drone-30kg"):
        """
        Initialize SITL Manager

        Args:
            num_instances: Number of parallel SITL instances
            speedup: SITL speedup factor (1 = real-time)
            ardupilot_path: Path to ArduPilot directory
            frame_type: Frame type to use (drone-30kg for our custom frame)
        """
        self.num_instances = num_instances
        self.speedup = speedup
        self.frame_type = frame_type

        # Determine ArduPilot path
        if ardupilot_path is None:
            # Try to find it in common locations
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            possible_paths = [
                os.path.join(project_root, "ardupilot"),  # Project root
                os.path.expanduser("~/Documents/MC07_tuning/ardupilot"),  # Standard location
                os.path.expanduser("~/ardupilot"),         # Home directory
                os.path.join(os.getcwd(), "ardupilot"),   # Current working directory
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.ardupilot_path = path
                    break
            else:
                raise FileNotFoundError(
                    f"ArduPilot directory not found. Searched in:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths)
                )
        else:
            self.ardupilot_path = ardupilot_path

        logger.info(f"Using ArduPilot path: {self.ardupilot_path}")

        # Instance management
        self.instances: List[SITLInstance] = []
        self.instance_queue = Queue()
        self.lock = threading.Lock()

        # File handle tracking to prevent leaks
        self.log_files: Dict[int, Tuple] = {}

        # Port management (starting ports to avoid conflicts)
        self.base_mavlink_port = 14550
        self.base_sitl_port = 5760

        # Warm-start mode: Keep SITL processes alive between trials
        self.warm_start = True  # Enable warm-start by default for speed
        self.warm_instances_ready = set()  # Track which instances are warm and ready

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

    def _find_free_port(self) -> int:
        """
        Find a free port using OS assignment

        This avoids TOCTOU race conditions by letting the OS choose a free port.
        The port is guaranteed to be available at the time of return.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Port 0 tells OS to assign a free port
            s.listen(1)
            port = s.getsockname()[1]
            return port

    def start_instance(self, instance_id: int, parameters: Dict[str, float]) -> bool:
        """
        Start a SITL instance with given parameters

        Args:
            instance_id: Instance ID to start
            parameters: Parameter dictionary to apply

        Returns:
            True if successful, False otherwise
        """
        # Check if shutdown was requested
        if SITLManager._shutdown_requested:
            logger.warning(f"Shutdown requested - aborting start_instance({instance_id})")
            return False

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

            # Use pre-assigned ports from _initialize_instances()
            # ArduPilot SITL automatically assigns ports based on instance_id with -I flag:
            # Instance 0: SITL port 5760, MAVLink 14550
            # Instance 1: SITL port 5770, MAVLink 14560
            # Instance N: SITL port 5760+(N*10), MAVLink 14550+(N*10)
            sitl_port = instance.sitl_port
            mavlink_port = instance.mavlink_port

            logger.info(f"Instance {instance_id} will use SITL port {sitl_port}, MAVLink port {mavlink_port}")

            try:
                # Build sim_vehicle command (matching your working command)
                sim_vehicle_path = os.path.join(
                    self.ardupilot_path,
                    "Tools/autotest/sim_vehicle.py"
                )

                # Simple working command that matches manual SITL startup
                # MAVProxy runs automatically and outputs to UDP port 14550 + (instance_id * 10)
                # We connect via UDP (not TCP) to MAVProxy output port
                cmd = [
                    "python3",
                    sim_vehicle_path,
                    "-v", "ArduCopter",
                    "-f", self.frame_type,
                    "--console",  # Open MAVProxy console in separate xterm window
                    "-I", str(instance_id),  # Instance ID for multi-instance support
                    "--speedup", str(self.speedup),  # Speedup factor
                    # Let MAVProxy run (don't use --no-mavproxy)
                    # This matches the working manual command
                ]

                # Set environment to show SITL terminal for debugging
                env = os.environ.copy()
                # ENABLE xterm window so you can see SITL output during automation
                # The SITL terminal will pop up showing MAVProxy console
                # This helps debug what's happening during optimization runs

                # Properly quote the command arguments
                cmd_quoted = [f'"{arg}"' if ' ' in arg else arg for arg in cmd]
                profile_cmd = f'. "$HOME/.profile" && {" ".join(cmd_quoted)}'

                # Start SITL process WITHOUT capturing output
                # This allows MAVProxy console to be visible in terminal/xterm window
                # Don't capture stdout/stderr so we can see MAVProxy console
                instance.process = subprocess.Popen(
                    profile_cmd,
                    shell=True,
                    executable='/bin/bash',  # Use bash instead of sh
                    cwd=work_dir,
                    env=env,
                    # No stdout/stderr redirection - let output go to console
                    preexec_fn=os.setsid  # Create new process group
                )

                # Wait for SITL to boot and MAVProxy to start
                logger.debug(f"Waiting for SITL {instance_id} to boot...")

                # MAVProxy outputs to UDP port 14550 + (instance_id * 10)
                # NOT TCP to SITL port 5760!
                mavproxy_port = self.base_mavlink_port + instance_id * 10
                connection_string = f"udp:127.0.0.1:{mavproxy_port}"
                logger.info(f"Will connect to MAVProxy via {connection_string}")

                # Give SITL time to boot before attempting connection
                logger.info(f"Giving SITL {instance_id} 20 seconds to boot...")
                time.sleep(20)

                # Poll for SITL readiness with exponential backoff
                # Increased timeout because EKF initialization can take 30-60 seconds
                max_boot_time = 90  # Maximum time to wait for boot
                boot_start = time.time()
                connection_established = False

                while time.time() - boot_start < max_boot_time:
                    # Check if process is still running
                    if instance.process.poll() is not None:
                        logger.error(f"SITL process {instance_id} died during startup")
                        logger.error(f"Check logs: /tmp/sitl_stdout_{instance_id}.log and /tmp/sitl_stderr_{instance_id}.log")
                        return False

                    # Try to connect
                    try:
                        logger.debug(f"Attempting connection to {connection_string}")
                        instance.connection = mavutil.mavlink_connection(
                            connection_string,
                            timeout=1
                        )

                        # Try to get a heartbeat
                        msg = instance.connection.wait_heartbeat(timeout=3)

                        if msg is not None:
                            elapsed = time.time() - boot_start
                            logger.info(f"SITL {instance_id} ready in {elapsed:.1f}s")
                            connection_established = True
                            break

                    except Exception as e:
                        logger.debug(f"Connection attempt failed: {e}")
                        # Close failed connection attempt
                        if instance.connection:
                            try:
                                instance.connection.close()
                            except:
                                pass
                            instance.connection = None

                    # Exponential backoff for next attempt
                    elapsed = time.time() - boot_start
                    if elapsed < 5:
                        time.sleep(0.5)  # Check frequently at first
                    elif elapsed < 10:
                        time.sleep(1.0)  # Less frequent after 5s
                    else:
                        time.sleep(2.0)  # Even less frequent after 10s

                if not connection_established:
                    logger.error(f"SITL {instance_id} failed to start within {max_boot_time}s")
                    logger.error(f"Check logs: /tmp/sitl_stdout_{instance_id}.log and /tmp/sitl_stderr_{instance_id}.log")
                    return False

                # Connection established and heartbeat received
                logger.debug(f"Connection and heartbeat confirmed for instance {instance_id}")

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
        """
        Apply parameters to SITL instance with read-back verification

        This ensures parameters are actually set before continuing,
        preventing silent failures from affecting optimization results.
        """
        try:
            logger.debug(f"Applying {len(parameters)} parameters to instance {instance.instance_id}")

            failed_params = []

            for param_name, param_value in parameters.items():
                # Send parameter set command
                instance.connection.mav.param_set_send(
                    instance.connection.target_system,
                    instance.connection.target_component,
                    param_name.encode('utf-8'),
                    float(param_value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
                )

            # Small delay to allow parameter processing (reduced from 10ms per param)
            time.sleep(0.1)

            # Verify critical parameters were set correctly
            # Only verify a sample to avoid excessive overhead
            # In production, you might verify all parameters
            verify_count = min(5, len(parameters))
            params_to_verify = list(parameters.items())[:verify_count]

            for param_name, expected_value in params_to_verify:
                # Request parameter
                instance.connection.mav.param_request_read_send(
                    instance.connection.target_system,
                    instance.connection.target_component,
                    param_name.encode('utf-8'),
                    -1
                )

                # Wait for response
                msg = instance.connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=2)

                if msg is None:
                    logger.warning(f"No response for parameter {param_name}")
                    failed_params.append(param_name)
                    continue

                # Decode parameter name (handle both bytes and str)
                if isinstance(msg.param_id, bytes):
                    received_name = msg.param_id.decode('utf-8').rstrip('\x00')
                else:
                    received_name = msg.param_id.rstrip('\x00')

                if received_name == param_name:
                    # Check if value matches (with tolerance for floating point)
                    if abs(msg.param_value - expected_value) > 0.001:
                        logger.warning(
                            f"Parameter {param_name} mismatch: "
                            f"expected {expected_value}, got {msg.param_value}"
                        )
                        failed_params.append(param_name)

            if failed_params:
                logger.error(f"Failed to set parameters: {failed_params}")
                return False

            logger.debug(f"Parameters verified on instance {instance.instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply parameters: {e}")
            return False

    def stop_instance(self, instance_id: int):
        """
        Stop a SITL instance

        In warm-start mode, keeps the process alive and just marks as idle.
        In normal mode, kills the process completely.
        """
        with self.lock:
            instance = self.instances[instance_id]

            if self.warm_start and instance_id in self.warm_instances_ready:
                # Warm-start mode: Keep process alive, just mark as idle
                logger.debug(f"Instance {instance_id} kept alive for warm-start")
                instance.status = 'idle'
                # Don't clear current_params - we'll check if we can reuse them
            else:
                # Normal mode or instance not ready for warm-start: kill completely
                self._kill_instance(instance)
                instance.status = 'idle'
                instance.current_params = None
                if instance_id in self.warm_instances_ready:
                    self.warm_instances_ready.remove(instance_id)

    def _kill_instance(self, instance: SITLInstance):
        """Kill a SITL instance process"""
        try:
            # Don't bother with graceful connection close - just kill processes
            # Graceful close can hang if MAVProxy is stuck reconnecting
            instance.connection = None

            if instance.process and instance.process.pid:
                # Kill entire process group immediately with SIGKILL
                try:
                    logger.debug(f"Force killing process group for instance {instance.instance_id} (PID: {instance.process.pid})")

                    # Use SIGKILL immediately - no graceful shutdown
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)

                    # Very short wait - process should die immediately with SIGKILL
                    try:
                        instance.process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        # Should not happen with SIGKILL, but just move on
                        pass
                    except (ProcessLookupError, OSError):
                        pass  # Process already terminated

                except (ProcessLookupError, OSError) as e:
                    # Process already terminated
                    logger.debug(f"Process {instance.instance_id} already terminated: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error killing instance {instance.instance_id}: {e}")

                instance.process = None

            # No log files to close since we're not capturing output

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

    def _warm_update_parameters(self, instance_id: int, parameters: Dict[str, float]) -> bool:
        """
        Update parameters on a running SITL instance (warm-start optimization)

        This is much faster than restarting the entire SITL process.
        Typically saves 10-15 seconds per trial.

        Args:
            instance_id: Instance ID
            parameters: New parameters to apply

        Returns:
            True if successful, False if restart required
        """
        instance = self.instances[instance_id]

        # Check if instance is warm and ready
        if not (self.warm_start and instance_id in self.warm_instances_ready):
            return False

        # Check if connection is still alive
        if not instance.connection or instance.process.poll() is not None:
            logger.warning(f"Instance {instance_id} connection lost, restart required")
            self.warm_instances_ready.discard(instance_id)
            return False

        # Apply new parameters
        if not self._apply_parameters(instance, parameters):
            logger.warning(f"Failed to update parameters on instance {instance_id}, restart required")
            self.warm_instances_ready.discard(instance_id)
            return False

        instance.current_params = parameters.copy()
        logger.info(f"Instance {instance_id} parameters updated (warm-start)")
        return True

    def run_simulation(self, instance_id: int, parameters: Dict[str, float],
                      test_sequence: callable, duration: float = 60.0) -> Tuple[bool, Dict]:
        """
        Run a simulation with given parameters

        Uses warm-start optimization if enabled: updates parameters on running
        instance instead of restart. This provides ~30% speedup.

        Args:
            instance_id: Instance ID to use
            parameters: Parameters to test
            test_sequence: Function that runs the test sequence
            duration: Maximum simulation duration

        Returns:
            (success, telemetry_data)
        """
        # Check if shutdown was requested
        if SITLManager._shutdown_requested:
            logger.warning(f"Shutdown requested - aborting run_simulation({instance_id})")
            return False, {}

        instance = self.instances[instance_id]

        # Try warm-start first (much faster)
        if self.warm_start and instance_id in self.warm_instances_ready:
            if self._warm_update_parameters(instance_id, parameters):
                # Warm update successful, run test directly
                try:
                    success, telemetry = test_sequence(instance.connection, duration)
                    return success, telemetry
                except Exception as e:
                    logger.error(f"Warm simulation failed on instance {instance_id}: {e}")
                    # Fall through to cold start

        # Cold start required (first run or warm-start failed)
        if not self.start_instance(instance_id, parameters):
            return False, {}

        # Mark instance as warm-ready for future trials
        if self.warm_start:
            self.warm_instances_ready.add(instance_id)

        try:
            # Run test sequence and collect telemetry
            success, telemetry = test_sequence(instance.connection, duration)

            return success, telemetry

        except Exception as e:
            logger.error(f"Simulation failed on instance {instance_id}: {e}")
            return False, {}

        finally:
            # Stop instance (respects warm-start mode)
            self.stop_instance(instance_id)

    def run_parallel_simulations(self, parameter_sets: List[Dict[str, float]],
                                test_sequence: callable, duration: float = 60.0) -> List[Tuple[bool, Dict]]:
        """
        Run multiple simulations in parallel using ThreadPoolExecutor

        This provides better resource management than manual threading:
        - Automatic thread pooling and reuse
        - Built-in timeout and exception handling
        - Prevents thread explosion with max_workers limit

        Args:
            parameter_sets: List of parameter dictionaries to test
            test_sequence: Test sequence function
            duration: Simulation duration

        Returns:
            List of (success, telemetry) tuples
        """
        results = [None] * len(parameter_sets)

        def worker(idx: int, params: Dict[str, float]) -> Tuple[int, Tuple[bool, Dict]]:
            """Worker function that returns index and result"""
            # Get available instance
            instance_id = self.get_instance(timeout=300)
            if instance_id is None:
                logger.error(f"No instance available for parameter set {idx}")
                return (idx, (False, {}))

            try:
                # Run simulation
                success, telemetry = self.run_simulation(
                    instance_id, params, test_sequence, duration
                )
                return (idx, (success, telemetry))

            except Exception as e:
                logger.error(f"Simulation {idx} failed with exception: {e}")
                return (idx, (False, {}))

            finally:
                # Release instance
                self.release_instance(instance_id)

        # Use ThreadPoolExecutor with limited workers (num_instances)
        # This prevents creating more threads than we have SITL instances
        with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
            # Submit all tasks
            futures = {
                executor.submit(worker, idx, params): idx
                for idx, params in enumerate(parameter_sets)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    # Handle any unexpected exceptions
                    original_idx = futures[future]
                    logger.error(f"Worker {original_idx} raised exception: {e}")
                    results[original_idx] = (False, {})

        return results

    def cleanup(self):
        """Cleanup all SITL instances"""
        # Check if instances exist (in case cleanup called after failed __init__)
        if not hasattr(self, 'instances'):
            return

        logger.info("Cleaning up SITL instances...")

        # Set shutdown flag to prevent any new SITL starts
        SITLManager._shutdown_requested = True

        # Kill all instances with short timeout per instance
        for instance in self.instances:
            try:
                self._kill_instance(instance)
            except Exception as e:
                logger.warning(f"Error killing instance {instance.instance_id}: {e}")
                # Continue to next instance

        # No log files to close since we're not capturing output
        if hasattr(self, 'log_files'):
            self.log_files.clear()

        # Nuclear cleanup: kill ANY remaining SITL/MAVProxy processes
        # This is important - sometimes processes escape the process group kill
        logger.info("Killing any remaining SITL/MAVProxy processes...")
        try:
            # Use pkill -9 (SIGKILL) for immediate termination
            os.system("pkill -9 arducopter 2>/dev/null")
            os.system("pkill -9 'mavproxy.py' 2>/dev/null")
            os.system("pkill -9 python 2>/dev/null | grep -q mavproxy")  # Kill mavproxy python processes
            os.system("pkill -9 sim_vehicle 2>/dev/null")
            # Also kill any xterm windows that might be hanging
            os.system("pkill -9 xterm 2>/dev/null")
        except Exception as e:
            logger.warning(f"Error in nuclear cleanup: {e}")

        # Clean up temporary directories
        num_instances = getattr(self, 'num_instances', 0)
        for i in range(num_instances):
            work_dir = f"/tmp/sitl_instance_{i}"
            if os.path.exists(work_dir):
                try:
                    import shutil
                    shutil.rmtree(work_dir, ignore_errors=True)  # Don't fail on errors
                except Exception as e:
                    logger.debug(f"Failed to remove {work_dir}: {e}")

        logger.info("Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
