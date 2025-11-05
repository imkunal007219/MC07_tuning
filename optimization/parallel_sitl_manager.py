#!/usr/bin/env python3
"""
Parallel SITL Manager for Multi-Instance Drone Simulation

Manages multiple ArduPilot SITL instances running simultaneously
to enable parallel evaluation of parameter candidates.

Supports up to N parallel instances (limited by CPU cores and RAM).
For 4-core system: recommend 3-4 parallel instances (leave 1 core for OS).

Author: MC07 Tuning System
Date: 2025-11-05
"""

import os
import subprocess
import time
import signal
import shutil
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SITLInstance:
    """Container for SITL instance information"""
    instance_id: int
    sitl_port: int
    mavlink_port: int
    work_dir: Path
    process: Optional[subprocess.Popen] = None
    status: str = 'initialized'  # initialized, running, completed, failed, crashed
    fitness_score: Optional[float] = None
    parameters: Optional[Dict[str, float]] = None


class ParallelSITLManager:
    """
    Manages multiple parallel SITL instances for parameter optimization
    """

    def __init__(self,
                 ardupilot_root: str,
                 num_instances: int = 4,
                 base_sitl_port: int = 5760,
                 base_mavlink_port: int = 14550,
                 work_dir: Optional[str] = None):
        """
        Initialize the parallel SITL manager

        Args:
            ardupilot_root: Path to ArduPilot root directory
            num_instances: Number of parallel instances (default: 4)
            base_sitl_port: Base SITL port (instances use base + N*10)
            base_mavlink_port: Base MAVLink port (instances use base + N*10)
            work_dir: Optional working directory (default: temp directory)
        """
        self.ardupilot_root = Path(ardupilot_root)
        self.num_instances = num_instances
        self.base_sitl_port = base_sitl_port
        self.base_mavlink_port = base_mavlink_port

        # Create working directory
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix='sitl_optimization_'))

        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Track instances
        self.instances: List[SITLInstance] = []
        self.lock = threading.Lock()

        # ArduPilot paths
        self.sim_vehicle_path = self.ardupilot_root / "Tools" / "autotest" / "sim_vehicle.py"
        self.copter_dir = self.ardupilot_root / "ArduCopter"

        # Verify paths
        if not self.sim_vehicle_path.exists():
            raise FileNotFoundError(f"sim_vehicle.py not found at {self.sim_vehicle_path}")
        if not self.copter_dir.exists():
            raise FileNotFoundError(f"ArduCopter directory not found at {self.copter_dir}")

        logger.info(f"ParallelSITLManager initialized with {num_instances} instances")
        logger.info(f"Working directory: {self.work_dir}")

    def create_instance(self, instance_id: int, parameters: Dict[str, float]) -> SITLInstance:
        """
        Create a SITL instance configuration

        Args:
            instance_id: Unique instance ID
            parameters: Dictionary of ArduPilot parameters for this instance

        Returns:
            SITLInstance object
        """
        # Calculate ports (ArduPilot convention: increment by 10 per instance)
        sitl_port = self.base_sitl_port + (instance_id * 10)
        mavlink_port = self.base_mavlink_port + (instance_id * 10)

        # Create instance working directory
        instance_dir = self.work_dir / f"instance_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)

        instance = SITLInstance(
            instance_id=instance_id,
            sitl_port=sitl_port,
            mavlink_port=mavlink_port,
            work_dir=instance_dir,
            parameters=parameters
        )

        # Write parameter file
        param_file = instance_dir / "candidate_params.parm"
        self._write_parameter_file(param_file, parameters)

        return instance

    def _write_parameter_file(self, file_path: Path, parameters: Dict[str, float]):
        """
        Write ArduPilot parameter file

        Args:
            file_path: Path to write parameter file
            parameters: Dictionary of parameters
        """
        with open(file_path, 'w') as f:
            f.write("# Auto-generated parameter file for optimization\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for param_name, value in parameters.items():
                f.write(f"{param_name}    {value}\n")

        logger.debug(f"Wrote {len(parameters)} parameters to {file_path}")

    def launch_instance(self,
                       instance: SITLInstance,
                       model: str = "drone-30kg",
                       defaults: Optional[str] = None,
                       timeout: int = 300) -> bool:
        """
        Launch a single SITL instance

        Args:
            instance: SITLInstance object
            model: Vehicle model name (default: "drone-30kg")
            defaults: Optional path to default parameter file
            timeout: Maximum runtime in seconds

        Returns:
            True if launch successful, False otherwise
        """
        try:
            param_file = instance.work_dir / "candidate_params.parm"

            # Build command
            cmd = [
                str(self.sim_vehicle_path),
                '-I', str(instance.instance_id),
                '-f', model,
                '--out', f'127.0.0.1:{instance.mavlink_port}',
                '--no-mavproxy',  # Don't launch MAVProxy (we control via dronekit)
                '--no-rebuild',   # Don't rebuild (assume already built)
                '-w',  # Wipe EEPROM
            ]

            # Add parameter file if exists
            if param_file.exists():
                cmd.extend(['--add-param-file', str(param_file)])

            # Add default parameters if specified
            if defaults:
                cmd.extend(['--defaults', defaults])

            # Environment variables
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            # Launch process
            logger.info(f"Launching SITL instance {instance.instance_id} on ports {instance.sitl_port}/{instance.mavlink_port}")
            logger.debug(f"Command: {' '.join(cmd)}")

            instance.process = subprocess.Popen(
                cmd,
                cwd=str(self.copter_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )

            instance.status = 'running'
            with self.lock:
                self.instances.append(instance)

            return True

        except Exception as e:
            logger.error(f"Failed to launch instance {instance.instance_id}: {e}")
            instance.status = 'failed'
            return False

    def launch_parallel(self,
                       candidate_params_list: List[Dict[str, float]],
                       test_function: Callable[[SITLInstance], float],
                       model: str = "drone-30kg",
                       defaults: Optional[str] = None) -> List[SITLInstance]:
        """
        Launch multiple SITL instances in parallel and evaluate them

        Args:
            candidate_params_list: List of parameter dictionaries to test
            test_function: Function that takes SITLInstance and returns fitness score
            model: Vehicle model name
            defaults: Optional path to default parameter file

        Returns:
            List of completed SITLInstance objects with fitness scores
        """
        num_candidates = len(candidate_params_list)
        logger.info(f"Launching {num_candidates} parallel SITL instances (batch size: {self.num_instances})")

        completed_instances = []

        # Process in batches
        for batch_start in range(0, num_candidates, self.num_instances):
            batch_end = min(batch_start + self.num_instances, num_candidates)
            batch = candidate_params_list[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//self.num_instances + 1}: "
                       f"candidates {batch_start} to {batch_end-1}")

            # Create instances for this batch
            batch_instances = []
            for i, params in enumerate(batch):
                instance_id = batch_start + i
                instance = self.create_instance(instance_id, params)
                batch_instances.append(instance)

            # Launch instances
            threads = []
            for instance in batch_instances:
                thread = threading.Thread(
                    target=self._run_instance,
                    args=(instance, test_function, model, defaults)
                )
                threads.append(thread)
                thread.start()

                # Small delay to stagger startups
                time.sleep(0.5)

            # Wait for all instances in batch to complete
            for thread in threads:
                thread.join()

            # Collect results
            completed_instances.extend(batch_instances)

            # Log batch results
            successful = sum(1 for inst in batch_instances if inst.status == 'completed')
            logger.info(f"Batch complete: {successful}/{len(batch_instances)} successful")

        return completed_instances

    def _run_instance(self,
                     instance: SITLInstance,
                     test_function: Callable[[SITLInstance], float],
                     model: str,
                     defaults: Optional[str]):
        """
        Internal method to run a single instance (called in thread)

        Args:
            instance: SITLInstance object
            test_function: Function to evaluate fitness
            model: Vehicle model name
            defaults: Optional default parameters
        """
        try:
            # Launch SITL
            if not self.launch_instance(instance, model, defaults):
                logger.error(f"Instance {instance.instance_id} failed to launch")
                return

            # Wait for SITL to be ready
            time.sleep(10)  # TODO: Improve with actual ready check

            # Run test and get fitness score
            fitness = test_function(instance)
            instance.fitness_score = fitness
            instance.status = 'completed'

            logger.info(f"Instance {instance.instance_id} completed with fitness: {fitness:.4f}")

        except Exception as e:
            logger.error(f"Instance {instance.instance_id} failed: {e}")
            instance.status = 'failed'
            instance.fitness_score = float('-inf')  # Worst possible score

        finally:
            # Clean up
            self.stop_instance(instance)

    def stop_instance(self, instance: SITLInstance):
        """
        Stop a running SITL instance

        Args:
            instance: SITLInstance to stop
        """
        if instance.process and instance.process.poll() is None:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(instance.process.pid), signal.SIGTERM)

                # Wait for graceful shutdown (timeout after 5 seconds)
                try:
                    instance.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if not terminated
                    os.killpg(os.getpgid(instance.process.pid), signal.SIGKILL)
                    instance.process.wait()

                logger.debug(f"Stopped instance {instance.instance_id}")

            except Exception as e:
                logger.warning(f"Error stopping instance {instance.instance_id}: {e}")

    def stop_all(self):
        """Stop all running instances"""
        logger.info("Stopping all SITL instances...")
        for instance in self.instances:
            self.stop_instance(instance)

    def cleanup(self):
        """Clean up temporary files and directories"""
        logger.info("Cleaning up temporary files...")
        try:
            # Stop all instances first
            self.stop_all()

            # Remove working directory
            if self.work_dir.exists() and 'sitl_optimization_' in str(self.work_dir):
                shutil.rmtree(self.work_dir)
                logger.info(f"Removed working directory: {self.work_dir}")

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.cleanup()


def example_test_function(instance: SITLInstance) -> float:
    """
    Example test function that returns a fitness score

    In real use, this would:
    1. Connect to SITL via dronekit
    2. Arm and takeoff
    3. Run test maneuvers
    4. Evaluate performance metrics
    5. Return fitness score

    Args:
        instance: SITLInstance to test

    Returns:
        Fitness score (higher is better)
    """
    # Placeholder: Random fitness for demonstration
    import random
    logger.info(f"Testing instance {instance.instance_id}...")
    time.sleep(5)  # Simulate test duration
    fitness = random.uniform(0, 100)
    return fitness


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("PARALLEL SITL MANAGER DEMONSTRATION")
    print("="*70 + "\n")

    # Configuration
    ardupilot_root = "/home/user/MC07_tuning/ardupilot"
    num_instances = 4

    # Example parameter candidates
    from parameter_search_spaces import ParameterSearchSpace
    search_space = ParameterSearchSpace()
    stage1_params = search_space.get_stage_1_rate_controller_space()

    # Create 8 random parameter candidates (will process in 2 batches of 4)
    import random
    candidates = []
    for i in range(8):
        params = {}
        for param_name, param_info in stage1_params.items():
            params[param_name] = random.uniform(param_info['min'], param_info['max'])
        candidates.append(params)

    print(f"Created {len(candidates)} parameter candidates")
    print(f"Will process in batches of {num_instances}\n")

    # Run parallel optimization
    try:
        with ParallelSITLManager(ardupilot_root, num_instances=num_instances) as manager:
            results = manager.launch_parallel(
                candidates,
                example_test_function,
                model="drone-30kg"
            )

            # Print results
            print("\n" + "="*70)
            print("RESULTS")
            print("="*70)
            for instance in results:
                print(f"Instance {instance.instance_id}: "
                      f"Status={instance.status}, "
                      f"Fitness={instance.fitness_score:.2f if instance.fitness_score else 'N/A'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
