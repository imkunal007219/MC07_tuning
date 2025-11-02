"""
Data Logging and Telemetry Collection System
Comprehensive logging for optimization trials, flight data, and analysis
"""

import os
import json
import csv
import time
import h5py
import pickle
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FlightLog:
    """Single flight log entry"""
    timestamp: float
    altitude: float
    roll: float
    pitch: float
    yaw: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    latitude: float
    longitude: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    motor1: float
    motor2: float
    motor3: float
    motor4: float
    battery_voltage: float
    battery_current: float
    throttle: float
    mode: str
    armed: bool


@dataclass
class TrialLog:
    """Complete trial data"""
    trial_id: int
    optimization_phase: str
    parameters: Dict[str, float]
    fitness: float
    crashed: bool
    duration: float
    test_results: Dict[str, Any]
    safety_violations: List[Dict]
    flight_data: List[FlightLog]
    timestamp: str


class TelemetryCollector:
    """Real-time telemetry collection from MAVLink"""

    def __init__(self, connection):
        self.connection = connection
        self.telemetry_buffer = []
        self.collecting = False
        self.start_time = None

    def start_collection(self):
        """Start collecting telemetry"""
        self.telemetry_buffer = []
        self.collecting = True
        self.start_time = time.time()
        logger.info("Telemetry collection started")

    def stop_collection(self):
        """Stop collecting telemetry"""
        self.collecting = False
        logger.info(f"Telemetry collection stopped. Collected {len(self.telemetry_buffer)} samples")

    def collect_sample(self) -> Optional[FlightLog]:
        """Collect single telemetry sample"""
        if not self.collecting:
            return None

        try:
            # Get ATTITUDE message
            attitude_msg = self.connection.recv_match(type='ATTITUDE', blocking=False, timeout=0.1)
            # Get GLOBAL_POSITION_INT
            position_msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=False, timeout=0.1)
            # Get LOCAL_POSITION_NED
            local_msg = self.connection.recv_match(type='LOCAL_POSITION_NED', blocking=False, timeout=0.1)
            # Get SERVO_OUTPUT_RAW for motors
            servo_msg = self.connection.recv_match(type='SERVO_OUTPUT_RAW', blocking=False, timeout=0.1)
            # Get BATTERY_STATUS
            battery_msg = self.connection.recv_match(type='BATTERY_STATUS', blocking=False, timeout=0.1)
            # Get HEARTBEAT for mode
            heartbeat_msg = self.connection.recv_match(type='HEARTBEAT', blocking=False, timeout=0.1)

            # Parse and create FlightLog
            current_time = time.time() - self.start_time

            flight_log = FlightLog(
                timestamp=current_time,
                altitude=position_msg.relative_alt / 1000.0 if position_msg else 0.0,
                roll=np.degrees(attitude_msg.roll) if attitude_msg else 0.0,
                pitch=np.degrees(attitude_msg.pitch) if attitude_msg else 0.0,
                yaw=np.degrees(attitude_msg.yaw) if attitude_msg else 0.0,
                roll_rate=np.degrees(attitude_msg.rollspeed) if attitude_msg else 0.0,
                pitch_rate=np.degrees(attitude_msg.pitchspeed) if attitude_msg else 0.0,
                yaw_rate=np.degrees(attitude_msg.yawspeed) if attitude_msg else 0.0,
                latitude=position_msg.lat / 1e7 if position_msg else 0.0,
                longitude=position_msg.lon / 1e7 if position_msg else 0.0,
                vx=local_msg.vx if local_msg else 0.0,
                vy=local_msg.vy if local_msg else 0.0,
                vz=local_msg.vz if local_msg else 0.0,
                ax=local_msg.vx / 0.1 if local_msg else 0.0,  # Approximate
                ay=local_msg.vy / 0.1 if local_msg else 0.0,
                az=local_msg.vz / 0.1 if local_msg else 0.0,
                motor1=servo_msg.servo1_raw / 2000.0 if servo_msg else 0.0,
                motor2=servo_msg.servo2_raw / 2000.0 if servo_msg else 0.0,
                motor3=servo_msg.servo3_raw / 2000.0 if servo_msg else 0.0,
                motor4=servo_msg.servo4_raw / 2000.0 if servo_msg else 0.0,
                battery_voltage=battery_msg.voltages[0] / 1000.0 if battery_msg and battery_msg.voltages else 48.0,
                battery_current=battery_msg.current_battery / 100.0 if battery_msg else 0.0,
                throttle=0.5,  # TODO: Extract from RC_CHANNELS
                mode=self._get_mode_string(heartbeat_msg.custom_mode) if heartbeat_msg else "UNKNOWN",
                armed=heartbeat_msg.base_mode & 128 == 128 if heartbeat_msg else False
            )

            self.telemetry_buffer.append(flight_log)
            return flight_log

        except Exception as e:
            logger.warning(f"Failed to collect telemetry sample: {e}")
            return None

    def _get_mode_string(self, mode_num: int) -> str:
        """Convert mode number to string"""
        modes = {
            0: 'STABILIZE',
            1: 'ACRO',
            2: 'ALT_HOLD',
            3: 'AUTO',
            4: 'GUIDED',
            5: 'LOITER',
            6: 'RTL',
            7: 'CIRCLE',
            9: 'LAND',
            16: 'POSHOLD',
            17: 'BRAKE',
            18: 'THROW',
            19: 'AVOID_ADSB',
            20: 'GUIDED_NOGPS',
            21: 'SMART_RTL',
        }
        return modes.get(mode_num, f'MODE_{mode_num}')

    def get_collected_data(self) -> List[FlightLog]:
        """Get all collected telemetry"""
        return self.telemetry_buffer.copy()

    def clear_buffer(self):
        """Clear telemetry buffer"""
        self.telemetry_buffer = []


class DataLogger:
    """
    Comprehensive data logging system
    Supports multiple formats: JSON, CSV, HDF5, pickle
    """

    def __init__(self, base_dir: str = "./logs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.trial_dir = self.base_dir / "trials"
        self.flight_dir = self.base_dir / "flight_data"
        self.analysis_dir = self.base_dir / "analysis"
        self.param_dir = self.base_dir / "parameters"

        for dir_path in [self.trial_dir, self.flight_dir, self.analysis_dir, self.param_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Data logger initialized. Session: {self.current_session}")

    def log_trial(self, trial_log: TrialLog):
        """Log complete trial data"""
        trial_file = self.trial_dir / f"trial_{trial_log.trial_id:04d}_{self.current_session}.json"

        # Convert to dictionary
        trial_dict = {
            'trial_id': trial_log.trial_id,
            'optimization_phase': trial_log.optimization_phase,
            'parameters': trial_log.parameters,
            'fitness': trial_log.fitness,
            'crashed': trial_log.crashed,
            'duration': trial_log.duration,
            'test_results': trial_log.test_results,
            'safety_violations': trial_log.safety_violations,
            'timestamp': trial_log.timestamp,
            'flight_data_samples': len(trial_log.flight_data)
        }

        # Save JSON
        with open(trial_file, 'w') as f:
            json.dump(trial_dict, f, indent=2)

        # Save flight data separately in CSV
        if trial_log.flight_data:
            flight_file = self.flight_dir / f"flight_{trial_log.trial_id:04d}_{self.current_session}.csv"
            self._save_flight_data_csv(trial_log.flight_data, flight_file)

        logger.info(f"Trial {trial_log.trial_id} logged to {trial_file}")

    def _save_flight_data_csv(self, flight_data: List[FlightLog], filepath: Path):
        """Save flight data to CSV"""
        if not flight_data:
            return

        with open(filepath, 'w', newline='') as f:
            fieldnames = list(asdict(flight_data[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for log_entry in flight_data:
                writer.writerow(asdict(log_entry))

    def save_flight_data_hdf5(self, flight_data: List[FlightLog], trial_id: int):
        """Save flight data to HDF5 for efficient storage and analysis"""
        if not flight_data:
            return

        filepath = self.flight_dir / f"flight_{trial_id:04d}_{self.current_session}.h5"

        with h5py.File(filepath, 'w') as f:
            # Create datasets for each field
            n_samples = len(flight_data)

            # Time series data
            f.create_dataset('timestamp', data=[d.timestamp for d in flight_data])
            f.create_dataset('altitude', data=[d.altitude for d in flight_data])
            f.create_dataset('roll', data=[d.roll for d in flight_data])
            f.create_dataset('pitch', data=[d.pitch for d in flight_data])
            f.create_dataset('yaw', data=[d.yaw for d in flight_data])
            f.create_dataset('roll_rate', data=[d.roll_rate for d in flight_data])
            f.create_dataset('pitch_rate', data=[d.pitch_rate for d in flight_data])
            f.create_dataset('yaw_rate', data=[d.yaw_rate for d in flight_data])

            # Position
            f.create_dataset('latitude', data=[d.latitude for d in flight_data])
            f.create_dataset('longitude', data=[d.longitude for d in flight_data])

            # Velocity
            f.create_dataset('vx', data=[d.vx for d in flight_data])
            f.create_dataset('vy', data=[d.vy for d in flight_data])
            f.create_dataset('vz', data=[d.vz for d in flight_data])

            # Acceleration
            f.create_dataset('ax', data=[d.ax for d in flight_data])
            f.create_dataset('ay', data=[d.ay for d in flight_data])
            f.create_dataset('az', data=[d.az for d in flight_data])

            # Motors
            motors = np.array([[d.motor1, d.motor2, d.motor3, d.motor4] for d in flight_data])
            f.create_dataset('motors', data=motors)

            # Battery
            f.create_dataset('battery_voltage', data=[d.battery_voltage for d in flight_data])
            f.create_dataset('battery_current', data=[d.battery_current for d in flight_data])

            # Metadata
            f.attrs['trial_id'] = trial_id
            f.attrs['n_samples'] = n_samples
            f.attrs['duration'] = flight_data[-1].timestamp if flight_data else 0

        logger.info(f"Flight data saved to HDF5: {filepath}")

    def log_parameters(self, parameters: Dict[str, float], phase: str, trial_id: int, fitness: float):
        """Log parameter values"""
        param_file = self.param_dir / f"params_{phase}_{trial_id:04d}.json"

        param_log = {
            'trial_id': trial_id,
            'phase': phase,
            'fitness': fitness,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat()
        }

        with open(param_file, 'w') as f:
            json.dump(param_log, f, indent=2)

    def log_optimization_summary(self, summary: Dict[str, Any]):
        """Log optimization summary"""
        summary_file = self.base_dir / f"optimization_summary_{self.current_session}.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Optimization summary saved to {summary_file}")

    def save_best_parameters(self, parameters: Dict[str, float], fitness: float):
        """Save best parameters to .parm file"""
        parm_file = self.param_dir / f"best_params_{self.current_session}.parm"

        with open(parm_file, 'w') as f:
            f.write(f"# Best Parameters - Fitness: {fitness:.4f}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Session: {self.current_session}\n\n")

            for param_name, value in sorted(parameters.items()):
                f.write(f"{param_name},{value:.6f}\n")

        logger.info(f"Best parameters saved to {parm_file}")
        return parm_file

    def export_trial_summary_csv(self):
        """Export all trials to summary CSV"""
        summary_file = self.base_dir / f"trial_summary_{self.current_session}.csv"

        # Load all trial files
        trial_files = sorted(self.trial_dir.glob(f"trial_*_{self.current_session}.json"))

        if not trial_files:
            logger.warning("No trial files found for export")
            return

        trials_data = []
        for trial_file in trial_files:
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
                trials_data.append(trial_data)

        # Write CSV
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['trial_id', 'optimization_phase', 'fitness', 'crashed', 'duration', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in trials_data:
                writer.writerow({
                    'trial_id': trial['trial_id'],
                    'optimization_phase': trial['optimization_phase'],
                    'fitness': trial['fitness'],
                    'crashed': trial['crashed'],
                    'duration': trial['duration'],
                    'timestamp': trial['timestamp']
                })

        logger.info(f"Trial summary exported to {summary_file}")

    def load_trial_data(self, trial_id: int) -> Optional[Dict]:
        """Load trial data by ID"""
        trial_files = list(self.trial_dir.glob(f"trial_{trial_id:04d}_*.json"))

        if not trial_files:
            logger.error(f"Trial {trial_id} not found")
            return None

        with open(trial_files[0], 'r') as f:
            return json.load(f)

    def load_flight_data_csv(self, trial_id: int) -> Optional[List[Dict]]:
        """Load flight data from CSV"""
        flight_files = list(self.flight_dir.glob(f"flight_{trial_id:04d}_*.csv"))

        if not flight_files:
            logger.error(f"Flight data for trial {trial_id} not found")
            return None

        flight_data = []
        with open(flight_files[0], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                flight_data.append(row)

        return flight_data

    def load_flight_data_hdf5(self, trial_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Load flight data from HDF5"""
        flight_files = list(self.flight_dir.glob(f"flight_{trial_id:04d}_*.h5"))

        if not flight_files:
            logger.error(f"HDF5 flight data for trial {trial_id} not found")
            return None

        flight_data = {}
        with h5py.File(flight_files[0], 'r') as f:
            for key in f.keys():
                flight_data[key] = f[key][:]

        return flight_data

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Calculate statistics across all trials"""
        trial_files = sorted(self.trial_dir.glob(f"trial_*_{self.current_session}.json"))

        if not trial_files:
            return {}

        fitnesses = []
        crashed_count = 0
        durations = []
        phases = {}

        for trial_file in trial_files:
            with open(trial_file, 'r') as f:
                trial = json.load(f)
                fitnesses.append(trial['fitness'])
                if trial['crashed']:
                    crashed_count += 1
                durations.append(trial['duration'])

                phase = trial['optimization_phase']
                if phase not in phases:
                    phases[phase] = []
                phases[phase].append(trial['fitness'])

        stats = {
            'total_trials': len(trial_files),
            'crashes': crashed_count,
            'crash_rate': crashed_count / len(trial_files) if trial_files else 0,
            'fitness': {
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'min': np.min(fitnesses),
                'max': np.max(fitnesses),
                'median': np.median(fitnesses)
            },
            'duration': {
                'mean': np.mean(durations),
                'total': np.sum(durations)
            },
            'phases': {}
        }

        for phase, phase_fitnesses in phases.items():
            stats['phases'][phase] = {
                'trials': len(phase_fitnesses),
                'mean_fitness': np.mean(phase_fitnesses),
                'max_fitness': np.max(phase_fitnesses)
            }

        return stats

    def create_backup(self):
        """Create backup of all logs"""
        import shutil

        backup_dir = self.base_dir / f"backup_{self.current_session}"
        shutil.copytree(self.base_dir, backup_dir, ignore=shutil.ignore_patterns('backup_*'))
        logger.info(f"Backup created at {backup_dir}")


class RealtimeLogger:
    """Realtime console and file logging for optimization progress"""

    def __init__(self, log_file: str = "optimization.log"):
        self.log_file = log_file
        self.trial_count = 0
        self.best_fitness = -float('inf')

        # Setup file logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    def log_trial_start(self, trial_id: int, phase: str, parameters: Dict[str, float]):
        """Log trial start"""
        self.trial_count += 1
        logger.info("=" * 80)
        logger.info(f"TRIAL {trial_id} START - Phase: {phase}")
        logger.info(f"Parameters: {len(parameters)}")
        logger.info("=" * 80)

    def log_trial_end(self, trial_id: int, fitness: float, crashed: bool, duration: float):
        """Log trial end"""
        logger.info("=" * 80)
        logger.info(f"TRIAL {trial_id} END")
        logger.info(f"Fitness: {fitness:.4f}")
        logger.info(f"Crashed: {crashed}")
        logger.info(f"Duration: {duration:.2f}s")

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            logger.info(f"*** NEW BEST FITNESS: {fitness:.4f} ***")

        logger.info("=" * 80)

    def log_phase_complete(self, phase: str, best_fitness: float, n_trials: int):
        """Log phase completion"""
        logger.info("")
        logger.info("#" * 80)
        logger.info(f"PHASE COMPLETE: {phase}")
        logger.info(f"Best Fitness: {best_fitness:.4f}")
        logger.info(f"Trials: {n_trials}")
        logger.info("#" * 80)
        logger.info("")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    data_logger = DataLogger("./logs")

    # Create sample trial log
    sample_flight_data = [
        FlightLog(
            timestamp=i * 0.1,
            altitude=10.0 + np.sin(i * 0.1),
            roll=np.random.randn() * 2,
            pitch=np.random.randn() * 2,
            yaw=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
            latitude=37.7749,
            longitude=-122.4194,
            vx=0.0, vy=0.0, vz=0.0,
            ax=0.0, ay=0.0, az=0.0,
            motor1=0.5, motor2=0.5, motor3=0.5, motor4=0.5,
            battery_voltage=48.0,
            battery_current=20.0,
            throttle=0.5,
            mode='GUIDED',
            armed=True
        )
        for i in range(100)
    ]

    trial_log = TrialLog(
        trial_id=1,
        optimization_phase='rate_roll',
        parameters={'ATC_RAT_RLL_P': 0.15, 'ATC_RAT_RLL_I': 0.10},
        fitness=0.85,
        crashed=False,
        duration=30.0,
        test_results={'overshoot': 5.0, 'settling_time': 2.5},
        safety_violations=[],
        flight_data=sample_flight_data,
        timestamp=datetime.now().isoformat()
    )

    data_logger.log_trial(trial_log)
    data_logger.save_flight_data_hdf5(sample_flight_data, trial_id=1)

    stats = data_logger.get_optimization_statistics()
    logger.info(f"Statistics: {stats}")
