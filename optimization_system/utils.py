"""
Utility functions for the optimization system
"""

import logging
import pickle
import json
import os
import numpy as np
from datetime import datetime


def setup_logging(log_file, level='INFO'):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create logger
    logger = logging.getLogger('AutoTune')
    logger.setLevel(getattr(logging, level))

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_results(filename, data):
    """Save results to pickle file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_results(filename):
    """Load results from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_json(filename, data):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    data = convert_types(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def calculate_rise_time(time_array, response_array, target_value, threshold=0.9):
    """Calculate rise time (time to reach 90% of target)"""
    try:
        target_90 = target_value * threshold
        idx = np.where(response_array >= target_90)[0]
        if len(idx) > 0:
            return time_array[idx[0]]
        return np.inf
    except:
        return np.inf


def calculate_settling_time(time_array, response_array, target_value, threshold=0.02):
    """Calculate settling time (time to stay within 2% of target)"""
    try:
        upper_bound = target_value * (1 + threshold)
        lower_bound = target_value * (1 - threshold)

        # Find where it enters the band
        in_band = (response_array >= lower_bound) & (response_array <= upper_bound)

        # Find last time it left the band
        for i in range(len(in_band) - 1, 0, -1):
            if not in_band[i]:
                return time_array[i + 1] if i + 1 < len(time_array) else time_array[-1]

        # If never left the band after entering
        idx = np.where(in_band)[0]
        if len(idx) > 0:
            return time_array[idx[0]]

        return np.inf
    except:
        return np.inf


def calculate_overshoot(response_array, target_value):
    """Calculate overshoot percentage"""
    try:
        max_value = np.max(response_array)
        overshoot = ((max_value - target_value) / target_value) * 100
        return max(0, overshoot)
    except:
        return np.inf


def calculate_steady_state_error(response_array, target_value, window=100):
    """Calculate steady-state error (average error in final window)"""
    try:
        final_window = response_array[-window:]
        avg_final = np.mean(final_window)
        error = abs((avg_final - target_value) / target_value) * 100
        return error
    except:
        return np.inf


def detect_oscillations(time_array, response_array, threshold=0.1):
    """Detect sustained oscillations in response"""
    try:
        # Calculate derivative
        diff = np.diff(response_array)

        # Count zero crossings
        zero_crossings = np.where(np.diff(np.sign(diff)))[0]

        # Check if oscillating
        if len(zero_crossings) > 10:  # More than 10 crossings indicates oscillation
            # Check amplitude
            peaks = []
            for i in range(1, len(response_array) - 1):
                if (response_array[i] > response_array[i-1] and
                    response_array[i] > response_array[i+1]):
                    peaks.append(response_array[i])

            if len(peaks) > 5:
                avg_peak_amplitude = np.mean(np.abs(np.diff(peaks)))
                if avg_peak_amplitude > threshold:
                    return True, avg_peak_amplitude

        return False, 0.0
    except:
        return False, 0.0


def check_motor_saturation(motor_outputs, threshold=0.95):
    """Check if motors are saturating"""
    try:
        max_output = np.max(motor_outputs)
        saturation_time = np.sum(motor_outputs > threshold) / len(motor_outputs)
        return max_output > threshold, saturation_time
    except:
        return False, 0.0


def calculate_power_consumption(motor_outputs, dt):
    """Calculate average power consumption"""
    try:
        # Simplified power model: P ∝ throttle³
        power = np.sum(motor_outputs ** 3) * dt
        avg_power = np.mean(motor_outputs ** 3)
        return power, avg_power
    except:
        return np.inf, np.inf


def detect_crash(attitude, altitude, velocity, max_angle=45, min_altitude=0.1):
    """Detect if drone crashed"""
    try:
        # Check extreme angles
        if np.abs(attitude['roll']) > max_angle or np.abs(attitude['pitch']) > max_angle:
            return True, "Extreme attitude"

        # Check altitude
        if altitude < min_altitude:
            return True, "Ground collision"

        # Check if flipped (roll or pitch > 90 degrees)
        if np.abs(attitude['roll']) > 90 or np.abs(attitude['pitch']) > 90:
            return True, "Vehicle flipped"

        # Check vertical velocity near ground
        if altitude < 1.0 and velocity['z'] < -5.0:
            return True, "High descent rate near ground"

        return False, None
    except:
        return True, "Data error"


def normalize_fitness(fitness_components, weights):
    """Normalize and combine fitness components"""
    try:
        total_fitness = 0.0

        for component, value in fitness_components.items():
            if component in weights:
                # Invert if needed (lower is better)
                if component in ['overshoot', 'steady_state_error', 'response_time']:
                    normalized = 1.0 / (1.0 + value)
                else:
                    normalized = value

                total_fitness += weights[component] * normalized

        return total_fitness
    except:
        return 0.0


def create_individual(bounds):
    """Create random individual within bounds"""
    individual = {}
    for param, (min_val, max_val) in bounds.items():
        individual[param] = np.random.uniform(min_val, max_val)
    return individual


def mutate_individual(individual, bounds, mutation_rate=0.1):
    """Mutate individual parameters"""
    mutated = individual.copy()

    for param in mutated.keys():
        if np.random.random() < mutation_rate:
            min_val, max_val = bounds[param]
            # Add Gaussian noise
            sigma = (max_val - min_val) * 0.1
            mutated[param] = np.clip(
                mutated[param] + np.random.normal(0, sigma),
                min_val,
                max_val
            )

    return mutated


def crossover_individuals(parent1, parent2, crossover_rate=0.7):
    """Crossover two individuals"""
    child1 = {}
    child2 = {}

    for param in parent1.keys():
        if np.random.random() < crossover_rate:
            # Swap
            child1[param] = parent2[param]
            child2[param] = parent1[param]
        else:
            # Keep
            child1[param] = parent1[param]
            child2[param] = parent2[param]

    return child1, child2


def select_tournament(population, fitness_scores, tournament_size=3):
    """Tournament selection"""
    selected = []
    for _ in range(len(population)):
        # Random tournament
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())

    return selected


def get_free_port(start_port=5760, max_attempts=100):
    """Find available port for SITL instance"""
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            continue

    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port+max_attempts}")


def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    """Print progress bar"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()
