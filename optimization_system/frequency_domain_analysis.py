"""
Frequency-Domain Analysis for Control System Tuning

Implements frequency-domain metrics (phase margin, gain margin, Bode plots)
to ensure formal control-theoretic stability rather than just empirical
"no crashes" assessment.

Based on control theory principles:
- Phase margin: PM = 180° + ∠G(jωc) at gain crossover (should be 45-60°)
- Gain margin: GM = 1/|G(jωp)| at phase crossover (should be > 6 dB)
- These margins indicate robustness to modeling errors and disturbances
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def estimate_transfer_function_from_step(time: np.ndarray,
                                        input_signal: np.ndarray,
                                        output_signal: np.ndarray,
                                        sample_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate frequency response from step response data using FFT

    Args:
        time: Time array (seconds)
        input_signal: Input signal (command)
        output_signal: Output signal (actual response)
        sample_rate: Sample rate in Hz

    Returns:
        Tuple of (frequencies, magnitude, phase) in Hz, absolute, and degrees
    """
    # Ensure arrays are numpy arrays
    time = np.asarray(time)
    input_signal = np.asarray(input_signal)
    output_signal = np.asarray(output_signal)

    # Remove DC component
    input_signal = input_signal - np.mean(input_signal)
    output_signal = output_signal - np.mean(output_signal)

    # Calculate FFT
    n_samples = len(output_signal)
    input_fft = fft(input_signal)
    output_fft = fft(output_signal)

    # Calculate transfer function in frequency domain: H(jω) = Y(jω) / U(jω)
    # Avoid division by zero
    transfer_function = np.zeros_like(output_fft, dtype=complex)
    nonzero_mask = np.abs(input_fft) > 1e-10
    transfer_function[nonzero_mask] = output_fft[nonzero_mask] / input_fft[nonzero_mask]

    # Get frequencies
    frequencies = fftfreq(n_samples, 1.0 / sample_rate)

    # Only use positive frequencies
    positive_freq_mask = frequencies > 0
    frequencies = frequencies[positive_freq_mask]
    transfer_function = transfer_function[positive_freq_mask]

    # Calculate magnitude and phase
    magnitude = np.abs(transfer_function)
    phase = np.angle(transfer_function, deg=True)

    return frequencies, magnitude, phase


def calculate_phase_margin(frequencies: np.ndarray,
                          magnitude: np.ndarray,
                          phase: np.ndarray) -> Tuple[float, float]:
    """
    Calculate phase margin from frequency response

    Phase margin = 180° + phase at gain crossover frequency (where |H(jω)| = 1 or 0 dB)

    Args:
        frequencies: Frequency array in Hz
        magnitude: Magnitude array (linear scale)
        phase: Phase array in degrees

    Returns:
        Tuple of (phase_margin_deg, crossover_frequency_hz)
    """
    if len(frequencies) < 2:
        return 0.0, 0.0

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Find gain crossover frequency (where magnitude = 0 dB)
    # Look for sign change in magnitude_db
    crossover_idx = None
    for i in range(len(magnitude_db) - 1):
        if magnitude_db[i] > 0 and magnitude_db[i+1] <= 0:
            # Interpolate to find exact crossover
            crossover_idx = i
            break

    if crossover_idx is None:
        # No crossover found, use peak magnitude frequency as approximation
        crossover_idx = np.argmax(magnitude)

    crossover_freq = frequencies[crossover_idx]

    # Interpolate phase at crossover frequency
    if crossover_idx > 0 and crossover_idx < len(frequencies) - 1:
        # Linear interpolation
        f1, f2 = frequencies[crossover_idx], frequencies[crossover_idx + 1]
        p1, p2 = phase[crossover_idx], phase[crossover_idx + 1]
        phase_at_crossover = p1 + (p2 - p1) * (crossover_freq - f1) / (f2 - f1 + 1e-10)
    else:
        phase_at_crossover = phase[crossover_idx]

    # Phase margin = 180° + phase at crossover
    phase_margin = 180.0 + phase_at_crossover

    # Wrap to [-180, 180]
    while phase_margin > 180:
        phase_margin -= 360
    while phase_margin < -180:
        phase_margin += 360

    return phase_margin, crossover_freq


def calculate_gain_margin(frequencies: np.ndarray,
                         magnitude: np.ndarray,
                         phase: np.ndarray) -> Tuple[float, float]:
    """
    Calculate gain margin from frequency response

    Gain margin = -magnitude_dB at phase crossover frequency (where phase = -180°)

    Args:
        frequencies: Frequency array in Hz
        magnitude: Magnitude array (linear scale)
        phase: Phase array in degrees

    Returns:
        Tuple of (gain_margin_db, phase_crossover_frequency_hz)
    """
    if len(frequencies) < 2:
        return 0.0, 0.0

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Find phase crossover frequency (where phase = -180°)
    phase_crossover_idx = None
    for i in range(len(phase) - 1):
        # Look for phase crossing -180°
        if (phase[i] > -180 and phase[i+1] <= -180) or (phase[i] < -180 and phase[i+1] >= -180):
            phase_crossover_idx = i
            break

    if phase_crossover_idx is None:
        # No phase crossover found - system is likely stable with high margin
        # Return large gain margin
        return 40.0, 0.0

    phase_crossover_freq = frequencies[phase_crossover_idx]

    # Get magnitude at phase crossover
    mag_at_crossover = magnitude_db[phase_crossover_idx]

    # Gain margin = -magnitude at phase crossover (in dB)
    gain_margin = -mag_at_crossover

    return gain_margin, phase_crossover_freq


def analyze_step_response_frequency_domain(time: np.ndarray,
                                          command: np.ndarray,
                                          response: np.ndarray,
                                          sample_rate: float = 100.0) -> Dict[str, float]:
    """
    Analyze step response in frequency domain and extract stability margins

    Args:
        time: Time array (seconds)
        command: Command signal (desired)
        response: Actual response signal
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with frequency-domain metrics
    """
    # Estimate transfer function from step response
    frequencies, magnitude, phase = estimate_transfer_function_from_step(
        time, command, response, sample_rate
    )

    # Calculate phase margin
    phase_margin, gain_crossover_freq = calculate_phase_margin(frequencies, magnitude, phase)

    # Calculate gain margin
    gain_margin, phase_crossover_freq = calculate_gain_margin(frequencies, magnitude, phase)

    # Calculate bandwidth (frequency where magnitude drops to -3dB)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    bandwidth_3db_idx = np.where(magnitude_db < -3.0)[0]
    if len(bandwidth_3db_idx) > 0:
        bandwidth_3db = frequencies[bandwidth_3db_idx[0]]
    else:
        bandwidth_3db = frequencies[-1] if len(frequencies) > 0 else 0.0

    return {
        'phase_margin_deg': phase_margin,
        'gain_margin_db': gain_margin,
        'gain_crossover_freq_hz': gain_crossover_freq,
        'phase_crossover_freq_hz': phase_crossover_freq,
        'bandwidth_3db_hz': bandwidth_3db,
        'frequencies': frequencies,
        'magnitude': magnitude,
        'phase': phase
    }


def calculate_frequency_domain_fitness(freq_metrics: Dict[str, float],
                                      target_phase_margin: float = 52.5,
                                      min_phase_margin: float = 45.0,
                                      target_gain_margin: float = 9.0,
                                      min_gain_margin: float = 6.0) -> Dict[str, float]:
    """
    Calculate fitness scores from frequency-domain metrics

    Args:
        freq_metrics: Dictionary with frequency domain analysis results
        target_phase_margin: Target phase margin (degrees), typically 45-60°
        min_phase_margin: Minimum acceptable phase margin (degrees)
        target_gain_margin: Target gain margin (dB), typically 6-12 dB
        min_gain_margin: Minimum acceptable gain margin (dB)

    Returns:
        Dictionary with normalized fitness scores
    """
    phase_margin = freq_metrics.get('phase_margin_deg', 0.0)
    gain_margin = freq_metrics.get('gain_margin_db', 0.0)

    # Phase margin score (0-1 scale)
    # Optimal range: 45-60 degrees
    if phase_margin >= min_phase_margin:
        # Score increases linearly from min to target
        if phase_margin <= target_phase_margin:
            phase_margin_score = (phase_margin - min_phase_margin) / (target_phase_margin - min_phase_margin)
        else:
            # Above target is good but with diminishing returns
            excess = phase_margin - target_phase_margin
            phase_margin_score = 1.0 + 0.2 * (1.0 - np.exp(-excess / 15.0))
    else:
        # Below minimum - heavy penalty
        phase_margin_score = max(0.0, phase_margin / min_phase_margin) * 0.5

    # Gain margin score (0-1 scale)
    # Optimal range: 6-12 dB
    if gain_margin >= min_gain_margin:
        if gain_margin <= target_gain_margin:
            gain_margin_score = (gain_margin - min_gain_margin) / (target_gain_margin - min_gain_margin)
        else:
            # Above target is good
            excess = gain_margin - target_gain_margin
            gain_margin_score = 1.0 + 0.2 * (1.0 - np.exp(-excess / 6.0))
    else:
        # Below minimum - heavy penalty
        gain_margin_score = max(0.0, gain_margin / min_gain_margin) * 0.5

    # Clip scores to reasonable range
    phase_margin_score = np.clip(phase_margin_score, 0.0, 1.5)
    gain_margin_score = np.clip(gain_margin_score, 0.0, 1.5)

    return {
        'phase_margin_score': phase_margin_score,
        'gain_margin_score': gain_margin_score,
        'phase_margin_raw': phase_margin,
        'gain_margin_raw': gain_margin
    }


def analyze_telemetry_frequency_domain(telemetry: Dict,
                                      sample_rate: float = 100.0) -> Dict[str, any]:
    """
    Perform frequency-domain analysis on flight telemetry data

    Analyzes roll, pitch, yaw, and altitude responses

    Args:
        telemetry: Flight telemetry dictionary with time series data
        sample_rate: Expected sample rate in Hz

    Returns:
        Dictionary with frequency-domain metrics for each axis
    """
    results = {
        'roll': {},
        'pitch': {},
        'yaw': {},
        'altitude': {},
        'overall': {}
    }

    # Extract time array (assumed to be uniform)
    if 'timestamp' in telemetry and len(telemetry['timestamp']) > 0:
        time = np.array(telemetry['timestamp'])
        time = time - time[0]  # Start from 0
        if len(time) > 1:
            sample_rate = 1.0 / np.mean(np.diff(time))
    else:
        logger.warning("No timestamp in telemetry, using default sample rate")
        time = None

    # Analyze Roll
    if 'roll_desired' in telemetry and 'roll' in telemetry:
        roll_cmd = np.array(telemetry['roll_desired'])
        roll_actual = np.array(telemetry['roll'])
        if len(roll_cmd) > 10 and time is not None:
            try:
                roll_freq = analyze_step_response_frequency_domain(
                    time, roll_cmd, roll_actual, sample_rate
                )
                results['roll'] = roll_freq
            except Exception as e:
                logger.debug(f"Roll frequency analysis failed: {e}")

    # Analyze Pitch
    if 'pitch_desired' in telemetry and 'pitch' in telemetry:
        pitch_cmd = np.array(telemetry['pitch_desired'])
        pitch_actual = np.array(telemetry['pitch'])
        if len(pitch_cmd) > 10 and time is not None:
            try:
                pitch_freq = analyze_step_response_frequency_domain(
                    time, pitch_cmd, pitch_actual, sample_rate
                )
                results['pitch'] = pitch_freq
            except Exception as e:
                logger.debug(f"Pitch frequency analysis failed: {e}")

    # Analyze Yaw
    if 'yaw_desired' in telemetry and 'yaw' in telemetry:
        yaw_cmd = np.array(telemetry['yaw_desired'])
        yaw_actual = np.array(telemetry['yaw'])
        if len(yaw_cmd) > 10 and time is not None:
            try:
                yaw_freq = analyze_step_response_frequency_domain(
                    time, yaw_cmd, yaw_actual, sample_rate
                )
                results['yaw'] = yaw_freq
            except Exception as e:
                logger.debug(f"Yaw frequency analysis failed: {e}")

    # Analyze Altitude
    if 'altitude_desired' in telemetry and 'altitude' in telemetry:
        alt_cmd = np.array(telemetry['altitude_desired'])
        alt_actual = np.array(telemetry['altitude'])
        if len(alt_cmd) > 10 and time is not None:
            try:
                alt_freq = analyze_step_response_frequency_domain(
                    time, alt_cmd, alt_actual, sample_rate
                )
                results['altitude'] = alt_freq
            except Exception as e:
                logger.debug(f"Altitude frequency analysis failed: {e}")

    # Calculate overall metrics (average of available axes)
    phase_margins = []
    gain_margins = []

    for axis in ['roll', 'pitch', 'yaw', 'altitude']:
        if 'phase_margin_deg' in results[axis]:
            phase_margins.append(results[axis]['phase_margin_deg'])
        if 'gain_margin_db' in results[axis]:
            gain_margins.append(results[axis]['gain_margin_db'])

    if phase_margins:
        results['overall']['phase_margin_deg'] = np.mean(phase_margins)
        results['overall']['min_phase_margin_deg'] = np.min(phase_margins)
    else:
        results['overall']['phase_margin_deg'] = 0.0
        results['overall']['min_phase_margin_deg'] = 0.0

    if gain_margins:
        results['overall']['gain_margin_db'] = np.mean(gain_margins)
        results['overall']['min_gain_margin_db'] = np.min(gain_margins)
    else:
        results['overall']['gain_margin_db'] = 0.0
        results['overall']['min_gain_margin_db'] = 0.0

    return results


def generate_bode_plot_data(frequencies: np.ndarray,
                           magnitude: np.ndarray,
                           phase: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate Bode plot data in standard format

    Args:
        frequencies: Frequency array in Hz
        magnitude: Magnitude array (linear scale)
        phase: Phase array in degrees

    Returns:
        Dictionary with Bode plot data
    """
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    return {
        'frequency_hz': frequencies,
        'magnitude_db': magnitude_db,
        'phase_deg': phase
    }
