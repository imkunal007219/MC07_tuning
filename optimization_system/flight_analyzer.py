"""
Flight Data Analyzer

Analyzes logged flight data to correlate parameters with performance.
Provides statistical proof of which parameters work and why.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from scipy import signal
from scipy.stats import pearsonr
import json

logger = logging.getLogger(__name__)


class FlightAnalyzer:
    """
    Analyzes flight data to understand parameter effects.

    Calculates:
    - Stability metrics (oscillations, overshoot, settling)
    - Response characteristics (rise time, bandwidth)
    - Parameter correlations with success
    - Statistical significance of parameter effects
    """

    def __init__(self, flight_logger):
        """
        Initialize analyzer.

        Args:
            flight_logger: FlightDataLogger instance with logged flights
        """
        self.logger = flight_logger

    def analyze_stability(self, telemetry: Dict) -> Dict:
        """
        Calculate stability metrics from telemetry.

        Args:
            telemetry: Flight telemetry data

        Returns:
            Dictionary with stability metrics
        """
        metrics = {
            'oscillation_score': 0.0,
            'overshoot_percentage': 0.0,
            'settling_time': 0.0,
            'steady_state_error': 0.0,
            'max_angle_deviation': 0.0,
            'altitude_stability': 0.0,
            'vibration_level': 0.0
        }

        try:
            # Analyze attitude stability (roll/pitch)
            if 'roll' in telemetry and len(telemetry['roll']) > 10:
                metrics.update(self._analyze_attitude_stability(
                    telemetry['roll'],
                    telemetry.get('roll_rate', []),
                    'roll'
                ))

            if 'pitch' in telemetry and len(telemetry['pitch']) > 10:
                pitch_metrics = self._analyze_attitude_stability(
                    telemetry['pitch'],
                    telemetry.get('pitch_rate', []),
                    'pitch'
                )
                # Combine roll and pitch metrics (take worse case)
                for key in ['oscillation_score', 'overshoot_percentage']:
                    if key in pitch_metrics:
                        metrics[key] = max(metrics.get(key, 0), pitch_metrics[key])

            # Analyze altitude stability
            if 'altitude' in telemetry and len(telemetry['altitude']) > 10:
                metrics.update(self._analyze_altitude_stability(
                    telemetry['altitude'],
                    telemetry.get('time', [])
                ))

            # Detect vibrations from accelerometer data
            if 'vibration' in telemetry:
                metrics['vibration_level'] = float(np.mean(telemetry['vibration']))
            elif 'ax' in telemetry and len(telemetry['ax']) > 10:
                # Calculate vibration from accelerometer noise
                ax = np.array(telemetry['ax'])
                metrics['vibration_level'] = float(np.std(ax))

        except Exception as e:
            logger.warning(f"Error calculating stability metrics: {e}")

        return metrics

    def _analyze_attitude_stability(self, angle_data: List[float],
                                     rate_data: List[float],
                                     axis: str) -> Dict:
        """Analyze attitude stability for one axis."""
        angles = np.array(angle_data)
        metrics = {}

        # Detect oscillations using FFT
        if len(angles) > 20:
            # Remove DC component
            angles_ac = angles - np.mean(angles)

            # Calculate FFT
            fft = np.fft.fft(angles_ac)
            freqs = np.fft.fftfreq(len(angles_ac), d=0.05)  # Assuming 20Hz sampling

            # Look for oscillations in 1-10 Hz range
            mask = (freqs > 1) & (freqs < 10)
            if np.any(mask):
                power = np.abs(fft[mask])
                max_power = np.max(power)

                # Oscillation score: higher power = more oscillation
                metrics['oscillation_score'] = float(max_power / len(angles))

                # Find dominant frequency
                if max_power > 0:
                    dominant_freq = freqs[mask][np.argmax(power)]
                    metrics[f'{axis}_oscillation_freq'] = float(dominant_freq)

        # Calculate overshoot and settling
        # Look for step responses (large changes in setpoint)
        if len(angles) > 30:
            # Detect steps by looking at moving average changes
            window = 10
            smoothed = np.convolve(angles, np.ones(window)/window, mode='valid')

            if len(smoothed) > 20:
                # Find largest deviation from final value
                final_value = np.mean(angles[-10:])
                max_deviation = np.max(np.abs(angles - final_value))
                metrics['max_angle_deviation'] = float(max_deviation)

                # Settling time: time to reach within 5% of final value
                threshold = 0.05 * max_deviation if max_deviation > 0 else 1.0
                settling_mask = np.abs(angles - final_value) > threshold

                if np.any(settling_mask):
                    # Find last time outside threshold
                    last_idx = np.where(settling_mask)[0][-1]
                    metrics['settling_time'] = float(last_idx * 0.05)  # Convert to seconds

        # Calculate steady-state error
        if len(angles) > 10:
            # Assume setpoint is 0 for hover (level flight)
            steady_state = angles[-10:]
            metrics['steady_state_error'] = float(np.mean(np.abs(steady_state)))

        return metrics

    def _analyze_altitude_stability(self, altitude_data: List[float],
                                     time_data: List[float]) -> Dict:
        """Analyze altitude hold performance."""
        altitudes = np.array(altitude_data)
        metrics = {}

        if len(altitudes) < 10:
            return metrics

        # Target altitude (assume last stable value)
        target_alt = np.mean(altitudes[-20:]) if len(altitudes) > 20 else np.mean(altitudes)

        # Altitude tracking error
        errors = altitudes - target_alt
        metrics['altitude_rmse'] = float(np.sqrt(np.mean(errors**2)))
        metrics['altitude_max_error'] = float(np.max(np.abs(errors)))

        # Altitude stability (variance)
        metrics['altitude_stability'] = float(np.std(altitudes))

        return metrics

    def correlate_parameters_with_success(self, param_names: List[str] = None) -> Dict:
        """
        Correlate parameter values with flight success.

        Args:
            param_names: List of parameter names to analyze (None = all)

        Returns:
            Dictionary with correlation analysis for each parameter
        """
        results = {}

        # Get all flights
        all_flights = self.logger.flight_index

        if len(all_flights) < 5:
            logger.warning("Not enough flights for correlation analysis")
            return {'error': 'Insufficient data (need at least 5 flights)'}

        # If no param names specified, get all unique parameters
        if param_names is None:
            param_names = set()
            for flight in all_flights:
                param_names.update(flight['parameters'].keys())
            param_names = sorted(param_names)

        # Analyze each parameter
        for param in param_names:
            analysis = self.logger.compare_parameters(param)

            if 'error' in analysis:
                continue

            # Calculate correlation with success
            param_values = []
            success_values = []

            for flight in all_flights:
                if param in flight['parameters']:
                    param_values.append(flight['parameters'][param])
                    success_values.append(1.0 if flight['success'] else 0.0)

            if len(param_values) > 3:
                # Calculate correlation coefficient
                try:
                    corr, p_value = pearsonr(param_values, success_values)
                    analysis['correlation'] = {
                        'coefficient': float(corr),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'interpretation': self._interpret_correlation(corr, p_value)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {param}: {e}")

            results[param] = analysis

        return results

    def _interpret_correlation(self, corr: float, p_value: float) -> str:
        """Provide human-readable interpretation of correlation."""
        if p_value >= 0.05:
            return "No significant correlation with success"

        abs_corr = abs(corr)
        direction = "Higher" if corr > 0 else "Lower"

        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return f"{direction} values show {strength} correlation with success"

    def find_optimal_ranges(self, param_name: str,
                           min_success_rate: float = 0.8) -> Dict:
        """
        Find optimal parameter ranges based on historical data.

        Args:
            param_name: Parameter to analyze
            min_success_rate: Minimum success rate for "optimal" range

        Returns:
            Dictionary with optimal range and confidence
        """
        successful = self.logger.get_successful_flights()
        failed = self.logger.get_failed_flights()

        if len(successful) < 3:
            return {'error': 'Not enough successful flights'}

        # Extract parameter values from successful flights
        success_values = [f['parameters'].get(param_name)
                         for f in successful
                         if param_name in f['parameters']]

        fail_values = [f['parameters'].get(param_name)
                      for f in failed
                      if param_name in f['parameters']]

        if not success_values:
            return {'error': f'Parameter {param_name} not found in flights'}

        success_values = np.array(success_values)

        # Calculate optimal range (mean ± 1 std dev of successful flights)
        optimal_center = float(np.mean(success_values))
        optimal_std = float(np.std(success_values))

        optimal_range = {
            'parameter': param_name,
            'optimal_center': optimal_center,
            'optimal_std': optimal_std,
            'recommended_min': float(optimal_center - optimal_std),
            'recommended_max': float(optimal_center + optimal_std),
            'successful_min': float(np.min(success_values)),
            'successful_max': float(np.max(success_values)),
            'confidence': self._calculate_confidence(success_values, fail_values)
        }

        return optimal_range

    def _calculate_confidence(self, success_values: np.ndarray,
                             fail_values: List[float]) -> str:
        """Calculate confidence level in optimal range."""
        n_success = len(success_values)

        if n_success < 5:
            return "Low (need more data)"
        elif n_success < 20:
            return "Medium"
        else:
            # Check if success and fail ranges overlap
            if fail_values:
                fail_array = np.array(fail_values)
                success_range = (np.min(success_values), np.max(success_values))
                fail_range = (np.min(fail_array), np.max(fail_array))

                # Check overlap
                if success_range[1] < fail_range[0] or fail_range[1] < success_range[0]:
                    return "High (clear separation from failures)"
                else:
                    return "Medium (some overlap with failures)"
            else:
                return "High (sufficient data)"

    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive analysis summary.

        Returns:
            Dictionary with complete analysis results
        """
        stats = self.logger.get_statistics()

        report = {
            'overview': stats,
            'best_flights': [],
            'parameter_correlations': {},
            'optimal_ranges': {},
            'recommendations': []
        }

        # Get best flights with detailed analysis
        best = self.logger.get_best_flights(n=5)
        for flight in best:
            flight_data = self.logger.load_flight(flight['flight_id'])
            if flight_data:
                stability = self.analyze_stability(flight_data['telemetry'])
                flight_summary = {
                    'flight_id': flight['flight_id'],
                    'duration': flight['metadata']['duration'],
                    'parameters': flight['parameters'],
                    'stability_metrics': stability
                }
                report['best_flights'].append(flight_summary)

        # Parameter correlation analysis
        if stats['total_flights'] >= 5:
            correlations = self.correlate_parameters_with_success()
            report['parameter_correlations'] = correlations

            # Find optimal ranges for significant parameters
            for param, analysis in correlations.items():
                if 'correlation' in analysis and analysis['correlation'].get('significant', False):
                    optimal = self.find_optimal_ranges(param)
                    if 'error' not in optimal:
                        report['optimal_ranges'][param] = optimal

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        stats = report['overview']

        # Success rate recommendations
        if stats['success_rate'] < 0.5:
            recommendations.append(
                f"⚠️ Low success rate ({stats['success_rate']:.1%}). "
                "Consider using more conservative parameter ranges."
            )
        elif stats['success_rate'] > 0.8:
            recommendations.append(
                f"✓ Good success rate ({stats['success_rate']:.1%}). "
                "Can explore more aggressive parameter values."
            )

        # Parameter-specific recommendations
        for param, analysis in report.get('parameter_correlations', {}).items():
            if 'correlation' in analysis:
                corr = analysis['correlation']
                if corr.get('significant', False):
                    if abs(corr['coefficient']) > 0.5:
                        recommendations.append(
                            f"📊 {param}: {corr['interpretation']}"
                        )

        # Optimal range recommendations
        for param, optimal in report.get('optimal_ranges', {}).items():
            if optimal.get('confidence') in ['High', 'High (clear separation from failures)']:
                recommendations.append(
                    f"🎯 {param}: Recommended range "
                    f"[{optimal['recommended_min']:.4f}, {optimal['recommended_max']:.4f}] "
                    f"(confidence: {optimal['confidence']})"
                )

        # Data collection recommendations
        if stats['total_flights'] < 20:
            recommendations.append(
                f"📈 Collect more data ({stats['total_flights']} flights so far). "
                "Need 20+ flights for robust statistical analysis."
            )

        return recommendations

    def export_analysis_json(self, output_file: str):
        """
        Export complete analysis to JSON file.

        Args:
            output_file: Output JSON file path
        """
        report = self.generate_summary_report()

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            """Recursively convert numpy types to Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj

        report_clean = convert_numpy(report)

        with open(output_file, 'w') as f:
            json.dump(report_clean, f, indent=2)

        logger.info(f"Analysis exported to {output_file}")
