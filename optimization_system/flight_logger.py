"""
Flight Data Logger

Logs all flight data with parameters for automated analysis.
Each flight is stored with complete telemetry and parameter set for comparison.
"""

import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List
import pickle
import gzip

logger = logging.getLogger(__name__)


class FlightDataLogger:
    """
    Logs flight data with parameters for automated analysis.

    Features:
    - Stores complete telemetry with parameter set
    - Automatic file management
    - Compressed storage for large datasets
    - JSON metadata for easy querying
    - Links parameters to outcomes
    """

    def __init__(self, log_dir: str = "flight_logs"):
        """
        Initialize flight data logger.

        Args:
            log_dir: Directory to store flight logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Index file tracks all flights
        self.index_file = os.path.join(log_dir, "flight_index.json")
        self.flight_index = self._load_index()

    def _load_index(self) -> List[Dict]:
        """Load flight index."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load index: {e}")
        return []

    def _save_index(self):
        """Save flight index."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.flight_index, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save index: {e}")

    def log_flight(self, parameters: Dict[str, float], telemetry: Dict,
                   success: bool, generation: int = None,
                   individual_id: int = None) -> str:
        """
        Log a flight with complete data.

        Args:
            parameters: Parameter set used for this flight
            telemetry: Telemetry data from flight
            success: Whether flight completed successfully
            generation: Optimization generation number
            individual_id: Individual ID in population

        Returns:
            Flight ID (filename)
        """
        timestamp = datetime.now()
        flight_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        # Create flight record
        flight_data = {
            'flight_id': flight_id,
            'timestamp': timestamp.isoformat(),
            'generation': generation,
            'individual_id': individual_id,
            'parameters': parameters,
            'success': success,
            'telemetry': self._serialize_telemetry(telemetry),
            'metadata': {
                'duration': float(telemetry['time'][-1]) if 'time' in telemetry and len(telemetry['time']) > 0 else 0,
                'samples': len(telemetry['time']) if 'time' in telemetry else 0,
                'crashed': telemetry.get('metrics', {}).get('crashed', not success),
            }
        }

        # Save flight data (compressed)
        data_file = os.path.join(self.log_dir, f"{flight_id}.pkl.gz")
        try:
            with gzip.open(data_file, 'wb') as f:
                pickle.dump(flight_data, f)
            logger.debug(f"Flight data saved: {data_file}")
        except Exception as e:
            logger.error(f"Failed to save flight data: {e}")
            return None

        # Update index (metadata only, no telemetry)
        index_entry = {
            'flight_id': flight_id,
            'timestamp': timestamp.isoformat(),
            'generation': generation,
            'individual_id': individual_id,
            'success': success,
            'parameters': parameters,
            'metadata': flight_data['metadata'],
            'data_file': data_file
        }
        self.flight_index.append(index_entry)
        self._save_index()

        logger.info(f"Flight logged: {flight_id} (Success: {success})")
        return flight_id

    def _serialize_telemetry(self, telemetry: Dict) -> Dict:
        """Convert numpy arrays to lists for JSON compatibility."""
        serialized = {}
        for key, value in telemetry.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_telemetry(value)
            else:
                serialized[key] = value
        return serialized

    def load_flight(self, flight_id: str) -> Dict:
        """
        Load flight data by ID.

        Args:
            flight_id: Flight ID to load

        Returns:
            Flight data dictionary
        """
        data_file = os.path.join(self.log_dir, f"{flight_id}.pkl.gz")

        if not os.path.exists(data_file):
            logger.error(f"Flight data not found: {flight_id}")
            return None

        try:
            with gzip.open(data_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load flight data: {e}")
            return None

    def get_successful_flights(self) -> List[Dict]:
        """Get all successful flights from index."""
        return [f for f in self.flight_index if f['success']]

    def get_failed_flights(self) -> List[Dict]:
        """Get all failed flights from index."""
        return [f for f in self.flight_index if not f['success']]

    def get_flights_by_generation(self, generation: int) -> List[Dict]:
        """Get all flights from a specific generation."""
        return [f for f in self.flight_index if f['generation'] == generation]

    def get_best_flights(self, n: int = 10) -> List[Dict]:
        """
        Get top N best flights based on success and duration.

        Args:
            n: Number of flights to return

        Returns:
            List of best flight metadata
        """
        successful = self.get_successful_flights()
        # Sort by duration (longer is better for stability)
        sorted_flights = sorted(
            successful,
            key=lambda x: x['metadata'].get('duration', 0),
            reverse=True
        )
        return sorted_flights[:n]

    def compare_parameters(self, param_name: str) -> Dict:
        """
        Analyze effect of a specific parameter across all flights.

        Args:
            param_name: Parameter name to analyze

        Returns:
            Analysis results
        """
        successful = self.get_successful_flights()
        failed = self.get_failed_flights()

        if not successful:
            return {'error': 'No successful flights to analyze'}

        # Extract parameter values and outcomes
        success_values = [f['parameters'].get(param_name, None) for f in successful]
        fail_values = [f['parameters'].get(param_name, None) for f in failed]

        # Remove None values
        success_values = [v for v in success_values if v is not None]
        fail_values = [v for v in fail_values if v is not None]

        return {
            'parameter': param_name,
            'successful_flights': len(successful),
            'failed_flights': len(failed),
            'success_values': {
                'mean': float(np.mean(success_values)) if success_values else None,
                'std': float(np.std(success_values)) if success_values else None,
                'min': float(np.min(success_values)) if success_values else None,
                'max': float(np.max(success_values)) if success_values else None,
                'values': success_values
            },
            'fail_values': {
                'mean': float(np.mean(fail_values)) if fail_values else None,
                'std': float(np.std(fail_values)) if fail_values else None,
                'min': float(np.min(fail_values)) if fail_values else None,
                'max': float(np.max(fail_values)) if fail_values else None,
                'values': fail_values
            }
        }

    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        total = len(self.flight_index)
        successful = len(self.get_successful_flights())
        failed = len(self.get_failed_flights())

        return {
            'total_flights': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'generations': len(set(f['generation'] for f in self.flight_index if f['generation'] is not None))
        }

    def export_csv(self, output_file: str):
        """
        Export flight index to CSV for external analysis.

        Args:
            output_file: Output CSV file path
        """
        import csv

        if not self.flight_index:
            logger.warning("No flights to export")
            return

        # Get all parameter names
        all_params = set()
        for flight in self.flight_index:
            all_params.update(flight['parameters'].keys())

        # Write CSV
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['flight_id', 'timestamp', 'generation', 'individual_id', 'success', 'duration', 'crashed']
            fieldnames.extend(sorted(all_params))

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for flight in self.flight_index:
                row = {
                    'flight_id': flight['flight_id'],
                    'timestamp': flight['timestamp'],
                    'generation': flight['generation'],
                    'individual_id': flight['individual_id'],
                    'success': flight['success'],
                    'duration': flight['metadata'].get('duration', 0),
                    'crashed': flight['metadata'].get('crashed', False),
                }
                # Add parameters
                for param in all_params:
                    row[param] = flight['parameters'].get(param, '')

                writer.writerow(row)

        logger.info(f"Exported {len(self.flight_index)} flights to {output_file}")
