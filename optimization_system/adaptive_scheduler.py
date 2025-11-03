"""
Adaptive Test Duration Scheduler

Dynamically adjusts test duration based on parameter quality.
Short tests for obviously bad parameters, longer for promising ones.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class AdaptiveScheduler:
    """
    Determines optimal test duration based on parameter fitness history

    Provides ~20% speedup by avoiding long tests on obviously bad parameters.
    """

    def __init__(self,
                 min_duration: float = 10.0,
                 normal_duration: float = 30.0,
                 max_duration: float = 60.0,
                 bad_threshold: float = -100.0,
                 good_threshold: float = 50.0):
        """
        Initialize adaptive scheduler

        Args:
            min_duration: Minimum test duration for bad parameters (seconds)
            normal_duration: Standard test duration (seconds)
            max_duration: Maximum test duration for excellent parameters (seconds)
            bad_threshold: Fitness below this gets min_duration
            good_threshold: Fitness above this gets max_duration
        """
        self.min_duration = min_duration
        self.normal_duration = normal_duration
        self.max_duration = max_duration
        self.bad_threshold = bad_threshold
        self.good_threshold = good_threshold

        # Track fitness history for adaptive decisions
        self.fitness_history: List[float] = []
        self.running_avg: Optional[float] = None
        self.running_std: Optional[float] = None

    def get_test_duration(self, predicted_fitness: Optional[float] = None,
                         generation: int = 0) -> float:
        """
        Determine optimal test duration

        Args:
            predicted_fitness: Predicted fitness (if available from surrogate model)
            generation: Current generation number

        Returns:
            Test duration in seconds
        """
        # Early generations: use normal duration to build history
        if generation < 5 or len(self.fitness_history) < 10:
            return self.normal_duration

        # Use predicted fitness if available
        if predicted_fitness is not None:
            return self._duration_from_fitness(predicted_fitness)

        # No prediction available: use normal duration
        return self.normal_duration

    def get_test_duration_from_params(self, parameters: Dict[str, float],
                                     historical_params: List[Dict[str, float]],
                                     historical_fitness: List[float],
                                     generation: int = 0) -> float:
        """
        Determine test duration based on parameter similarity to history

        Args:
            parameters: Current parameters to test
            historical_params: Previously tested parameter sets
            historical_fitness: Fitness scores for historical params
            generation: Current generation

        Returns:
            Test duration in seconds
        """
        # Early generations: use normal duration
        if generation < 5 or len(historical_fitness) < 10:
            return self.normal_duration

        # Find similar parameters in history
        similarities = []
        for hist_params, hist_fitness in zip(historical_params, historical_fitness):
            similarity = self._calculate_similarity(parameters, hist_params)
            similarities.append((similarity, hist_fitness))

        # If we have similar parameters, use their fitness to predict
        similarities.sort(reverse=True)  # Most similar first
        if similarities[0][0] > 0.8:  # Very similar parameters found
            predicted_fitness = similarities[0][1]
            logger.debug(f"Using similar params fitness: {predicted_fitness:.2f}")
            return self._duration_from_fitness(predicted_fitness)

        # No similar parameters: use normal duration
        return self.normal_duration

    def _duration_from_fitness(self, fitness: float) -> float:
        """
        Map fitness value to test duration

        Args:
            fitness: Fitness score

        Returns:
            Test duration in seconds
        """
        if fitness < self.bad_threshold:
            # Very bad parameters: quick test
            duration = self.min_duration
            logger.debug(f"Bad params (fitness={fitness:.1f}): {duration}s test")

        elif fitness < self.good_threshold:
            # Medium parameters: normal test
            # Linear interpolation between min and normal
            ratio = (fitness - self.bad_threshold) / (self.good_threshold - self.bad_threshold)
            duration = self.min_duration + ratio * (self.normal_duration - self.min_duration)
            logger.debug(f"Medium params (fitness={fitness:.1f}): {duration:.1f}s test")

        else:
            # Good parameters: thorough test
            # Linear interpolation between normal and max
            if fitness > self.good_threshold + 50:
                duration = self.max_duration
            else:
                ratio = (fitness - self.good_threshold) / 50.0
                duration = self.normal_duration + ratio * (self.max_duration - self.normal_duration)
            logger.debug(f"Good params (fitness={fitness:.1f}): {duration:.1f}s test")

        return max(self.min_duration, min(self.max_duration, duration))

    def _calculate_similarity(self, params1: Dict[str, float],
                             params2: Dict[str, float]) -> float:
        """
        Calculate similarity between two parameter sets

        Args:
            params1: First parameter set
            params2: Second parameter set

        Returns:
            Similarity score [0, 1], where 1 is identical
        """
        # Get common parameter names
        common_keys = set(params1.keys()) & set(params2.keys())

        if not common_keys:
            return 0.0

        # Calculate normalized Euclidean distance
        differences = []
        for key in common_keys:
            val1 = params1[key]
            val2 = params2[key]

            # Normalize difference (assuming parameters are 0-1 normalized)
            diff = abs(val1 - val2)
            differences.append(diff)

        # Convert distance to similarity
        avg_diff = np.mean(differences)
        similarity = 1.0 - min(1.0, avg_diff)

        return similarity

    def update_history(self, fitness: float):
        """
        Update fitness history with new result

        Args:
            fitness: New fitness value
        """
        self.fitness_history.append(fitness)

        # Keep only recent history (last 100 trials)
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]

        # Update running statistics
        if len(self.fitness_history) >= 5:
            self.running_avg = np.mean(self.fitness_history)
            self.running_std = np.std(self.fitness_history)

    def get_statistics(self) -> Dict:
        """
        Get scheduler statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'history_size': len(self.fitness_history),
            'running_avg': self.running_avg,
            'running_std': self.running_std,
            'min_duration': self.min_duration,
            'normal_duration': self.normal_duration,
            'max_duration': self.max_duration
        }

    def estimate_speedup(self) -> float:
        """
        Estimate speedup from adaptive scheduling

        Returns:
            Estimated speedup percentage
        """
        if len(self.fitness_history) < 10:
            return 0.0

        # Estimate how many trials would use shorter durations
        bad_count = sum(1 for f in self.fitness_history if f < self.bad_threshold)
        good_count = sum(1 for f in self.fitness_history if f > self.good_threshold)

        # Calculate weighted average duration
        total = len(self.fitness_history)
        avg_duration = (
            (bad_count / total) * self.min_duration +
            ((total - bad_count - good_count) / total) * self.normal_duration +
            (good_count / total) * self.max_duration
        )

        # Speedup compared to always using normal duration
        speedup = (1.0 - avg_duration / self.normal_duration) * 100

        return max(0.0, speedup)
