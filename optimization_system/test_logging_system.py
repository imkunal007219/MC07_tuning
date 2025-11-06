#!/usr/bin/env python3
"""
Test script for automated logging and analysis system.

Tests:
1. Flight data logging
2. Analysis engine
3. Report generation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .flight_logger import FlightDataLogger
from .flight_analyzer import FlightAnalyzer
from .report_generator import ReportGenerator


def create_sample_telemetry(success: bool = True, duration: float = 30.0):
    """Create sample telemetry data for testing."""
    # Simulate 20Hz data collection
    num_samples = int(duration * 20)
    time = np.linspace(0, duration, num_samples)

    if success:
        # Successful flight - stable
        roll = 2.0 * np.sin(0.5 * time) + np.random.normal(0, 0.5, num_samples)
        pitch = 1.5 * np.sin(0.3 * time) + np.random.normal(0, 0.4, num_samples)
        altitude = 10.0 + np.random.normal(0, 0.2, num_samples)
    else:
        # Failed flight - unstable oscillations
        roll = 15.0 * np.sin(5 * time) + np.random.normal(0, 5, num_samples)
        pitch = 12.0 * np.sin(4 * time) + np.random.normal(0, 4, num_samples)
        altitude = 10.0 - 0.3 * time + np.random.normal(0, 1, num_samples)  # Descending

    telemetry = {
        'time': time,
        'latitude': -35.363261 + np.random.normal(0, 0.00001, num_samples),
        'longitude': 149.165237 + np.random.normal(0, 0.00001, num_samples),
        'altitude': altitude,
        'roll': roll,
        'pitch': pitch,
        'yaw': np.random.normal(0, 1, num_samples),
        'roll_rate': np.gradient(roll) * 20,  # Derivative
        'pitch_rate': np.gradient(pitch) * 20,
        'yaw_rate': np.random.normal(0, 2, num_samples),
        'vx': np.random.normal(0, 0.5, num_samples),
        'vy': np.random.normal(0, 0.5, num_samples),
        'vz': np.random.normal(0, 0.2, num_samples),
        'metrics': {
            'duration': float(duration),
            'max_altitude': float(np.max(altitude)),
            'max_roll': float(np.max(np.abs(roll))),
            'max_pitch': float(np.max(np.abs(pitch))),
            'crashed': not success,
        }
    }

    return telemetry


def test_logging():
    """Test flight data logging."""
    print("="*60)
    print("TEST 1: Flight Data Logging")
    print("="*60)

    logger = FlightDataLogger(log_dir="test_flight_logs")

    # Log some test flights
    test_params_sets = [
        # Successful flights with good parameters
        {'ATC_RAT_RLL_P': 0.15, 'ATC_RAT_RLL_I': 0.09, 'ATC_RAT_RLL_D': 0.012},
        {'ATC_RAT_RLL_P': 0.16, 'ATC_RAT_RLL_I': 0.10, 'ATC_RAT_RLL_D': 0.013},
        {'ATC_RAT_RLL_P': 0.17, 'ATC_RAT_RLL_I': 0.095, 'ATC_RAT_RLL_D': 0.011},
        {'ATC_RAT_RLL_P': 0.155, 'ATC_RAT_RLL_I': 0.092, 'ATC_RAT_RLL_D': 0.014},

        # Failed flights with poor parameters
        {'ATC_RAT_RLL_P': 0.05, 'ATC_RAT_RLL_I': 0.02, 'ATC_RAT_RLL_D': 0.002},
        {'ATC_RAT_RLL_P': 0.08, 'ATC_RAT_RLL_I': 0.03, 'ATC_RAT_RLL_D': 0.003},
    ]

    for i, params in enumerate(test_params_sets):
        success = i < 4  # First 4 are successful
        generation = i // 2  # 2 flights per generation
        telemetry = create_sample_telemetry(success=success, duration=30.0)

        flight_id = logger.log_flight(
            parameters=params,
            telemetry=telemetry,
            success=success,
            generation=generation,
            individual_id=i
        )

        status = "SUCCESS" if success else "FAILURE"
        print(f"  Logged flight {flight_id}: {status}")

    # Show statistics
    stats = logger.get_statistics()
    print(f"\n✓ Logged {stats['total_flights']} flights")
    print(f"  - Successful: {stats['successful']}")
    print(f"  - Failed: {stats['failed']}")
    print(f"  - Success rate: {stats['success_rate']:.1%}")

    return logger


def test_analysis(logger):
    """Test flight analysis."""
    print("\n" + "="*60)
    print("TEST 2: Flight Analysis")
    print("="*60)

    analyzer = FlightAnalyzer(logger)

    # Test parameter correlation
    print("\nParameter Correlation Analysis:")
    correlations = analyzer.correlate_parameters_with_success(['ATC_RAT_RLL_P'])

    if 'ATC_RAT_RLL_P' in correlations:
        corr_data = correlations['ATC_RAT_RLL_P']
        if 'correlation' in corr_data:
            corr = corr_data['correlation']
            print(f"  ATC_RAT_RLL_P:")
            print(f"    Coefficient: {corr['coefficient']:.3f}")
            print(f"    P-value: {corr['p_value']:.4f}")
            print(f"    Significant: {corr['significant']}")
            print(f"    Interpretation: {corr['interpretation']}")

    # Test optimal range finding
    print("\nOptimal Range Analysis:")
    optimal = analyzer.find_optimal_ranges('ATC_RAT_RLL_P')
    if 'error' not in optimal:
        print(f"  ATC_RAT_RLL_P:")
        print(f"    Recommended: [{optimal['recommended_min']:.4f}, {optimal['recommended_max']:.4f}]")
        print(f"    Confidence: {optimal['confidence']}")

    # Test stability analysis
    print("\nStability Analysis (loading first flight):")
    best_flights = logger.get_successful_flights()
    if best_flights:
        flight_data = logger.load_flight(best_flights[0]['flight_id'])
        if flight_data:
            stability = analyzer.analyze_stability(flight_data['telemetry'])
            print(f"  Oscillation score: {stability.get('oscillation_score', 0):.4f}")
            print(f"  Max angle deviation: {stability.get('max_angle_deviation', 0):.2f}°")
            print(f"  Altitude RMSE: {stability.get('altitude_rmse', 0):.4f}m")

    print("\n✓ Analysis complete")
    return analyzer


def test_report_generation(logger, analyzer):
    """Test report generation."""
    print("\n" + "="*60)
    print("TEST 3: Report Generation")
    print("="*60)

    generator = ReportGenerator(logger, analyzer)

    # Generate HTML report
    try:
        os.makedirs("test_reports", exist_ok=True)
        report_file = "test_reports/test_report.html"

        print(f"  Generating report: {report_file}")
        generator.generate_html_report(report_file)

        # Check if file exists
        if os.path.exists(report_file):
            size = os.path.getsize(report_file)
            print(f"  ✓ Report generated successfully ({size} bytes)")
            print(f"  Open with: firefox {report_file}")
        else:
            print("  ✗ Report file not found")

    except Exception as e:
        print(f"  ✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test JSON export
    try:
        json_file = "test_reports/analysis.json"
        print(f"\n  Exporting JSON: {json_file}")
        analyzer.export_analysis_json(json_file)

        if os.path.exists(json_file):
            print(f"  ✓ JSON export successful")
        else:
            print("  ✗ JSON file not found")

    except Exception as e:
        print(f"  ✗ JSON export failed: {e}")

    # Test CSV export
    try:
        csv_file = "test_reports/flights.csv"
        print(f"\n  Exporting CSV: {csv_file}")
        logger.export_csv(csv_file)

        if os.path.exists(csv_file):
            print(f"  ✓ CSV export successful")
        else:
            print("  ✗ CSV file not found")

    except Exception as e:
        print(f"  ✗ CSV export failed: {e}")

    print("\n✓ Report generation complete")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AUTOMATED LOGGING & ANALYSIS SYSTEM TEST")
    print("="*60)

    try:
        # Test 1: Logging
        logger = test_logging()

        # Test 2: Analysis
        analyzer = test_analysis(logger)

        # Test 3: Report generation
        test_report_generation(logger, analyzer)

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        print("\nNext steps:")
        print("1. Check test_reports/test_report.html in your browser")
        print("2. Review test_reports/analysis.json for raw data")
        print("3. Open test_reports/flights.csv in a spreadsheet")
        print("\nThe logging system is ready for use in optimization!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
