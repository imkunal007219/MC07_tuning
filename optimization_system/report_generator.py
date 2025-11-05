"""
Report Generator

Creates HTML reports with visualizations showing parameter effects.
Provides visual proof of which parameters work and why.
"""

import os
import logging
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive HTML reports with embedded plots.

    Creates visualizations for:
    - Parameter evolution across generations
    - Success/failure parameter distributions
    - Stability metrics over time
    - Correlation matrices
    - Optimal parameter ranges
    """

    def __init__(self, flight_logger, flight_analyzer):
        """
        Initialize report generator.

        Args:
            flight_logger: FlightDataLogger instance
            flight_analyzer: FlightAnalyzer instance
        """
        self.logger = flight_logger
        self.analyzer = flight_analyzer

    def generate_html_report(self, output_file: str = "optimization_report.html"):
        """
        Generate comprehensive HTML report.

        Args:
            output_file: Output HTML file path
        """
        logger.info(f"Generating optimization report: {output_file}")

        # Get analysis data
        report_data = self.analyzer.generate_summary_report()

        # Create plots directory
        plot_dir = os.path.join(os.path.dirname(output_file), "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Generate plots
        plots = self._generate_all_plots(report_data, plot_dir)

        # Generate HTML
        html = self._create_html(report_data, plots)

        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(html)

        logger.info(f"✓ Report generated: {output_file}")

    def _generate_all_plots(self, report_data: Dict, plot_dir: str) -> Dict[str, str]:
        """Generate all plots and return their file paths."""
        plots = {}

        try:
            # Plot 1: Success rate over generations
            plots['success_rate'] = self._plot_success_rate(plot_dir)

            # Plot 2: Parameter evolution
            plots['parameter_evolution'] = self._plot_parameter_evolution(plot_dir)

            # Plot 3: Parameter distributions (success vs failure)
            plots['parameter_distributions'] = self._plot_parameter_distributions(
                report_data.get('parameter_correlations', {}),
                plot_dir
            )

            # Plot 4: Stability metrics for best flights
            plots['stability_metrics'] = self._plot_stability_metrics(
                report_data.get('best_flights', []),
                plot_dir
            )

            # Plot 5: Parameter correlation heatmap
            plots['correlation_matrix'] = self._plot_correlation_matrix(
                report_data.get('parameter_correlations', {}),
                plot_dir
            )

        except Exception as e:
            logger.error(f"Error generating plots: {e}")

        return plots

    def _plot_success_rate(self, plot_dir: str) -> str:
        """Plot success rate over generations."""
        filename = os.path.join(plot_dir, "success_rate.png")

        # Group flights by generation
        generations = {}
        for flight in self.logger.flight_index:
            gen = flight.get('generation')
            if gen is not None:
                if gen not in generations:
                    generations[gen] = {'total': 0, 'success': 0}
                generations[gen]['total'] += 1
                if flight['success']:
                    generations[gen]['success'] += 1

        if not generations:
            logger.warning("No generation data available for success rate plot")
            return ""

        # Calculate success rates
        gen_nums = sorted(generations.keys())
        success_rates = [generations[g]['success'] / generations[g]['total']
                        for g in gen_nums]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(gen_nums, success_rates, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.title('Success Rate Evolution Across Generations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])

        # Add trend line
        if len(gen_nums) > 2:
            z = np.polyfit(gen_nums, success_rates, 1)
            p = np.poly1d(z)
            plt.plot(gen_nums, p(gen_nums), "r--", alpha=0.5, label='Trend')
            plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return os.path.basename(filename)

    def _plot_parameter_evolution(self, plot_dir: str) -> str:
        """Plot how parameters evolved over generations."""
        filename = os.path.join(plot_dir, "parameter_evolution.png")

        # Get all parameter names from best flights
        param_names = set()
        for flight in self.logger.get_successful_flights():
            param_names.update(flight['parameters'].keys())

        if not param_names:
            logger.warning("No parameters to plot")
            return ""

        # Limit to top 6 most important parameters (for readability)
        param_names = sorted(param_names)[:6]

        # Group by generation
        gen_data = {}
        for flight in self.logger.flight_index:
            gen = flight.get('generation')
            if gen is not None and flight['success']:
                if gen not in gen_data:
                    gen_data[gen] = {p: [] for p in param_names}

                for param in param_names:
                    if param in flight['parameters']:
                        gen_data[gen][param].append(flight['parameters'][param])

        if not gen_data:
            logger.warning("No generation data for parameter evolution")
            return ""

        # Calculate averages per generation
        gen_nums = sorted(gen_data.keys())
        param_avgs = {p: [] for p in param_names}

        for gen in gen_nums:
            for param in param_names:
                if gen_data[gen][param]:
                    param_avgs[param].append(np.mean(gen_data[gen][param]))
                else:
                    param_avgs[param].append(None)

        # Create plot with subplots
        n_params = len(param_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, param in enumerate(param_names):
            if idx >= len(axes):
                break

            ax = axes[idx]
            values = param_avgs[param]

            # Filter out None values
            valid_gens = [gen_nums[i] for i, v in enumerate(values) if v is not None]
            valid_vals = [v for v in values if v is not None]

            if valid_vals:
                ax.plot(valid_gens, valid_vals, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Generation', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                ax.set_title(param, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Parameter Evolution (Successful Flights Only)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return os.path.basename(filename)

    def _plot_parameter_distributions(self, correlations: Dict, plot_dir: str) -> str:
        """Plot parameter value distributions for success vs failure."""
        filename = os.path.join(plot_dir, "parameter_distributions.png")

        # Find parameters with significant correlations
        significant_params = []
        for param, data in correlations.items():
            if 'correlation' in data and data['correlation'].get('significant', False):
                significant_params.append(param)

        if not significant_params:
            logger.warning("No significant parameters to plot distributions")
            return ""

        # Limit to top 6
        significant_params = significant_params[:6]

        # Create subplots
        n_params = len(significant_params)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, param in enumerate(significant_params):
            if idx >= len(axes):
                break

            ax = axes[idx]
            data = correlations[param]

            success_vals = data['success_values']['values']
            fail_vals = data['fail_values']['values']

            # Create histograms
            bins = 15
            if success_vals:
                ax.hist(success_vals, bins=bins, alpha=0.6, label='Success', color='green')
            if fail_vals:
                ax.hist(fail_vals, bins=bins, alpha=0.6, label='Failure', color='red')

            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(param, fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Parameter Value Distributions: Success vs Failure', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return os.path.basename(filename)

    def _plot_stability_metrics(self, best_flights: List[Dict], plot_dir: str) -> str:
        """Plot stability metrics for best flights."""
        filename = os.path.join(plot_dir, "stability_metrics.png")

        if not best_flights:
            logger.warning("No flight data for stability metrics plot")
            return ""

        # Extract metrics
        metrics_to_plot = [
            'oscillation_score',
            'overshoot_percentage',
            'settling_time',
            'steady_state_error',
            'altitude_stability'
        ]

        flight_ids = [f['flight_id'][:8] for f in best_flights]  # Truncate IDs
        metric_data = {m: [] for m in metrics_to_plot}

        for flight in best_flights:
            metrics = flight.get('stability_metrics', {})
            for metric in metrics_to_plot:
                metric_data[metric].append(metrics.get(metric, 0))

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]
            values = metric_data[metric]

            ax.bar(range(len(values)), values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(flight_ids)))
            ax.set_xticklabels(flight_ids, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide extra subplot
        axes[5].axis('off')

        plt.suptitle('Stability Metrics for Best Flights', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return os.path.basename(filename)

    def _plot_correlation_matrix(self, correlations: Dict, plot_dir: str) -> str:
        """Plot correlation matrix showing which parameters affect success."""
        filename = os.path.join(plot_dir, "correlation_matrix.png")

        if not correlations:
            logger.warning("No correlation data for matrix plot")
            return ""

        # Extract correlation coefficients
        params = []
        coeffs = []

        for param, data in correlations.items():
            if 'correlation' in data:
                params.append(param)
                coeffs.append(data['correlation']['coefficient'])

        if not params:
            logger.warning("No correlation coefficients available")
            return ""

        # Sort by absolute correlation
        sorted_indices = np.argsort(np.abs(coeffs))[::-1]
        params = [params[i] for i in sorted_indices[:10]]  # Top 10
        coeffs = [coeffs[i] for i in sorted_indices[:10]]

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green' if c > 0 else 'red' for c in coeffs]
        bars = ax.barh(range(len(coeffs)), coeffs, color=colors, alpha=0.7)

        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Parameter Correlation with Success\n(Green = Higher is Better, Red = Lower is Better)',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, coeff) in enumerate(zip(bars, coeffs)):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                   f'{coeff:.3f}', ha=ha, va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        return os.path.basename(filename)

    def _create_html(self, report_data: Dict, plots: Dict[str, str]) -> str:
        """Create HTML report with embedded plots."""
        stats = report_data['overview']
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Drone Parameter Optimization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            list-style-type: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 10px;
            margin: 5px 0;
            background-color: white;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }}
        .parameter-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .parameter-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .parameter-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .parameter-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .failure {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚁 Drone Parameter Optimization Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <h2>📊 Overview Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Flights</div>
                <div class="stat-value">{stats['total_flights']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Successful</div>
                <div class="stat-value success">{stats['successful']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value failure">{stats['failed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{stats['success_rate']:.1%}</div>
            </div>
        </div>

        <h2>🎯 Recommendations</h2>
        <div class="recommendations">
            <ul>
"""

        for rec in report_data.get('recommendations', []):
            html += f"                <li>{rec}</li>\n"

        html += """            </ul>
        </div>
"""

        # Add plots
        if plots.get('success_rate'):
            html += f"""
        <h2>📈 Success Rate Evolution</h2>
        <div class="plot">
            <img src="plots/{plots['success_rate']}" alt="Success Rate Evolution">
        </div>
"""

        if plots.get('parameter_evolution'):
            html += f"""
        <h2>🔧 Parameter Evolution</h2>
        <div class="plot">
            <img src="plots/{plots['parameter_evolution']}" alt="Parameter Evolution">
        </div>
"""

        if plots.get('correlation_matrix'):
            html += f"""
        <h2>🔗 Parameter Correlations with Success</h2>
        <div class="plot">
            <img src="plots/{plots['correlation_matrix']}" alt="Correlation Matrix">
        </div>
"""

        if plots.get('parameter_distributions'):
            html += f"""
        <h2>📊 Parameter Distributions</h2>
        <div class="plot">
            <img src="plots/{plots['parameter_distributions']}" alt="Parameter Distributions">
        </div>
"""

        if plots.get('stability_metrics'):
            html += f"""
        <h2>⚖️ Stability Metrics</h2>
        <div class="plot">
            <img src="plots/{plots['stability_metrics']}" alt="Stability Metrics">
        </div>
"""

        # Add optimal ranges table
        optimal_ranges = report_data.get('optimal_ranges', {})
        if optimal_ranges:
            html += """
        <h2>🎯 Optimal Parameter Ranges</h2>
        <table class="parameter-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Recommended Min</th>
                    <th>Recommended Max</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
"""
            for param, data in optimal_ranges.items():
                html += f"""                <tr>
                    <td><strong>{param}</strong></td>
                    <td>{data['recommended_min']:.4f}</td>
                    <td>{data['recommended_max']:.4f}</td>
                    <td>{data['confidence']}</td>
                </tr>
"""
            html += """            </tbody>
        </table>
"""

        # Close HTML
        html += """
    </div>
</body>
</html>
"""

        return html
