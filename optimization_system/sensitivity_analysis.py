"""
Parameter Sensitivity Analysis

Analyzes which parameters have the most impact on drone performance.
Helps focus tuning efforts on the most critical parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """
    Analyzes parameter sensitivity and importance

    Uses multiple methods:
    - Correlation analysis
    - Random Forest feature importance
    - Permutation importance
    - Variance-based sensitivity
    """

    def __init__(self, parameter_names: List[str]):
        """
        Initialize sensitivity analyzer

        Args:
            parameter_names: Names of parameters to analyze
        """
        self.parameter_names = parameter_names
        self.results = {}

    def analyze(self, parameter_history: List[Dict[str, float]],
               fitness_history: List[float]) -> Dict:
        """
        Perform comprehensive sensitivity analysis

        Args:
            parameter_history: List of parameter dictionaries
            fitness_history: Corresponding fitness values

        Returns:
            Dictionary with analysis results
        """
        if len(parameter_history) < 10:
            logger.warning("Not enough data for sensitivity analysis (need at least 10 samples)")
            return {}

        logger.info("Performing parameter sensitivity analysis...")

        # Convert to DataFrame for easier manipulation
        df_params = pd.DataFrame(parameter_history)
        df_fitness = pd.Series(fitness_history, name='fitness')

        # Run different analysis methods
        self.results['correlation'] = self._correlation_analysis(df_params, df_fitness)
        self.results['variance'] = self._variance_analysis(df_params, df_fitness)
        self.results['random_forest'] = self._random_forest_importance(df_params, df_fitness)

        # Aggregate results
        self.results['summary'] = self._aggregate_rankings()

        logger.info("Sensitivity analysis complete")
        return self.results

    def _correlation_analysis(self, df_params: pd.DataFrame,
                              df_fitness: pd.Series) -> Dict:
        """
        Analyze correlation between parameters and fitness

        Args:
            df_params: Parameter DataFrame
            df_fitness: Fitness Series

        Returns:
            Dictionary with correlation results
        """
        correlations = {}

        for param in self.parameter_names:
            if param in df_params.columns:
                # Pearson correlation (linear relationship)
                pearson_corr, pearson_p = pearsonr(df_params[param], df_fitness)

                # Spearman correlation (monotonic relationship)
                spearman_corr, spearman_p = spearmanr(df_params[param], df_fitness)

                correlations[param] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'significant': pearson_p < 0.05,  # Statistical significance
                    'importance': abs(pearson_corr)  # Absolute correlation as importance
                }

        return correlations

    def _variance_analysis(self, df_params: pd.DataFrame,
                          df_fitness: pd.Series) -> Dict:
        """
        Analyze how parameter variance affects fitness variance

        Args:
            df_params: Parameter DataFrame
            df_fitness: Fitness Series

        Returns:
            Dictionary with variance analysis results
        """
        variance_results = {}

        for param in self.parameter_names:
            if param in df_params.columns:
                # Split data into quartiles based on parameter value
                param_quartiles = pd.qcut(df_params[param], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                         duplicates='drop')

                # Calculate fitness variance in each quartile
                quartile_fitness = df_fitness.groupby(param_quartiles)

                fitness_by_quartile = quartile_fitness.mean()
                variance_across_quartiles = quartile_fitness.var().mean()

                # Range of fitness across quartiles
                fitness_range = fitness_by_quartile.max() - fitness_by_quartile.min()

                variance_results[param] = {
                    'fitness_range': fitness_range,
                    'variance': variance_across_quartiles,
                    'importance': fitness_range / (df_fitness.std() + 1e-6)  # Normalized importance
                }

        return variance_results

    def _random_forest_importance(self, df_params: pd.DataFrame,
                                  df_fitness: pd.Series) -> Dict:
        """
        Use Random Forest to determine parameter importance

        Args:
            df_params: Parameter DataFrame
            df_fitness: Fitness Series

        Returns:
            Dictionary with Random Forest importance results
        """
        try:
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                      max_depth=10, min_samples_split=5)
            rf.fit(df_params, df_fitness)

            # Get feature importances
            importances = rf.feature_importances_

            # Calculate permutation importance
            perm_importance = permutation_importance(rf, df_params, df_fitness,
                                                    n_repeats=10, random_state=42)

            results = {}
            for i, param in enumerate(df_params.columns):
                results[param] = {
                    'gini_importance': importances[i],
                    'permutation_importance': perm_importance.importances_mean[i],
                    'permutation_std': perm_importance.importances_std[i],
                    'importance': perm_importance.importances_mean[i]  # Use permutation as primary
                }

            return results

        except Exception as e:
            logger.warning(f"Random Forest analysis failed: {e}")
            return {}

    def _aggregate_rankings(self) -> Dict:
        """
        Aggregate results from different methods into final ranking

        Returns:
            Dictionary with aggregated rankings
        """
        aggregated = {}

        for param in self.parameter_names:
            scores = []

            # Correlation importance
            if 'correlation' in self.results and param in self.results['correlation']:
                scores.append(self.results['correlation'][param]['importance'])

            # Variance importance
            if 'variance' in self.results and param in self.results['variance']:
                scores.append(self.results['variance'][param]['importance'])

            # Random Forest importance
            if 'random_forest' in self.results and param in self.results['random_forest']:
                scores.append(self.results['random_forest'][param]['importance'])

            # Aggregate score (average of available scores)
            aggregated[param] = {
                'importance_score': np.mean(scores) if scores else 0.0,
                'methods_used': len(scores),
                'raw_scores': scores
            }

        # Sort by importance
        sorted_params = sorted(aggregated.items(),
                             key=lambda x: x[1]['importance_score'],
                             reverse=True)

        return {
            'rankings': sorted_params,
            'top_5': [p[0] for p in sorted_params[:5]],
            'bottom_5': [p[0] for p in sorted_params[-5:]]
        }

    def plot_sensitivity(self, output_path: str = "/tmp/parameter_sensitivity.png"):
        """
        Create visualization of parameter sensitivity

        Args:
            output_path: Path to save plot
        """
        if 'summary' not in self.results:
            logger.warning("No analysis results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Overall importance ranking
        ax = axes[0, 0]
        rankings = self.results['summary']['rankings']
        params = [r[0] for r in rankings[:10]]  # Top 10
        scores = [r[1]['importance_score'] for r in rankings[:10]]

        ax.barh(params, scores, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 10 Most Important Parameters')
        ax.invert_yaxis()

        # Plot 2: Correlation heatmap
        ax = axes[0, 1]
        if 'correlation' in self.results:
            corr_data = {k: v['pearson_corr'] for k, v in self.results['correlation'].items()}
            params_sorted = sorted(corr_data.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            params = [p[0] for p in params_sorted]
            correlations = [p[1] for p in params_sorted]

            colors = ['green' if c > 0 else 'red' for c in correlations]
            ax.barh(params, [abs(c) for c in correlations], color=colors, alpha=0.7)
            ax.set_xlabel('|Correlation with Fitness|')
            ax.set_title('Parameter-Fitness Correlation')
            ax.invert_yaxis()

        # Plot 3: Variance contribution
        ax = axes[1, 0]
        if 'variance' in self.results:
            var_data = {k: v['fitness_range'] for k, v in self.results['variance'].items()}
            params_sorted = sorted(var_data.items(), key=lambda x: x[1], reverse=True)[:10]
            params = [p[0] for p in params_sorted]
            ranges = [p[1] for p in params_sorted]

            ax.bar(range(len(params)), ranges, color='coral')
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_ylabel('Fitness Range')
            ax.set_title('Parameter Impact on Fitness Variance')

        # Plot 4: Method agreement
        ax = axes[1, 1]
        if 'summary' in self.results:
            rankings = self.results['summary']['rankings']
            params = [r[0] for r in rankings[:10]]
            methods_used = [r[1]['methods_used'] for r in rankings[:10]]

            ax.bar(range(len(params)), methods_used, color='lightgreen')
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels(params, rotation=45, ha='right')
            ax.set_ylabel('Number of Methods')
            ax.set_title('Method Agreement (Top 10)')
            ax.set_ylim([0, 3.5])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Sensitivity plot saved to {output_path}")

    def generate_report(self) -> str:
        """
        Generate text report of sensitivity analysis

        Returns:
            Formatted report string
        """
        if 'summary' not in self.results:
            return "No analysis results available"

        report = []
        report.append("=" * 60)
        report.append("PARAMETER SENSITIVITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Top parameters
        report.append("TOP 5 MOST IMPORTANT PARAMETERS:")
        for i, param in enumerate(self.results['summary']['top_5'], 1):
            ranking = next(r for r in self.results['summary']['rankings'] if r[0] == param)
            score = ranking[1]['importance_score']
            report.append(f"  {i}. {param:30s} Score: {score:.4f}")

        report.append("")
        report.append("BOTTOM 5 LEAST IMPORTANT PARAMETERS:")
        for i, param in enumerate(self.results['summary']['bottom_5'], 1):
            ranking = next(r for r in self.results['summary']['rankings'] if r[0] == param)
            score = ranking[1]['importance_score']
            report.append(f"  {i}. {param:30s} Score: {score:.4f}")

        report.append("")
        report.append("DETAILED ANALYSIS:")
        report.append("")

        for param, data in list(self.results['summary']['rankings'])[:5]:
            report.append(f"Parameter: {param}")

            if 'correlation' in self.results and param in self.results['correlation']:
                corr = self.results['correlation'][param]
                report.append(f"  - Pearson Correlation: {corr['pearson_corr']:.4f} "
                            f"(p={corr['pearson_p_value']:.4f})")

            if 'variance' in self.results and param in self.results['variance']:
                var = self.results['variance'][param]
                report.append(f"  - Fitness Range Impact: {var['fitness_range']:.4f}")

            if 'random_forest' in self.results and param in self.results['random_forest']:
                rf = self.results['random_forest'][param]
                report.append(f"  - RF Importance: {rf['permutation_importance']:.4f} "
                            f"(±{rf['permutation_std']:.4f})")

            report.append("")

        report.append("=" * 60)
        report.append("RECOMMENDATIONS:")
        report.append("=" * 60)
        report.append("")
        report.append("Focus tuning efforts on the top 5 parameters listed above.")
        report.append("Parameters in the bottom 5 have minimal impact and can use")
        report.append("default or conservative values to reduce optimization complexity.")
        report.append("")

        return "\n".join(report)
