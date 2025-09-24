"""
Visualization utilities for BEVFormer evaluation results
Provides plotting and visualization functions for metrics and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

try:
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Visualization functionality will be limited.")


class ResultVisualizer:
    """
    Visualization utility for BEVFormer evaluation results
    Creates plots for metrics, predictions, and analysis
    """

    def __init__(self,
                 class_names: List[str] = None,
                 save_dir: str = "./visualization"):
        """
        Initialize result visualizer

        Args:
            class_names: List of class names
            save_dir: Directory to save visualizations
        """
        if class_names is None:
            class_names = [
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
            ]
        self.class_names = class_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")

    def plot_metrics_summary(self,
                            metrics: Dict[str, float],
                            save_name: str = "metrics_summary.png",
                            title: str = "BEVFormer Evaluation Metrics") -> None:
        """
        Plot summary of main evaluation metrics

        Args:
            metrics: Dictionary of computed metrics
            save_name: Filename to save the plot
            title: Plot title
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        # Main metrics to display
        main_metrics = ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']

        # Extract values
        metric_names = []
        metric_values = []

        for metric in main_metrics:
            if metric in metrics:
                metric_names.append(metric)
                metric_values.append(metrics[metric])

        if not metric_values:
            print("No metrics to plot")
            return

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Main metrics bar plot
        colors = sns.color_palette("husl", len(metric_names))
        bars = ax1.bar(metric_names, metric_values, color=colors)
        ax1.set_title(f"{title}\nMain Metrics")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Per-class mAP
        class_aps = []
        class_names_with_ap = []

        for i, class_name in enumerate(self.class_names):
            key = f'mAP_{class_name}'
            if key in metrics:
                class_aps.append(metrics[key])
                class_names_with_ap.append(class_name)

        if class_aps:
            colors2 = sns.color_palette("Set2", len(class_aps))
            bars2 = ax2.bar(range(len(class_aps)), class_aps, color=colors2)
            ax2.set_title("Per-Class mAP")
            ax2.set_ylabel("mAP Score")
            ax2.set_xlabel("Class")
            ax2.set_xticks(range(len(class_names_with_ap)))
            ax2.set_xticklabels(class_names_with_ap, rotation=45, ha='right')
            ax2.set_ylim(0, 1.0)

            # Add value labels
            for bar, value in zip(bars2, class_aps):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Metrics summary plot saved to: {save_path}")

    def plot_training_curves(self,
                           train_metrics: Dict[str, List[float]],
                           val_metrics: Dict[str, List[float]],
                           epochs: List[int],
                           save_name: str = "training_curves.png") -> None:
        """
        Plot training and validation curves

        Args:
            train_metrics: Training metrics over epochs
            val_metrics: Validation metrics over epochs
            epochs: List of epoch numbers
            save_name: Filename to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        # Common metrics to plot
        metrics_to_plot = ['NDS', 'mAP', 'loss']

        # Find available metrics
        available_metrics = []
        for metric in metrics_to_plot:
            if metric in train_metrics or metric in val_metrics:
                available_metrics.append(metric)

        if not available_metrics:
            print("No training curves to plot")
            return

        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            ax = axes[i]

            # Plot training curve
            if metric in train_metrics:
                train_values = train_metrics[metric]
                ax.plot(epochs[:len(train_values)], train_values,
                       label=f'Train {metric}', marker='o', linewidth=2)

            # Plot validation curve
            if metric in val_metrics:
                val_values = val_metrics[metric]
                ax.plot(epochs[:len(val_values)], val_values,
                       label=f'Val {metric}', marker='s', linewidth=2)

            ax.set_title(f'{metric} Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves plot saved to: {save_path}")

    def plot_error_analysis(self,
                          metrics: Dict[str, float],
                          save_name: str = "error_analysis.png") -> None:
        """
        Plot detailed error analysis

        Args:
            metrics: Dictionary of computed metrics
            save_name: Filename to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        # Error metrics
        error_metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
        error_names = ['Translation', 'Scale', 'Orientation', 'Velocity', 'Attribute']

        # Extract error values
        error_values = []
        for metric in error_metrics:
            if metric in metrics:
                error_values.append(metrics[metric])
            else:
                error_values.append(0.0)

        # Create radar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Error bar chart
        colors = sns.color_palette("Reds_r", len(error_names))
        bars = ax1.bar(error_names, error_values, color=colors)
        ax1.set_title("Error Analysis")
        ax1.set_ylabel("Error Value")
        ax1.set_xlabel("Error Type")

        # Add value labels
        for bar, value in zip(bars, error_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Normalized errors (for NDS contribution)
        normalized_errors = []
        for i, (metric, value) in enumerate(zip(error_metrics, error_values)):
            if metric == 'mAOE':
                # Orientation error is normalized by π
                normalized = min(1.0, value / np.pi)
            else:
                # Other errors are clamped at 1.0
                normalized = min(1.0, value)
            normalized_errors.append(1.0 - normalized)  # Convert to contribution (higher is better)

        # NDS contribution pie chart
        if any(normalized_errors):
            # Add mAP contribution (equally weighted with average of error terms)
            map_contribution = metrics.get('mAP', 0.0)
            avg_error_contribution = np.mean(normalized_errors)

            contributions = [map_contribution, avg_error_contribution]
            labels = ['mAP', 'Error Terms']
            colors = ['lightblue', 'lightcoral']

            ax2.pie(contributions, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax2.set_title("NDS Score Composition")

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Error analysis plot saved to: {save_path}")

    def plot_class_performance_matrix(self,
                                    metrics: Dict[str, float],
                                    save_name: str = "class_performance.png") -> None:
        """
        Plot class performance matrix

        Args:
            metrics: Dictionary of computed metrics
            save_name: Filename to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        # Extract per-class mAP scores
        class_aps = []
        for class_name in self.class_names:
            key = f'mAP_{class_name}'
            if key in metrics:
                class_aps.append(metrics[key])
            else:
                class_aps.append(0.0)

        # Create performance matrix (1D data reshaped for heatmap)
        n_classes = len(class_aps)
        n_cols = min(5, n_classes)  # Max 5 columns
        n_rows = (n_classes + n_cols - 1) // n_cols

        # Pad with zeros if needed
        padded_aps = class_aps + [0.0] * (n_rows * n_cols - n_classes)
        performance_matrix = np.array(padded_aps).reshape(n_rows, n_cols)

        # Create labels matrix
        padded_names = self.class_names + [''] * (n_rows * n_cols - n_classes)
        labels_matrix = np.array(padded_names).reshape(n_rows, n_cols)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(n_cols * 2, n_rows * 1.5))

        # Create heatmap with custom colormap
        sns.heatmap(performance_matrix,
                   annot=labels_matrix,
                   fmt='',
                   cmap='RdYlBu_r',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'mAP Score'},
                   ax=ax)

        ax.set_title('Per-Class mAP Performance')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Class performance matrix saved to: {save_path}")

    def create_evaluation_report(self,
                               metrics: Dict[str, float],
                               epoch: Optional[int] = None,
                               save_name: str = "evaluation_report") -> None:
        """
        Create comprehensive evaluation report with multiple visualizations

        Args:
            metrics: Dictionary of computed metrics
            epoch: Current epoch (if applicable)
            save_name: Base name for saved files
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for creating evaluation report")
            return

        epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""

        print("Creating comprehensive evaluation report...")

        # Create all visualizations
        self.plot_metrics_summary(
            metrics,
            save_name=f"{save_name}_summary{epoch_suffix}.png",
            title=f"BEVFormer Evaluation{f' - Epoch {epoch}' if epoch else ''}"
        )

        self.plot_error_analysis(
            metrics,
            save_name=f"{save_name}_errors{epoch_suffix}.png"
        )

        self.plot_class_performance_matrix(
            metrics,
            save_name=f"{save_name}_classes{epoch_suffix}.png"
        )

        # Save metrics as JSON for reference
        metrics_path = self.save_dir / f"{save_name}_metrics{epoch_suffix}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"Evaluation report created in: {self.save_dir}")

    def compare_models(self,
                      model_metrics: Dict[str, Dict[str, float]],
                      save_name: str = "model_comparison.png") -> None:
        """
        Compare metrics across different models

        Args:
            model_metrics: Dictionary mapping model names to their metrics
            save_name: Filename to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        model_names = list(model_metrics.keys())
        if len(model_names) < 2:
            print("Need at least 2 models for comparison")
            return

        # Main metrics to compare
        metrics_to_compare = ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE']

        # Create comparison plot
        fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(4 * len(metrics_to_compare), 6))

        if len(metrics_to_compare) == 1:
            axes = [axes]

        x_pos = np.arange(len(model_names))
        bar_width = 0.8 / len(model_names)

        for i, metric in enumerate(metrics_to_compare):
            ax = axes[i]
            values = [model_metrics[name].get(metric, 0.0) for name in model_names]

            bars = ax.bar(x_pos, values, bar_width, label=metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Model comparison plot saved to: {save_path}")


# Utility functions
def load_evaluation_results(result_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file

    Args:
        result_path: Path to results JSON file

    Returns:
        Dictionary with loaded results
    """
    with open(result_path, 'r') as f:
        return json.load(f)


def extract_metrics_history(log_dir: str) -> Tuple[Dict, Dict, List[int]]:
    """
    Extract training metrics history from log files (placeholder)

    Args:
        log_dir: Directory containing log files

    Returns:
        Tuple of (train_metrics, val_metrics, epochs)
    """
    # This would parse actual log files or TensorBoard data
    # For now, return empty placeholders
    return {}, {}, []


# Example usage
if __name__ == "__main__":
    # Test visualization functionality
    print("Testing BEVFormer result visualization...")

    # Create dummy metrics for testing
    test_metrics = {
        'NDS': 0.517,
        'mAP': 0.416,
        'mATE': 0.68,
        'mASE': 0.27,
        'mAOE': 0.55,
        'mAVE': 0.87,
        'mAAE': 0.23,
        'mAP_car': 0.85,
        'mAP_truck': 0.45,
        'mAP_construction_vehicle': 0.12,
        'mAP_bus': 0.38,
        'mAP_trailer': 0.15,
        'mAP_barrier': 0.58,
        'mAP_motorcycle': 0.42,
        'mAP_bicycle': 0.28,
        'mAP_pedestrian': 0.78,
        'mAP_traffic_cone': 0.55,
    }

    # Create visualizer
    visualizer = ResultVisualizer(save_dir="./test_visualization")

    # Create evaluation report
    if MATPLOTLIB_AVAILABLE:
        visualizer.create_evaluation_report(test_metrics, epoch=24)
        print("Test visualization completed!")
    else:
        print("Matplotlib not available - skipping visualization test")