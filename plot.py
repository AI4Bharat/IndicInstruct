import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

# Base directory containing task folders
assert len(argv) > 1, "Please provide the results folder as the first argument"
base_dir = argv[1]

# Iterate through each task folder
for task_name in os.listdir(base_dir):
    task_path = os.path.join(base_dir, task_name)
    
    if os.path.isdir(task_path):
        # Dictionary to store all language metrics for each k_shot
        k_shot_language_metrics = {}
        
        # Iterate through each language folder within the task folder
        for language_name in os.listdir(task_path):
            language_path = os.path.join(task_path, language_name)
            
            if os.path.isdir(language_path):
                # Iterate through each k_shot folder within the language folder
                for k_shot_name in os.listdir(language_path):
                    k_shot_path = os.path.join(language_path, k_shot_name)
                    metric_file = os.path.join(k_shot_path, 'metrics.json')
                    
                    # Read the metrics.json file if it exists
                    if os.path.isfile(metric_file):
                        with open(metric_file, 'r') as file:
                            metrics = json.load(file)
                            for k,v in metrics.items():
                                if metrics[k] > 1:
                                    metrics[k] = metrics[k] / 100
                        
                        # Initialize nested dictionary structure for k_shot and language
                        if k_shot_name not in k_shot_language_metrics:
                            k_shot_language_metrics[k_shot_name] = {}
                        k_shot_language_metrics[k_shot_name][language_name] = metrics

        # Generate plots for each k_shot, including all languages in the same plot
        for k_shot_name, language_metrics in k_shot_language_metrics.items():
            languages = list(language_metrics.keys())
            metric_names = sorted(set(metric for metrics in language_metrics.values() for metric in metrics))
            
            # Initialize figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.2  # Width of each bar
            num_metrics = len(metric_names)
            
            # Generate bars for each metric
            for idx, metric in enumerate(metric_names):
                values = [language_metrics.get(lang, {}).get(metric, 0) for lang in languages]
                bar_positions = np.arange(len(languages)) + idx * bar_width
                
                # Plot each metric bar and assign label for the legend
                ax.bar(bar_positions, values, bar_width, label=metric)
            
            ax.set_xlabel('Language')
            ax.set_ylabel('Score')
            ax.set_title(f'{task_name} - {k_shot_name} Task Metrics by Language')
            ax.set_ylim(0, 1.5)
            ax.set_xticks(np.arange(len(languages)) + (num_metrics - 1) * bar_width / 2)
            ax.set_xticklabels(languages, rotation=45)
            
            # Add legend only if there are labels
            if metric_names:
                ax.legend(title="Metrics")
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Save the figure as PNG with task and k_shot name
            plt.tight_layout()
            plt.savefig(os.path.join(task_path, f"{task_name}_{k_shot_name}_metrics.png"))
            plt.close(fig)

print(f"Plots generated for each k_shot of each task with all languages in each subfolder of {base_dir}")
