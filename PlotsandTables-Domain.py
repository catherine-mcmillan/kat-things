import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional

def parse_args():
    parser = argparse.ArgumentParser(description='Process ASR results across different systems')
    parser.add_argument('--output-dir', type=str, default="/big/users/kat-mcmillan/whisper-4o",
                      help='Directory to save all outputs')
    parser.add_argument('--domain-mapping', type=str, default="/big/users/kat-mcmillan/dataset_domains_new.json",
                      help='Path to domain mapping JSON file')
    return parser.parse_args()

# Column mapping for standardization
COLUMN_MAPPING = {
    'tokens': 'total_words',
    'source_id': 'resource_id',
    'truth_count': 'tokens',
    'match_count': 'matches',
    'error_count': 'errors',
    'word_ins': 'I',
    'word_del': 'D',
    'word_repl': 'R'
}

# Required columns for metrics calculation
REQUIRED_COLUMNS = ['wer', 'wrr', 'total_words']

def standardize_columns(df: pd.DataFrame, system_name: str) -> pd.DataFrame:
    """Standardize column names across different CSV formats."""
    # Rename columns based on mapping
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Add missing required columns with NaN if they don't exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Missing required column '{col}' in {system_name}")
            df[col] = pd.NA
            
    return df

def process_file(file_path: str, system_name: str, domain_mapping: Dict) -> Optional[pd.DataFrame]:
    """Process a single results file with error handling."""
    try:
        df = pd.read_csv(file_path)
        print(f"\nProcessing {system_name} from {file_path}")
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Standardize columns
        df = standardize_columns(df, system_name)

        # Add domain using dataset_path if dataset column doesn't exist
        if 'dataset' in df.columns:
            df['domain'] = df['dataset'].map(domain_mapping)
        elif 'dataset_path' in df.columns:
            df['domain'] = df['dataset_path'].apply(lambda x: x.split('/')[0])
        else:
            print(f"Error: Missing both 'dataset' and 'dataset_path' columns in {system_name}")
            return None

        # Add system name column
        df['GMC'] = system_name
        
        return df

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def create_plot(data: pd.DataFrame, x: str, y: str, hue: str, plot_type: str,
               title: str, ylabel: str, filename: str, output_dir: str,
               show_fliers: bool = False) -> None:
    """Create and save a plot with proper error handling."""
    try:
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            sns.barplot(data=data, x=x, y=y, hue=hue,
                       palette=get_palette(data[hue].unique()))
        elif plot_type == 'box':
            sns.boxplot(data=data, x=x, y=y, hue=hue,
                       showfliers=show_fliers,
                       palette=get_palette(data[hue].unique()))
        
        plt.title(title)
        plt.xlabel('Domain')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)

        # Manually create legend based on custom colors
        handles = [
            mlines.Line2D([], [], color=provider_colors.get(provider, "#7f7f7f"),
                         marker='o', linestyle='', markersize=10, label=provider)
            for provider in sorted(data[hue].unique())
        ]
        plt.legend(title="Providers", handles=handles, bbox_to_anchor=(1.05, 1),
                  loc='upper left')

        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        print(f"Created plot: {filename}")
        
    except Exception as e:
        print(f"Error creating {plot_type} plot for {filename}: {str(e)}")

def create_plots(domain_provider_metrics: pd.DataFrame, combined_results: pd.DataFrame,
                output_dir: str) -> None:
    """Create all plots with progress tracking."""
    # Bar plots for aggregated metrics
    bar_plots = [
        ('median_WER', 'Median WER', 'Median WER by Domain', 'median_WER_domain_comparison.png'),
        ('mean_WER', 'Mean WER', 'Mean WER by Domain', 'mean_WER_domain_comparison.png'),
        ('median_WRR', 'Median WRR', 'Median WRR by Domain', 'median_WRR_domain_comparison.png'),
        ('mean_WRR', 'Mean WRR', 'Mean WRR by Domain', 'mean_WRR_domain_comparison.png'),
        ('overall_WER', 'Overall WER', 'Overall WER by Domain', 'overall_WER_domain_comparison.png')
    ]
    
    print("\nCreating bar plots...")
    for metric, ylabel, title, filename in tqdm(bar_plots, desc="Creating bar plots"):
        if metric in domain_provider_metrics.columns:
            create_plot(
                data=domain_provider_metrics,
                x='domain',
                y=metric,
                hue='GMC',
                plot_type='bar',
                title=title,
                ylabel=ylabel,
                filename=filename,
                output_dir=output_dir
            )
        else:
            print(f"Warning: Metric '{metric}' not found in data")

    # Box plots for distributions
    box_plots = [
        ('wer', 'WER', 'WER Distribution by Domain', 'wer_distribution_by_domain_boxplot.png'),
        ('wrr', 'WRR', 'WRR Distribution by Domain', 'wrr_distribution_by_domain_boxplot.png')
    ]
    
    print("\nCreating box plots...")
    for metric, ylabel, title, filename in tqdm(box_plots, desc="Creating box plots"):
        if metric in combined_results.columns:
            plot_data = combined_results.dropna(subset=['domain', 'GMC', metric])
            create_plot(
                data=plot_data,
                x='domain',
                y=metric,
                hue='GMC',
                plot_type='box',
                title=title,
                ylabel=ylabel,
                filename=filename,
                output_dir=output_dir,
                show_fliers=False
            )
        else:
            print(f"Warning: Metric '{metric}' not found in data")

def calculate_weighted_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted metrics with proper error handling."""
    try:
        # Convert columns to numeric, coercing errors to NaN
        for col in ['wer', 'wrr', 'total_words']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate weighted metrics only for valid rows
        df['weighted_wer'] = df['wer'].multiply(df['total_words'], fill_value=0)
        df['weighted_wrr'] = df['wrr'].multiply(df['total_words'], fill_value=0)
        
        return df
    except Exception as e:
        print(f"Error calculating weighted metrics: {str(e)}")
        return df

def calculate_overall_metrics(group_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall metrics for a group with proper error handling."""
    try:
        metrics = group_df.agg(
            total_words=('total_words', 'sum'),
            weighted_wer_sum=('weighted_wer', 'sum'),
            weighted_wrr_sum=('weighted_wrr', 'sum')
        )
        
        # Calculate overall metrics only if we have valid totals
        if metrics['total_words'] > 0:
            metrics['overall_WER'] = metrics['weighted_wer_sum'] / metrics['total_words']
            metrics['overall_WRR'] = metrics['weighted_wrr_sum'] / metrics['total_words']
        else:
            metrics['overall_WER'] = pd.NA
            metrics['overall_WRR'] = pd.NA
            
        return metrics
    except Exception as e:
        print(f"Error calculating overall metrics: {str(e)}")
        # Return DataFrame with NaN values if calculation fails
        return pd.DataFrame({
            'total_words': [0],
            'weighted_wer_sum': [0],
            'weighted_wrr_sum': [0],
            'overall_WER': [pd.NA],
            'overall_WRR': [pd.NA]
        })

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load domain mapping
    try:
        with open(args.domain_mapping, "r") as f:
            domain_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: Domain mapping file not found: {args.domain_mapping}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in domain mapping file: {args.domain_mapping}")
        return

    # Dictionary mapping file paths to system labels
    results_dict = {
        "/big/users/kat-mcmillan/whisper-4o/gpt-mini-file_results.csv": "4o-mini",
        "/big/users/kat-mcmillan/whisper-4o/gpt4-file_results.csv": "4o",
        "/big/users/kat-mcmillan/whisper-4o/whisper-1-file_results.csv": "whisper",
        "/home/users/kat-mcmillan/projects/benchmarks/gmc/nova-3-results/results.csv": "nova-3",
        "/big/quarantine/gmc_automation/en/test_results/dg.nova-2.en-US.20250106.batch.asr/results.csv": "nova-2",
    }

    # Initialize an empty DataFrame for combined results
    combined_results = pd.DataFrame()

    # Process each file in the dictionary with progress bar
    for file_path, system_name in tqdm(results_dict.items(), desc="Processing files"):
        df = process_file(file_path, system_name, domain_mapping)
        if df is not None:
            combined_results = pd.concat([combined_results, df], ignore_index=True)
            print(f"Combined results shape: {combined_results.shape}")

    # Check if we have any data before proceeding
    if combined_results.empty:
        print("Error: No data was loaded. Please check the file paths and required columns.")
        return

    # Calculate metrics only if we have the required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in combined_results.columns]
    if missing_cols:
        print(f"Error: Missing required columns for metrics calculation: {missing_cols}")
        return

    print("\nCalculating metrics...")
    # Calculate weighted metrics
    combined_results = calculate_weighted_metrics(combined_results)

    # Calculate metrics grouped by domain and provider
    print("Calculating domain-provider metrics...")
    domain_provider_metrics = combined_results.groupby(['domain', 'GMC'], as_index=False).agg(
        median_WER=('wer', lambda x: x.median()),
        mean_WER=('wer', lambda x: x.mean()),
        median_WRR=('wrr', lambda x: x.median()),
        mean_WRR=('wrr', lambda x: x.mean()),
    )

    # Calculate overall metrics
    print("Calculating overall metrics...")
    overall_metrics = pd.DataFrame()
    for (domain, gmc), group in combined_results.groupby(['domain', 'GMC']):
        metrics = calculate_overall_metrics(group)
        metrics_df = pd.DataFrame({
            'domain': [domain],
            'GMC': [gmc],
            'total_words': [metrics['total_words']],
            'overall_WER': [metrics['overall_WER']],
            'overall_WRR': [metrics['overall_WRR']]
        })
        overall_metrics = pd.concat([overall_metrics, metrics_df], ignore_index=True)

    # Merge metrics
    print("Merging metrics...")
    domain_provider_metrics = pd.merge(
        domain_provider_metrics,
        overall_metrics[['domain', 'GMC', 'overall_WER', 'overall_WRR']],
        on=['domain', 'GMC'],
        how='left'
    )

    # Save metrics to CSV
    metrics_path = os.path.join(args.output_dir, "domain_provider_metrics.csv")
    domain_provider_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved domain-based metrics to: {metrics_path}")

    # Custom color palette
    provider_colors = {
        "Nova-2": "#149afb",  # Azure
        "Nova-3": "#13ef93",  # Spring Green
        "Assembly": "#ee028c",
        "AWS": "#ae63f9",
        "Google [tele]": "#bbbbbf",
        "Google [long]": "#949498",
        "Google [short]": "#4e4e52",
        "RevAI": "#f04438",
        "Whisper": "#fec84b",
        "Speechmatics": "#12b76a",
        "Azure": "#ae63f9"
    }
    default_palette = sns.color_palette("viridis", n_colors=6)

    # Function to get colors dynamically
    def get_palette(hue_values):
        return [provider_colors.get(val, "#7f7f7f") for val in hue_values]

    # Create plots
    create_plots(domain_provider_metrics, combined_results, args.output_dir)
    
    print(f"\nDomain-based metrics and plots have been saved to: {args.output_dir}")

if __name__ == "__main__":
    main()