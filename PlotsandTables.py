import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Directory to save all outputs
output_dir = "/big/users/kat-mcmillan/benchmarks/nova-3-medical/plotsTables"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary mapping file paths to system labels
results_dict = {
    "/big/quarantine/gmc_automation/en/test_results/dg.349c83dc-989b-4a6c-9c67-a43abd0587d0.en.20250225.medical.batch.asr/results.json": "Nova-2-medical",
    "/big/quarantine/gmc_automation/en/test_results/dg.3b3aabe4-608a-46ac-9585-7960a25daf1a.en.20250225.medical.batch.asr/results.json": "Nova-3-general",
    "/big/quarantine/gmc_automation/en/test_results/dg.91d566f5-ff0d-4584-9b9b-2414a929e62e.en.20250225.medical.batch.asr/results.json": "Nova-3-medical",
    "/big/quarantine/gmc_automation/en/test_results/aws.en-US.en.20250225.medical.CONVERSATION.batch.asr/results.json": "AWS",
    "/big/quarantine/gmc_automation/en/test_results/azure.en-US.en.20250225.medical.batch.asr/results.json": "Azure",
    "/big/quarantine/gmc_automation/en/test_results/revai.en-US.en.20250225.medical.batch.asr/results.json": "RevAI",
    "/big/quarantine/gmc_automation/en/test_results/assembly.en-US.en.20250225.medical.batch.asr/results.json": "Assembly",
    "/big/quarantine/gmc_automation/en/test_results/whisper_api.en-US.en.20250225.medical.large-v2.batch.asr/results.json": "Whisper-large",
    "/big/quarantine/gmc_automation/en/test_results/speechmatics.en-US.en.20250225.medical.batch.asr/results.json": "Speechmatics"
}

# Initialize an empty DataFrame for combined results
combined_results = pd.DataFrame()

# Process each file in the dictionary
for file_path, system_name in results_dict.items():
    # Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue
    
    # Rename word count column to a standard name
    if 'truth_count' in df.columns:
        df.rename(columns={'truth_count': 'total_words'}, inplace=True)
    elif 'tokens' in df.columns:
        df.rename(columns={'tokens': 'total_words'}, inplace=True)
    
    # Check if necessary columns exist
    if 'wer' not in df.columns or 'wrr' not in df.columns:
        print(f"Skipping file due to missing columns: {file_path}")
        continue
    
    # Add a column for the system name
    df['GMC'] = system_name
    
    # Combine into the main DataFrame
    combined_results = pd.concat([combined_results, df], ignore_index=True)

# Calculate metrics for each provider
provider_metrics = combined_results.groupby('GMC', as_index=False).agg(
    median_WER=('wer', 'median'),
    mean_WER=('wer', 'mean'),
    median_WRR=('wrr', 'median'),
    mean_WRR=('wrr', 'mean'),
)

# Calculate overall WER and WRR (weighted by total_words)
combined_results['weighted_wer'] = combined_results['wer'] * combined_results['total_words']
combined_results['weighted_wrr'] = combined_results['wrr'] * combined_results['total_words']

overall_metrics = combined_results.groupby('GMC', as_index=False).agg(
    total_words=('total_words', 'sum'),
    weighted_wer_sum=('weighted_wer', 'sum'),
    weighted_wrr_sum=('weighted_wrr', 'sum')
)

overall_metrics['overall_WER'] = overall_metrics['weighted_wer_sum'] / overall_metrics['total_words']
overall_metrics['overall_WRR'] = overall_metrics['weighted_wrr_sum'] / overall_metrics['total_words']

# Merge with provider metrics
provider_metrics = pd.merge(provider_metrics, overall_metrics[['GMC', 'overall_WER', 'overall_WRR']], on='GMC', how='left')

# Save the results to CSV
provider_metrics.to_csv(os.path.join(output_dir, "provider_metrics.csv"), index=False)

# Display metrics in a readable format
print(provider_metrics)

# Bar plots for WER and WRR
for metric, ylabel, title in [
    ('median_WER', 'Median WER', 'Median WER'),
    ('mean_WER', 'Mean WER', 'Mean WER'),
    ('median_WRR', 'Median WRR', 'Median WRR'),
    ('mean_WRR', 'Mean WRR', 'Mean WRR'),
    ('overall_WER', 'Overall WER', 'Overall WER')
]:
    sns.barplot(data=provider_metrics, x='GMC', y=metric, palette="viridis")
    plt.title(title)
    plt.xlabel('Provider')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"))
    plt.close()

# Box plots for WER and WRR
for metric, ylabel, title, filename in [
    ('wer', 'WER', 'WER Distribution', 'wer_distribution_boxplot.png'),
    ('wrr', 'WRR', 'WRR Distribution', 'wrr_distribution_boxplot.png'),
]:
    sns.boxplot(data=combined_results, x='GMC', y=metric, showfliers=False, palette="viridis")
    plt.title(title)
    plt.xlabel('Provider')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"All outputs (provider metrics, bar plots, and box plots) have been saved to the directory: {output_dir}")
