import argparse
import re
import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import Levenshtein
import json
import anthropic
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def normalize_text(text, ignore_case=True, ignore_numeric_equivalents=True):
    """
    Normalize text according to specified criteria:
    - Convert to lowercase if ignore_case is True
    - Standardize numeric representations if ignore_numeric_equivalents is True
    """
    # Create a copy of the original text
    normalized = text
    
    # Convert to lowercase if specified
    if ignore_case:
        normalized = normalized.lower()
    
    if ignore_numeric_equivalents:
        # Convert spelled-out numbers to their digit form
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
            'million': '1000000', 'billion': '1000000000'
        }
        
        # Handle compound numbers (like "twenty-one" or "twenty one")
        for tens in ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']:
            tens_val = number_words[tens]
            for units in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
                units_val = number_words[units]
                compound = f"{tens}-{units}"
                compound_val = str(int(tens_val) + int(units_val))
                number_words[compound] = compound_val
                # Also handle without hyphen
                compound_space = f"{tens} {units}"
                number_words[compound_space] = compound_val
        
        # Find and replace spelled-out numbers
        for word, digit in sorted(number_words.items(), key=lambda x: len(x[0]), reverse=True):
            normalized = re.sub(r'\b' + word + r'\b', digit, normalized, flags=re.IGNORECASE)
        
        # Normalize URL formats
        normalized = re.sub(r'(\s*dot\s*)', '.', normalized)
        normalized = re.sub(r'w\s+w\s+w', 'www', normalized)
        normalized = re.sub(r'(\s*www\s+)', 'www.', normalized)
        
        # Remove leading zeros from numbers (except single zero)
        normalized = re.sub(r'\b0+([1-9][0-9]*)\b', r'\1', normalized)
        
        # Standardize common format variations
        normalized = re.sub(r'\bpercent\b', '%', normalized)
        normalized = re.sub(r'\sand\b', '', normalized)  # Remove "and" in number expressions
    
    return normalized

def calc_char_wer_wrr(reference, hypothesis):
    """
    Calculate character-level WER and WRR between reference and hypothesis,
    ignoring case and numeric representation differences.
    """
    # Normalize both texts before comparison
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    
    # Calculate Levenshtein distance at character level
    distance = Levenshtein.distance(norm_ref, norm_hyp)
    
    # Avoid division by zero
    if len(norm_ref) == 0:
        if len(norm_hyp) == 0:
            return 0.0, 1.0  # Perfect match
        else:
            return 1.0, 0.0  # All insertions
    
    char_wer = distance / len(norm_ref)
    char_wrr = 1 - char_wer
    
    return char_wer, char_wrr

def identify_error_types(reference, hypothesis):
    """
    Identify the types of errors between reference and hypothesis.
    """
    # Normalize both texts
    norm_ref = normalize_text(reference)
    norm_hyp = normalize_text(hypothesis)
    
    # If they're equivalent after normalization, no errors to analyze
    if norm_ref == norm_hyp:
        return {}
    
    # Find differences
    errors = {}
    
    # Check for alphanumeric errors
    alpha_errors_ref = re.findall(r'[A-Za-z0-9]+', reference)
    alpha_errors_hyp = re.findall(r'[A-Za-z0-9]+', hypothesis)
    
    # Look for mismatches in alphanumeric tokens
    alpha_diff = set(alpha_errors_ref).symmetric_difference(set(alpha_errors_hyp))
    if alpha_diff:
        errors['alphanumeric_errors'] = list(alpha_diff)
    
    # Check for entity recognition issues (proper nouns, organizations, etc.)
    entities_ref = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b', reference)
    entities_hyp = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b', hypothesis)
    entity_diff = set(entities_ref).symmetric_difference(set(entities_hyp))
    if entity_diff:
        errors['entity_errors'] = list(entity_diff)
    
    # Check for numeric/word form differences
    num_pattern = r'\b\d+(\.\d+)?\b|%'
    word_num_pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b'
    
    ref_nums = re.findall(num_pattern, reference, re.IGNORECASE)
    ref_word_nums = re.findall(word_num_pattern, reference, re.IGNORECASE)
    hyp_nums = re.findall(num_pattern, hypothesis, re.IGNORECASE)
    hyp_word_nums = re.findall(word_num_pattern, hypothesis, re.IGNORECASE)
    
    if (ref_nums and hyp_word_nums) or (ref_word_nums and hyp_nums):
        errors['numeric_word_format'] = True
    
    # Check for URL or web address formatting
    url_pattern = r'\b(www\.|http://|https://|\.com|\.org|\.net|\.edu|\.gov)\b|(\s+dot\s+)'
    ref_urls = re.findall(url_pattern, reference)
    hyp_urls = re.findall(url_pattern, hypothesis)
    
    if (ref_urls and not hyp_urls) or (hyp_urls and not ref_urls):
        errors['url_formatting'] = True
    
    # Check for spacing issues
    if re.sub(r'\s+', ' ', reference).strip() != re.sub(r'\s+', ' ', hypothesis).strip():
        errors['spacing_errors'] = True
    
    # Check for capitalization differences
    if reference.lower() == hypothesis.lower() and reference != hypothesis:
        errors['capitalization'] = True
    
    # Classify if this seems like an entity recognition or formatting issue
    if 'entity_errors' in errors or 'numeric_word_format' in errors:
        errors['entity_recognition_issue'] = True
    
    if 'url_formatting' in errors or 'spacing_errors' in errors or 'capitalization' in errors:
        errors['formatting_issue'] = True
    
    return errors

def parse_resource(lines, start_idx):
    """
    Parse a single resource from the lines of text starting at start_idx.
    Returns the resource data and the index where the next resource starts.
    """
    resource = {
        'triplets': [],
        'metadata': {}
    }
    
    i = start_idx
    
    # Parse resource metadata
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('resource_id:'):
            resource['metadata']['resource_id'] = line.split('resource_id:')[1].strip()
        elif line.startswith('dataset:'):
            resource['metadata']['dataset'] = line.split('dataset:')[1].strip()
        elif line.startswith('name:'):
            resource['metadata']['name'] = line.split('name:')[1].strip()
        elif line.startswith('WER:'):
            # Extract original WER/WRR
            parts = line.split('/')
            wer_part = parts[0].strip()
            wrr_part = parts[1].strip() if len(parts) > 1 else ""
            
            try:
                resource['metadata']['original_wer'] = float(wer_part.split(':')[1].strip())
                resource['metadata']['original_wrr'] = float(wrr_part.split(':')[1].strip()) if wrr_part else 0.0
            except:
                resource['metadata']['original_wer'] = 0.0
                resource['metadata']['original_wrr'] = 0.0
            
            i += 1  # Move past WER/WRR line
            break
        
        i += 1
    
    # Parse triplets
    human_text = ""
    machine_text = ""
    comparison_text = ""
    state = 0  # 0: expecting human, 1: expecting machine, 2: expecting comparison
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if we've reached the next resource
        if line.startswith('resource_id:'):
            # Save the last triplet if we have one
            if human_text and machine_text:
                resource['triplets'].append({
                    'human': human_text,
                    'machine': machine_text,
                    'comparison': comparison_text
                })
            break
        
        # Skip empty lines
        if not line:
            i += 1
            
            # If we've collected the comparison, save the triplet and reset
            if state == 2 or state == 1:  # We've collected at least human and machine
                if human_text and machine_text:
                    resource['triplets'].append({
                        'human': human_text,
                        'machine': machine_text,
                        'comparison': comparison_text
                    })
                    human_text = ""
                    machine_text = ""
                    comparison_text = ""
                    state = 0
            continue
        
        # Process based on current state
        if state == 0:
            # Human text
            human_text = line
            state = 1
        elif state == 1:
            # Machine text
            machine_text = line
            state = 2
        elif state == 2:
            # Comparison text
            comparison_text = line
            
            # Save the triplet and reset for the next one
            resource['triplets'].append({
                'human': human_text,
                'machine': machine_text,
                'comparison': comparison_text
            })
            human_text = ""
            machine_text = ""
            comparison_text = ""
            state = 0
        
        i += 1
    
    # If we have a partial triplet at the end, add it
    if human_text and machine_text and state >= 1:
        resource['triplets'].append({
            'human': human_text,
            'machine': machine_text,
            'comparison': comparison_text
        })
    
    return resource, i

def process_batch(lines, start_idx, batch_size):
    """
    Process a batch of resources from the file.
    Returns a list of processed resources and the index to start the next batch.
    """
    resources = []
    i = start_idx
    
    # Process resources until we hit the end of the batch or file
    resource_count = 0
    while i < len(lines) and resource_count < batch_size:
        # Check if we're at the start of a resource
        if i < len(lines) and lines[i].strip().startswith('resource_id:'):
            resource, new_idx = parse_resource(lines, i)
            resources.append(resource)
            i = new_idx
            resource_count += 1
        else:
            i += 1
            if i >= len(lines):
                break
    
    return resources, i

def analyze_resource(resource):
    """
    Analyze a single resource, computing character-level metrics and error types.
    """
    # Initialize metrics
    total_char_wer = 0.0
    total_char_wrr = 0.0
    error_counts = Counter()
    
    # Process each triplet
    analyzed_triplets = []
    
    for triplet in resource['triplets']:
        human = triplet['human']
        machine = triplet['machine']
        comparison = triplet['comparison']
        
        # Calculate character-level WER/WRR
        char_wer, char_wrr = calc_char_wer_wrr(human, machine)
        
        # Identify error types
        error_types = identify_error_types(human, machine)
        
        analyzed_triplets.append({
            'human': human,
            'machine': machine,
            'comparison': comparison,
            'char_wer': char_wer,
            'char_wrr': char_wrr,
            'error_types': error_types
        })
        
        # Update metrics
        total_char_wer += char_wer
        total_char_wrr += char_wrr
        
        # Count error types
        for error_type in error_types:
            error_counts[error_type] += 1
    
    # Calculate averages
    num_triplets = len(resource['triplets'])
    avg_char_wer = total_char_wer / num_triplets if num_triplets > 0 else 0.0
    avg_char_wrr = total_char_wrr / num_triplets if num_triplets > 0 else 1.0
    
    # Add metrics to resource
    resource['char_wer'] = avg_char_wer
    resource['char_wrr'] = avg_char_wrr
    resource['error_counts'] = dict(error_counts)
    resource['analyzed_triplets'] = analyzed_triplets
    
    return resource

def analyze_batch(resources):
    """
    Analyze a batch of resources in parallel.
    """
    analyzed_resources = []
    
    # Process resources in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_resource, resource) for resource in resources]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing resources"):
            analyzed_resources.append(future.result())
    
    return analyzed_resources

def compute_overall_metrics(all_resources):
    """
    Compute overall dataset metrics from all analyzed resources.
    """
    # Initialize counters
    total_char_wer = 0.0
    total_char_wrr = 0.0
    total_resources = len(all_resources)
    all_error_counts = Counter()
    error_by_resource = defaultdict(list)
    error_examples = defaultdict(list)
    
    # Gather metrics and error counts
    for resource in all_resources:
        total_char_wer += resource.get('char_wer', 0)
        total_char_wrr += resource.get('char_wrr', 0)
        
        # Gather error counts
        for error_type, count in resource.get('error_counts', {}).items():
            all_error_counts[error_type] += count
            error_by_resource[error_type].append(resource['metadata']['resource_id'])
        
        # Collect examples for the most common errors
        for triplet in resource.get('analyzed_triplets', []):
            for error_type in triplet.get('error_types', {}):
                # Limit to 5 examples per error type
                if len(error_examples[error_type]) < 5:
                    error_examples[error_type].append({
                        'human': triplet['human'],
                        'machine': triplet['machine'],
                        'resource_id': resource['metadata']['resource_id']
                    })
    
    # Calculate averages
    avg_char_wer = total_char_wer / total_resources if total_resources > 0 else 0.0
    avg_char_wrr = total_char_wrr / total_resources if total_resources > 0 else 1.0
    
    # Calculate error frequencies
    total_errors = sum(all_error_counts.values())
    error_percentages = {}
    for error_type, count in all_error_counts.items():
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        error_percentages[error_type] = percentage
    
    # Calculate error prevalence across resources
    error_prevalence = {}
    for error_type, resources_with_error in error_by_resource.items():
        prevalence = len(set(resources_with_error)) / total_resources * 100
        error_prevalence[error_type] = prevalence
    
    return {
        'avg_char_wer': avg_char_wer,
        'avg_char_wrr': avg_char_wrr,
        'error_counts': dict(all_error_counts),
        'error_percentages': error_percentages,
        'error_prevalence': error_prevalence,
        'error_examples': {k: v for k, v in error_examples.items()},
        'total_resources': total_resources,
        'total_errors': total_errors
    }

def generate_analysis_summary(metrics):
    """
    Generate a summary of the metrics for Claude to analyze
    """
    summary = {
        'metrics': {
            'avg_char_wer': metrics['avg_char_wer'],
            'avg_char_wrr': metrics['avg_char_wrr'],
            'total_resources': metrics['total_resources'],
            'total_errors': metrics['total_errors']
        },
        'error_percentages': metrics['error_percentages'],
        'error_prevalence': metrics['error_prevalence'],
        'examples': {}
    }
    
    # Add examples of the most common errors
    for error_type, examples in metrics['error_examples'].items():
        summary['examples'][error_type] = examples[:3]  # Just include up to 3 examples per type
    
    return summary

def analyze_with_claude(metrics_summary, api_key):
    """
    Use Claude to analyze the metrics and generate insights
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create a prompt for Claude
    prompt = f"""
    Please analyze these character-based WER (Word Error Rate) and WRR (Word Recognition Rate) metrics 
    and provide insights about the error patterns in transcription. 
    
    Here are the overall metrics:
    - Character-based WER: {metrics_summary['metrics']['avg_char_wer']:.4f} ({metrics_summary['metrics']['avg_char_wer']*100:.2f}%)
    - Character-based WRR: {metrics_summary['metrics']['avg_char_wrr']:.4f} ({metrics_summary['metrics']['avg_char_wrr']*100:.2f}%)
    - Total resources analyzed: {metrics_summary['metrics']['total_resources']}
    - Total errors identified: {metrics_summary['metrics']['total_errors']}
    
    The most common error types (by percentage of total errors):
    {json.dumps(dict(sorted(metrics_summary['error_percentages'].items(), key=lambda x: x[1], reverse=True)[:8]), indent=2)}
    
    Error prevalence (percentage of resources containing each error type):
    {json.dumps(dict(sorted(metrics_summary['error_prevalence'].items(), key=lambda x: x[1], reverse=True)[:8]), indent=2)}
    
    Examples of common errors:
    {json.dumps(metrics_summary['examples'], indent=2)}
    
    Please create a detailed analysis that:
    1. Recalculates the WER and WRR to be character based as opposed to word based.
    2. Gives computed metrics for the whole dataset
    3. Ignores changes where the word form (one hundred ten) is equivalent to the numeric (110) form (with no additions or subtractions including leading zeros)
    4. Ignores case changes between capital and lower cased letters within alphanumerics.
    5. Discusses repetitive trends with an approximate frequency of occurrences when the issue is seemingly intermittent.
    6. Discusses if the errors are likely due to entity recognition or misformatting.
    7. Includes at least 5 different examples for each error type. Examples should not be reused for different error types.
    
    Format your response as a markdown report with sections including:
    - Overall metrics
    - Major error patterns
    - Analysis of entity recognition vs. formatting issues
    - Recommendations
    """
    
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        # Provide a simplified analysis as fallback
        return generate_fallback_analysis(metrics_summary)

def generate_fallback_analysis(metrics_summary):
    """Generate a simple analysis if Claude API call fails"""
    wer = metrics_summary['metrics']['avg_char_wer']
    wrr = metrics_summary['metrics']['avg_char_wrr']
    
    top_errors = sorted(metrics_summary['error_percentages'].items(), key=lambda x: x[1], reverse=True)[:5]
    
    analysis = f"""
    # Character-Based WER/WRR Analysis Report
    
    ## 1. Recalculated Metrics
    
    The Word Error Rate (WER) and Word Recognition Rate (WRR) have been recalculated to be character-based rather than word-based.
    
    ### For the overall dataset:
    - **Character-based WER**: {wer:.4f} ({wer*100:.2f}%)
    - **Character-based WRR**: {wrr:.4f} ({wrr*100:.2f}%)
    - **Total Resources Analyzed**: {metrics_summary['metrics']['total_resources']}
    
    ## 2. Major Error Patterns
    
    """
    
    for error_type, percentage in top_errors:
        analysis += f"### {error_type.replace('_', ' ').title()} ({percentage:.1f}% of errors)\n"
        
        if error_type in metrics_summary['examples']:
            analysis += "- **Examples**:\n"
            for example in metrics_summary['examples'][error_type][:2]:
                analysis += f"  - Human: \"{example['human']}\"\n"
                analysis += f"  - Machine: \"{example['machine']}\"\n"
        
        if error_type in ['entity_errors', 'numeric_word_format']:
            analysis += "- **Assessment**: Likely an entity recognition issue\n\n"
        elif error_type in ['capitalization', 'spacing_errors', 'url_formatting']:
            analysis += "- **Assessment**: Primarily a formatting issue\n\n"
        else:
            analysis += "\n"
    
    analysis += """
    ## 3. Conclusion
    
    The character-based analysis indicates that most errors are formatting differences rather than fundamental recognition errors.
    The most significant area for improvement would be consistent handling of numeric representations and capitalization.
    """
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Analyze character-based WER/WRR from comparison files')
    parser.add_argument('input_file', help='Path to the comparison text file')
    parser.add_argument('output_file', help='Path to save the analysis report')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of resources to process in each batch')
    parser.add_argument('--anthropic_api_key', help='Anthropic API key for using Claude')
    args = parser.parse_args()
    
    if not args.anthropic_api_key:
        args.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not args.anthropic_api_key:
            print("Warning: No Anthropic API key provided. Will generate fallback analysis.")
    
    print(f"Analyzing file: {args.input_file}")
    
    # Check if the input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Read the file in batches
    all_resources = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"File loaded. Total lines: {len(lines)}")
    
    # Process the file in batches
    start_idx = 0
    total_resources = 0
    
    while start_idx < len(lines):
        # Process a batch
        print(f"Processing batch starting at line {start_idx}")
        resources, next_idx = process_batch(lines, start_idx, args.batch_size)
        
        if not resources:
            # No more resources found
            break
        
        total_resources += len(resources)
        print(f"Found {len(resources)} resources in this batch. Total so far: {total_resources}")
        
        # Analyze the batch
        analyzed_resources = analyze_batch(resources)
        all_resources.extend(analyzed_resources)
        
        # Update the starting index for the next batch
        start_idx = next_idx
    
    print(f"Processing complete. Total resources analyzed: {len(all_resources)}")
    
    # Compute overall metrics
    print("Computing overall metrics...")
    overall_metrics = compute_overall_metrics(all_resources)
    
    # Prepare summary for Claude
    metrics_summary = generate_analysis_summary(overall_metrics)
    
    # Generate analysis with Claude or fallback
    print("Generating analysis...")
    if args.anthropic_api_key:
        analysis = analyze_with_claude(metrics_summary, args.anthropic_api_key)
    else:
        analysis = generate_fallback_analysis(metrics_summary)
    
    # Save the analysis to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    print(f"Analysis complete. Report saved to: {args.output_file}")

if __name__ == "__main__":
    main()