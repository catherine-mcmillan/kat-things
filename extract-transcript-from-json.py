import json
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_transcripts(input_dir: str, output_dir: str):
    """
    Extract transcripts from JSON files in input directory and write to individual txt files
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to write transcript files to
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob('*.json'))
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file) as jf:
                data = json.load(jf)
                transcript = data['channel']['alternatives'][0]['transcript']
                
                # Create output txt file with same name as json file
                output_file = output_path / f"{json_file.stem}.txt"
                with open(output_file, 'w') as f:
                    f.write(transcript)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--output_dir", required=True, help="Output directory for transcript files")
    args = parser.parse_args()
    
    extract_transcripts(args.input_dir, args.output_dir)