import json
import glob
from pathlib import Path
from tqdm import tqdm
import argparse

def merge_transcripts(json_dir, output_dir):
    """
    Merge transcripts from multiple JSON files, handling overlapping segments.
    Outputs individual txt files for each merged transcript.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in directory
    json_files = sorted(glob.glob(str(Path(json_dir) / "*.json")))
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # More robust transcript extraction
            all_words = []
            
            # Extract words from all responses
            if "responses" in data:
                for response in data["responses"]:
                    if ("channel" in response and 
                        "alternatives" in response["channel"] and 
                        response["channel"]["alternatives"]):
                        
                        for alt in response["channel"]["alternatives"]:
                            if "words" in alt and alt["words"]:
                                all_words.extend(alt["words"])
            
            if not all_words:
                print(f"Warning: No words found in {json_file}")
                continue
            
            # Sort words by start time
            all_words.sort(key=lambda x: x["start"])
            
            if all_words:
                # Step 1: Time-window based grouping
                time_window = 0.1
                time_groups = []
                current_group = [all_words[0]]
                
                for word in all_words[1:]:
                    if abs(word["start"] - current_group[-1]["start"]) <= time_window:
                        current_group.append(word)
                    else:
                        time_groups.append(current_group)
                        current_group = [word]
                
                if current_group:
                    time_groups.append(current_group)

                # Step 2: Within each time group, keep non-exact duplicates
                merged_words = []
                for group in time_groups:
                    seen_words = set()
                    group_words = []
                    
                    for word in group:
                        # Create a key combining start time (rounded) and the word itself
                        word_key = (round(word["start"], 2), word.get("punctuated_word", word.get("word", "")))
                        
                        if word_key not in seen_words:
                            group_words.append(word)
                            seen_words.add(word_key)
                    
                    # For each group, add all unique words
                    merged_words.extend(sorted(group_words, key=lambda x: x["start"]))

                # Sort final merged words by start time
                merged_words.sort(key=lambda x: x["start"])

                # Add debug logging
                print(f"File: {json_file}")
                print(f"Original word count: {len(all_words)}")
                print(f"Merged word count: {len(merged_words)}")
                
                # Reconstruct transcript
                transcript = ""
                for word in merged_words:
                    if "punctuated_word" in word:
                        transcript += word["punctuated_word"] + " "
                    elif "word" in word:
                        transcript += word["word"] + " "
                
                transcript = transcript.strip()
                
                if not transcript:
                    print(f"Warning: Empty transcript generated for {json_file}")
                    continue

                # Write transcript to txt file
                output_file = output_dir / f"{Path(json_file).stem}.txt"
                with open(output_file, "w", encoding='utf-8') as f:
                    f.write(transcript)
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--output_dir", required=True, help="Output directory for transcript txt files")
    args = parser.parse_args()
    
    merge_transcripts(args.input_dir, args.output_dir)