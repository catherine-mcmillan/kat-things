import shutil
import csv
import os
import sys
import glob

def find_file_with_extension(file_path):
    """
    Finds a file with any audio extension matching the given file path (ignoring extension).

    Args:
        file_path (str): Path to the file without the extension.

    Returns:
        str: Full path to the file if found, otherwise None.
    """
    # List of common audio file extensions
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.aiff',".opus",".webm",".unk_audio"]
    
    directory, base_name = os.path.split(file_path)
    base_name_without_extension = os.path.splitext(base_name)[0]
    
    # Glob pattern to match the file with any of the audio extensions
    for ext in audio_extensions:
        potential_file = os.path.join(directory, base_name_without_extension + ext)
        if os.path.exists(potential_file):
            return potential_file

    # If not found, use glob to look for any matching file (with any extension)
    pattern = os.path.join(directory, base_name_without_extension + ".*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    
    return None

def copy_files_from_csv(csv_file_path, destination_directory):
    """
    Copies files listed in a CSV file to a specified destination directory.
    
    Args:
        csv_file_path (str): Path to the CSV file containing file paths.
        destination_directory (str): Path to the destination directory.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Read the CSV file and copy each file to the destination directory
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Skip the header if there is one (uncomment the next line if needed)
        # next(csvreader)

        for row in csvreader:
            if not row:  # Skip empty rows
                continue

            file_path = row[0]  # Assuming file paths are in the first column
            
            # Find the file with any audio extension
            actual_file_path = find_file_with_extension(file_path)

            if actual_file_path and os.path.exists(actual_file_path):
                shutil.copy(actual_file_path, destination_directory)
                print(f"Copied {actual_file_path} to {destination_directory}")
            else:
                print(f"File not found: {file_path}")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <csv_file_path> <destination_directory>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    destination_directory = sys.argv[2]

    copy_files_from_csv(csv_file_path, destination_directory)
