import subprocess
import argparse
import sys
import io
from pathlib import Path

# EXPERT FIX: Force UTF-8 encoding for the Windows Console to prevent 'charmap' errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv"}

def batch_process(input_folder, logo_path, extra_flags):
    input_path = Path(input_folder)
    
    if not input_path.is_dir():
        print(f"Error: {input_folder} is not a valid directory.")
        return

    video_files = [
        f for f in input_path.iterdir() 
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    print(f"🚀 Found {len(video_files)} videos. Starting batch processing...\n")

    for i, video in enumerate(video_files, 1):
        print(f"--- [{i}/{len(video_files)}] Processing: {video.name} ---")
        
        cmd = [
            "python", "process.py", 
            str(video.resolve()), 
            str(Path(logo_path).resolve())
        ] + extra_flags

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully processed {video.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {video.name}: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process a folder of videos using process.py")
    parser.add_argument("folder", help="Path to the folder containing videos")
    parser.add_argument("logo", help="Path to the logo file to use for all videos")
    args, unknown = parser.parse_known_args()
    batch_process(args.folder, args.logo, unknown)

