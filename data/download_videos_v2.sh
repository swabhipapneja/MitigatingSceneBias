#!/bin/bash

# Directories and file paths
MISSING_VIDEOS_DIR="/scratch/supalami/ProjectDataSplitwise/mkworkingdir/missing_videos"
INPUT_FILE="/scratch/supalami/ProjectDataSplitwise/mkworkingdir/script1nov21output.txt"

# Maximum number of concurrent jobs
MAX_CONCURRENT_JOBS=$(nproc) # Uses the number of CPU cores

# Function to download a video
download_video() {
    local class_label="$1"
    local video_id="$2"
    
    # Create the class directory if it doesn't exist
    local class_dir="${MISSING_VIDEOS_DIR}/${class_label}"
    mkdir -p "$class_dir"

    # Define the output path for the video
    local output_path="${class_dir}/${video_id}.mp4"

    # Skip videos that already exist
    if [[ -f "$output_path" ]]; then
        echo "Skipping: Video $video_id already exists in $class_dir."
        return
    fi

    # Download the video using yt-dlp
    echo "Downloading: Class $class_label, Video ID $video_id"
    yt-dlp -o "${output_path}" "https://www.youtube.com/watch?v=${video_id}"

    # Check the exit status of yt-dlp
    if [[ $? -eq 0 ]]; then
        echo "Downloaded: $video_id"
    else
        echo "Failed to download: $video_id"
    fi
}

# Check if yt-dlp is installed
if ! command -v yt-dlp &> /dev/null; then
    echo "yt-dlp not found. Please install it using 'pip install yt-dlp'."
    exit 1
fi

# Check if the input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: $INPUT_FILE does not exist."
    exit 1
fi

# Process each line in the input file
while IFS=',' read -r class_label video_id; do
    # Launch the download_video function in the background
    download_video "$class_label" "$video_id" &

    # Limit the number of concurrent jobs
    while [[ $(jobs -r | wc -l) -ge $MAX_CONCURRENT_JOBS ]]; do
        sleep 1 # Wait for an available slot
    done
done < "$INPUT_FILE"

# Wait for all background jobs to complete
wait

echo "Download process completed."

