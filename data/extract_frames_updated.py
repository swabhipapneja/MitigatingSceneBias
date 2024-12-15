import os
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path, video_output_dir, frame_rate):
    """Process a single video to extract frames."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // frame_rate)  # Extract every nth frame based on frame_rate

        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)  # Save frame as JPEG

            success, frame = cap.read()
            frame_count += 1

        cap.release()
        return f"Frames extracted for video: {os.path.basename(video_path)}"
    except Exception as e:
        return f"Failed to process video {video_path}: {e}"

async def extract_frames(video_dir, output_dir, frame_rate=1):
    """
    Extract frames from each video in the directory structure asynchronously and save them as images.
    
    Parameters:
        video_dir (str): Path to the root video directory.
        output_dir (str): Path to the output directory to save extracted frames.
        frame_rate (int): Number of frames to save per second of video.
    """
    # Verify the video directory exists
    if not os.path.exists(video_dir):
        print(f"Video directory '{video_dir}' does not exist.")
        return
    
    print("Starting frame extraction...")

    # Create a thread pool for I/O-bound tasks
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        video_counter = 0  # To keep track of how many videos have been processed

        # Process each class folder in the video directory
        class_folders = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
        print(f"Found {len(class_folders)} class folders in '{video_dir}'")

        for class_folder in class_folders:
            class_folder_path = os.path.join(video_dir, class_folder)
            class_output_dir = os.path.join(output_dir, class_folder)

            # Check if the class folder has already been processed
            if os.path.exists(class_output_dir):
                print(f"Skipping extraction for already processed class folder: {class_folder}")
                continue  # Skip to the next class folder

            # Create output directory for this class folder
            os.makedirs(class_output_dir, exist_ok=True)

            # List all video files directly inside the class folder
            video_files = [f for f in os.listdir(class_folder_path) if f.endswith('.mp4')]
            print(f"Found {len(video_files)} video files in '{class_folder_path}'")

            for video_name in video_files:
                video_path = os.path.join(class_folder_path, video_name)
                video_basename = os.path.splitext(video_name)[0]

                # Create output directory for this video within the class folder
                video_output_dir = os.path.join(class_output_dir, video_basename)
                os.makedirs(video_output_dir, exist_ok=True)

                # Submit a task to process the video asynchronously
                print(f"Scheduling extraction for video: {video_path}")
                task = loop.run_in_executor(executor, process_video, video_path, video_output_dir, frame_rate)
                tasks.append(task)

                # Increment video counter and log every 100 videos
                video_counter += 1
                if video_counter % 100 == 0:
                    print(f"Scheduled {video_counter} videos for processing.")

        if not tasks:
            print("No tasks were scheduled. Check if video files exist in the specified directory.")
            return

        # Wait for all tasks to complete
        print(f"Waiting for {len(tasks)} tasks to complete...")
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result)

        print("Frame extraction completed.")

# Usage example
video_dir = 'classwise/mini-kinetics-200-30-199/train_videos/'
output_dir = '/scratch/supalami/ProjectDataSplitwise/extracted_frames/'

async def main():
    # Run the asynchronous extraction
    await extract_frames(video_dir, output_dir, frame_rate=1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred while running the extraction: {e}")
