"""
Configuration settings for the video motion detection application.
"""
import os

# Default directories (can be changed via GUI)
INPUT_FOLDER = os.path.join(os.getcwd(), "input_videos")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "output_clips")

# Motion detection settings
MOTION_THRESHOLD = 25  # Threshold for pixel difference to be considered motion
MIN_CONTOUR_AREA = 500  # Minimum area of contour to be considered motion
MOTION_SENSITIVITY = 0.01  # Percentage of ROI area that needs to change to trigger detection

# Video export settings
PADDING_BEFORE_SECONDS = 2  # Seconds of video to include before motion
PADDING_AFTER_SECONDS = 2  # Seconds of video to include after motion
MIN_CLIP_DURATION_SECONDS = 2  # Minimum duration for exported clips
MERGE_GAP_SECONDS = 10  # Gap threshold to merge motion segments (throttle)

# GUI settings
PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 600

# Processing settings
FRAME_SKIP = 2  # Process every Nth frame for faster processing (1 = process all frames)
