"""
Security Camera Motion Detector

A tool to detect motion in security camera footage and export clips
of detected motion events within a user-defined region of interest.

Usage:
    CLI Mode:
        python main.py --roi 100,100,500,400 [options]
    
    GUI Mode:
        python main.py --gui

Requirements:
    - opencv-python
    - numpy
    - Pillow

Install dependencies:
    pip install -r requirements.txt
"""
import sys
import os
import argparse

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def ensure_directories(input_folder=None, output_folder=None):
    """Ensure input and output directories exist."""
    input_dir = input_folder or config.INPUT_FOLDER
    output_dir = output_folder or config.OUTPUT_FOLDER
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return input_dir, output_dir


def parse_roi(roi_string):
    """Parse ROI string in format 'x1,y1,x2,y2' to tuple and normalize coordinates."""
    try:
        parts = [int(x.strip()) for x in roi_string.split(',')]
        if len(parts) != 4:
            raise ValueError("ROI must have exactly 4 values")
        x1, y1, x2, y2 = parts
        # Normalize coordinates (ensure x1 < x2 and y1 < y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid ROI format: {roi_string}. "
            "Expected format: x1,y1,x2,y2 (e.g., 100,100,500,400)"
        )


def run_gui():
    """Run the GUI application."""
    from gui import create_app
    
    print("=" * 50)
    print("Security Camera Motion Detector")
    print("=" * 50)
    print()
    
    ensure_directories()
    
    print(f"Input folder: {config.INPUT_FOLDER}")
    print(f"Output folder: {config.OUTPUT_FOLDER}")
    print("Starting GUI...")
    
    app = create_app()
    app.run()


def run_cli(args):
    """Run the CLI processing mode."""
    from video_processor import VideoProcessor
    
    print("=" * 50)
    print("Security Camera Motion Detector - CLI Mode")
    print("=" * 50)
    print()
    
    # Update config with CLI arguments
    if args.threshold is not None:
        config.MOTION_THRESHOLD = args.threshold
    if args.min_area is not None:
        config.MIN_CONTOUR_AREA = args.min_area
    if args.sensitivity is not None:
        config.MOTION_SENSITIVITY = args.sensitivity
    if args.padding_before is not None:
        config.PADDING_BEFORE_SECONDS = args.padding_before
    if args.padding_after is not None:
        config.PADDING_AFTER_SECONDS = args.padding_after
    if args.merge_gap is not None:
        config.MERGE_GAP_SECONDS = args.merge_gap
    if args.frame_skip is not None:
        config.FRAME_SKIP = args.frame_skip
    
    # Ensure directories exist
    input_dir, output_dir = ensure_directories(args.input, args.output)
    
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"ROI: {args.roi}")
    print()
    
    # Print current settings
    print("Motion Detection Settings:")
    print(f"  - Threshold: {config.MOTION_THRESHOLD}")
    print(f"  - Min contour area: {config.MIN_CONTOUR_AREA}")
    print(f"  - Sensitivity: {config.MOTION_SENSITIVITY}")
    print(f"  - Frame skip: {config.FRAME_SKIP}")
    print()
    print("Export Settings:")
    print(f"  - Padding before: {config.PADDING_BEFORE_SECONDS}s")
    print(f"  - Padding after: {config.PADDING_AFTER_SECONDS}s")
    print(f"  - Merge gap: {config.MERGE_GAP_SECONDS}s")
    print()
    
    # Create video processor
    processor = VideoProcessor(
        input_folder=input_dir,
        output_folder=output_dir,
        padding_before=config.PADDING_BEFORE_SECONDS,
        padding_after=config.PADDING_AFTER_SECONDS,
        merge_gap=config.MERGE_GAP_SECONDS
    )
    
    # Get video files
    if args.videos:
        # Process specific videos
        video_files = []
        for video in args.videos:
            if os.path.isabs(video):
                video_path = video
            else:
                video_path = os.path.join(input_dir, video)
            
            if os.path.exists(video_path):
                video_files.append(video_path)
            else:
                print(f"Warning: Video not found: {video_path}")
    else:
        # Process all videos in input folder
        video_files = processor.get_video_files()
    
    if not video_files:
        print("No video files found to process.")
        return 1
    
    print(f"Found {len(video_files)} video(s) to process:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    print()
    
    # Process videos
    total_clips = 0
    roi = args.roi
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"[{i+1}/{len(video_files)}] Processing: {video_name}")
        
        def progress_callback(status, current, total):
            if status.startswith("Analyzing frame"):
                print(f"  {status}", end='\r', flush=True)
            else:
                print(f"  {status}")
        
        try:
            clips = processor.process_video(video_path, roi, progress_callback)
            total_clips += len(clips)
            
            if clips:
                print(f"  Exported {len(clips)} clip(s):")
                for clip in clips:
                    print(f"    - {os.path.basename(clip)}")
            else:
                print("  No motion detected")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    print("=" * 50)
    print(f"Processing complete! Total clips exported: {total_clips}")
    print(f"Output folder: {output_dir}")
    print("=" * 50)
    
    return 0


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Security Camera Motion Detector - Detect motion in video footage and export clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run GUI mode:
    python main.py --gui

  Process videos with ROI:
    python main.py --roi 100,100,500,400

  Process specific videos:
    python main.py --roi 100,100,500,400 --videos video1.mp4 video2.mp4

  Custom settings:
    python main.py --roi 100,100,500,400 --threshold 30 --sensitivity 0.02 --padding-before 3 --padding-after 5

  Custom input/output folders:
    python main.py --roi 100,100,500,400 --input /path/to/videos --output /path/to/clips
        """
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument(
        '--gui',
        action='store_true',
        help='Run in GUI mode (default is CLI mode)'
    )
    
    # ROI settings
    roi_group = parser.add_argument_group('Region of Interest')
    roi_group.add_argument(
        '--roi',
        type=parse_roi,
        metavar='X1,Y1,X2,Y2',
        help='Region of interest coordinates (required for CLI mode). Format: x1,y1,x2,y2'
    )
    
    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        metavar='PATH',
        help=f'Input folder containing video files (default: {config.INPUT_FOLDER})'
    )
    io_group.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        metavar='PATH',
        help=f'Output folder for exported clips (default: {config.OUTPUT_FOLDER})'
    )
    io_group.add_argument(
        '--videos',
        nargs='+',
        metavar='FILE',
        help='Specific video files to process (default: all videos in input folder)'
    )
    
    # Motion detection settings
    motion_group = parser.add_argument_group('Motion Detection')
    motion_group.add_argument(
        '--threshold',
        type=int,
        default=None,
        metavar='N',
        help=f'Motion threshold for pixel difference (default: {config.MOTION_THRESHOLD})'
    )
    motion_group.add_argument(
        '--min-area',
        type=int,
        default=None,
        metavar='N',
        help=f'Minimum contour area for motion detection (default: {config.MIN_CONTOUR_AREA})'
    )
    motion_group.add_argument(
        '--sensitivity',
        type=float,
        default=None,
        metavar='F',
        help=f'Motion sensitivity (0.0-1.0, percentage of ROI area change) (default: {config.MOTION_SENSITIVITY})'
    )
    motion_group.add_argument(
        '--frame-skip',
        type=int,
        default=None,
        metavar='N',
        help=f'Process every Nth frame for faster processing (default: {config.FRAME_SKIP})'
    )
    
    # Export settings
    export_group = parser.add_argument_group('Export Settings')
    export_group.add_argument(
        '--padding-before',
        type=float,
        default=None,
        metavar='SEC',
        help=f'Seconds of video to include before motion (default: {config.PADDING_BEFORE_SECONDS})'
    )
    export_group.add_argument(
        '--padding-after',
        type=float,
        default=None,
        metavar='SEC',
        help=f'Seconds of video to include after motion (default: {config.PADDING_AFTER_SECONDS})'
    )
    export_group.add_argument(
        '--merge-gap',
        type=float,
        default=None,
        metavar='SEC',
        help=f'Gap threshold in seconds to merge motion segments (default: {config.MERGE_GAP_SECONDS})'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.gui:
        run_gui()
    else:
        # CLI mode requires ROI
        if args.roi is None:
            parser.error("CLI mode requires --roi argument. Use --gui for GUI mode or specify --roi X1,Y1,X2,Y2")
        
        return run_cli(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
