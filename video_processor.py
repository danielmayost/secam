"""
Video processing module for extracting and exporting video clips with detected motion.
"""
import cv2
import os
from typing import List, Tuple, Optional, Callable
from datetime import datetime
import config
from motion_detector import MotionDetector


class VideoProcessor:
    """Processes videos to detect motion and export clips."""
    
    def __init__(
        self,
        input_folder: str = config.INPUT_FOLDER,
        output_folder: str = config.OUTPUT_FOLDER,
        padding_before: float = config.PADDING_BEFORE_SECONDS,
        padding_after: float = config.PADDING_AFTER_SECONDS,
        merge_gap: float = config.MERGE_GAP_SECONDS
    ):
        """
        Initialize the video processor.
        
        Args:
            input_folder: Folder containing input video files
            output_folder: Folder to save exported clips
            padding_before: Seconds to include before motion
            padding_after: Seconds to include after motion
            merge_gap: Gap threshold in seconds to merge motion segments
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.padding_before = padding_before
        self.padding_after = padding_after
        self.merge_gap = merge_gap
        self.motion_detector = MotionDetector()
        self._stop_requested = False
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
    
    def request_stop(self):
        """Request to stop processing."""
        self._stop_requested = True
    
    def is_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self._stop_requested
    
    def get_video_files(self) -> List[str]:
        """
        Get all MP4 video files in the input folder.
        
        Returns:
            List of full paths to video files
        """
        if not os.path.exists(self.input_folder):
            return []
        
        video_extensions = ('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV')
        video_files = []
        
        for filename in os.listdir(self.input_folder):
            if filename.endswith(video_extensions):
                video_files.append(os.path.join(self.input_folder, filename))
        
        return sorted(video_files)
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': video_path,
            'filename': os.path.basename(video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def export_clip(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Export a clip from a video file.
        
        Args:
            video_path: Path to the source video
            start_frame: Starting frame number
            end_frame: Ending frame number
            output_name: Optional custom output filename
            progress_callback: Optional callback function(current_frame, total_frames)
            
        Returns:
            Path to the exported clip
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Apply padding
        padding_frames_before = int(self.padding_before * fps)
        padding_frames_after = int(self.padding_after * fps)
        
        actual_start = max(0, start_frame - padding_frames_before)
        actual_end = min(total_frames - 1, end_frame + padding_frames_after)
        
        # Generate output filename
        if output_name is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            start_time = actual_start / fps
            end_time = actual_end / fps
            output_name = f"{base_name}_motion_{start_time:.1f}s-{end_time:.1f}s_{timestamp}.mp4"
        
        output_path = os.path.join(self.output_folder, output_name)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        
        # Write frames
        frames_to_write = actual_end - actual_start + 1
        frames_written = 0
        
        while frames_written < frames_to_write:
            ret, frame = cap.read()
            if not ret:
                break
            
            writer.write(frame)
            frames_written += 1
            
            if progress_callback:
                progress_callback(frames_written, frames_to_write)
        
        cap.release()
        writer.release()
        
        return output_path
    
    def process_video(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> List[str]:
        """
        Process a video file, detect motion, and export clips.
        
        Args:
            video_path: Path to the video file
            roi: Region of interest as (x1, y1, x2, y2)
            progress_callback: Optional callback(status_message, current, total)
            
        Returns:
            List of paths to exported clips
        """
        exported_clips = []
        
        # Check for stop request
        if self._stop_requested:
            return []
        
        # Analyze video for motion
        if progress_callback:
            progress_callback("Analyzing video for motion...", 0, 100)
        
        def analysis_progress(current, total):
            if progress_callback:
                progress_callback(
                    f"Analyzing frame {current}/{total}",
                    int(current * 50 / total),  # First 50% for analysis
                    100
                )
            # Return True to signal stop
            return self._stop_requested
        
        motion_ranges = self.motion_detector.analyze_video_for_motion(
            video_path,
            roi,
            merge_gap_seconds=self.merge_gap,
            progress_callback=analysis_progress,
            stop_check=self.is_stop_requested
        )
        
        # Check for stop request
        if self._stop_requested:
            return []
        
        if not motion_ranges:
            if progress_callback:
                progress_callback("No motion detected", 100, 100)
            return []
        
        # Export clips for each motion range
        total_clips = len(motion_ranges)
        for i, (start_frame, end_frame) in enumerate(motion_ranges):
            # Check for stop request
            if self._stop_requested:
                return exported_clips
            
            if progress_callback:
                progress_callback(
                    f"Exporting clip {i+1}/{total_clips}",
                    50 + int((i + 1) * 50 / total_clips),
                    100
                )
            
            clip_path = self.export_clip(video_path, start_frame, end_frame)
            exported_clips.append(clip_path)
        
        if progress_callback:
            progress_callback(f"Exported {len(exported_clips)} clips", 100, 100)
        
        return exported_clips
    
    def process_all_videos(
        self,
        roi: Tuple[int, int, int, int],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    ) -> dict:
        """
        Process all video files in the input folder.
        
        Args:
            roi: Region of interest as (x1, y1, x2, y2)
            progress_callback: Optional callback(status, video_index, total_videos, video_name)
            
        Returns:
            Dictionary mapping video paths to lists of exported clip paths
        """
        video_files = self.get_video_files()
        
        if not video_files:
            return {}
        
        results = {}
        total_videos = len(video_files)
        
        for i, video_path in enumerate(video_files):
            # Check for stop request
            if self._stop_requested:
                break
            
            video_name = os.path.basename(video_path)
            
            if progress_callback:
                progress_callback(
                    f"Processing video {i+1}/{total_videos}",
                    i,
                    total_videos,
                    video_name
                )
            
            try:
                def video_progress(status, current, total):
                    if progress_callback:
                        progress_callback(status, i, total_videos, video_name)
                
                clips = self.process_video(video_path, roi, video_progress)
                results[video_path] = clips
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[video_path] = []
        
        return results


def get_first_frame(video_path: str) -> Optional[tuple]:
    """
    Get the first frame of a video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frame, width, height) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    height, width = frame.shape[:2]
    cap.release()
    
    return frame, width, height
