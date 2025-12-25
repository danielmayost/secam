"""
Motion detection module for analyzing video frames within a specified region of interest.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import config


class MotionDetector:
    """Detects motion within a specified region of interest in video frames."""
    
    def __init__(
        self,
        threshold: int = config.MOTION_THRESHOLD,
        min_contour_area: int = config.MIN_CONTOUR_AREA,
        sensitivity: float = config.MOTION_SENSITIVITY
    ):
        """
        Initialize the motion detector.
        
        Args:
            threshold: Threshold for pixel difference to be considered motion
            min_contour_area: Minimum contour area to be considered motion
            sensitivity: Percentage of ROI area that needs to change
        """
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.sensitivity = sensitivity
        self.previous_frame: Optional[np.ndarray] = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
    
    def reset(self):
        """Reset the detector state for a new video."""
        self.previous_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
    
    def detect_motion(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Detect motion in the frame within the specified region of interest.
        
        Args:
            frame: The current video frame
            roi: Region of interest as (x1, y1, x2, y2), or None for full frame
            
        Returns:
            Tuple of (motion_detected, motion_percentage, visualization_frame)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Extract ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            # Normalize coordinates (ensure x1 < x2 and y1 < y2)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            # Ensure coordinates are within frame bounds
            h, w = gray.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Ensure ROI has valid dimensions
            if x2 <= x1 or y2 <= y1:
                self.previous_frame = None
                return False, 0.0, frame.copy()
            roi_gray = gray[y1:y2, x1:x2]
        else:
            roi_gray = gray
            x1, y1 = 0, 0
        
        # Initialize previous frame if needed
        if self.previous_frame is None:
            self.previous_frame = roi_gray.copy()
            visualization = frame.copy()
            if roi is not None:
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return False, 0.0, visualization
        
        # Resize previous frame if ROI size changed
        if self.previous_frame.shape != roi_gray.shape:
            self.previous_frame = roi_gray.copy()
            visualization = frame.copy()
            if roi is not None:
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return False, 0.0, visualization
        
        # Compute absolute difference
        frame_delta = cv2.absdiff(self.previous_frame, roi_gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate motion percentage
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check if significant motion detected
        significant_motion = False
        total_contour_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                significant_motion = True
                total_contour_area += area
        
        # Motion is detected if percentage exceeds sensitivity threshold
        motion_detected = significant_motion and motion_percentage >= self.sensitivity
        
        # Create visualization frame
        visualization = frame.copy()
        
        # Draw ROI rectangle
        if roi is not None:
            color = (0, 0, 255) if motion_detected else (0, 255, 0)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
            
            # Draw contours within ROI
            for contour in contours:
                if cv2.contourArea(contour) >= self.min_contour_area:
                    # Offset contour coordinates to frame coordinates
                    contour_offset = contour + np.array([x1, y1])
                    cv2.drawContours(visualization, [contour_offset], -1, (0, 255, 255), 2)
        
        # Add motion indicator text
        status = "MOTION DETECTED" if motion_detected else "No Motion"
        cv2.putText(
            visualization,
            f"{status} ({motion_percentage*100:.1f}%)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if motion_detected else (0, 255, 0),
            2
        )
        
        # Update previous frame
        self.previous_frame = roi_gray.copy()
        
        return motion_detected, motion_percentage, visualization
    
    def analyze_video_for_motion(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        frame_skip: int = config.FRAME_SKIP,
        merge_gap_seconds: float = config.MERGE_GAP_SECONDS,
        progress_callback=None,
        stop_check=None
    ) -> List[Tuple[int, int]]:
        """
        Analyze a video file and return frame ranges where motion is detected.
        
        Args:
            video_path: Path to the video file
            roi: Region of interest as (x1, y1, x2, y2)
            frame_skip: Process every Nth frame
            merge_gap_seconds: Gap threshold in seconds to merge motion segments
            progress_callback: Optional callback function(current_frame, total_frames)
            stop_check: Optional callable that returns True if processing should stop
            
        Returns:
            List of tuples (start_frame, end_frame) where motion was detected
        """
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        motion_ranges = []
        current_motion_start = None
        frame_count = 0
        
        while True:
            # Check for stop request
            if stop_check and stop_check():
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                motion_detected, _, _ = self.detect_motion(frame, roi)
                
                if motion_detected:
                    if current_motion_start is None:
                        current_motion_start = frame_count
                else:
                    if current_motion_start is not None:
                        motion_ranges.append((current_motion_start, frame_count - 1))
                        current_motion_start = None
            
            frame_count += 1
            
            if progress_callback:
                progress_callback(frame_count, total_frames)
        
        # Don't forget to close the last motion range
        if current_motion_start is not None:
            motion_ranges.append((current_motion_start, frame_count - 1))
        
        cap.release()
        
        # Merge nearby motion ranges using the configurable gap threshold
        merged_ranges = self._merge_ranges(motion_ranges, fps, merge_gap_seconds)
        
        return merged_ranges
    
    def _merge_ranges(
        self,
        ranges: List[Tuple[int, int]],
        fps: float,
        gap_threshold_seconds: float = 2.0
    ) -> List[Tuple[int, int]]:
        """
        Merge motion ranges that are close together.
        
        Args:
            ranges: List of (start_frame, end_frame) tuples
            fps: Frames per second of the video
            gap_threshold_seconds: Maximum gap between ranges to merge
            
        Returns:
            Merged list of ranges
        """
        if not ranges:
            return []
        
        gap_threshold_frames = int(gap_threshold_seconds * fps)
        
        # Sort by start frame
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        merged = [sorted_ranges[0]]
        
        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # If this range is close to the previous one, merge them
            if start - last_end <= gap_threshold_frames:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged
