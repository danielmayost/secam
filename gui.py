"""
GUI module for video preview and region of interest selection.
"""
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
from typing import Optional, Tuple
import config
from video_processor import VideoProcessor, get_first_frame


class VideoPreviewWindow:
    """Window for previewing video and selecting region of interest."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the preview window."""
        self.root = root
        self.root.title("Security Camera Motion Detector")
        self.root.geometry("1000x750")
        
        # Variables
        self.video_path: Optional[str] = None
        self.input_folder: str = config.INPUT_FOLDER
        self.output_folder: str = config.OUTPUT_FOLDER
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[any] = None
        self.original_frame: Optional[any] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.display_width: int = config.PREVIEW_WIDTH
        self.display_height: int = config.PREVIEW_HEIGHT
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        
        # ROI drawing variables
        self.roi_start: Optional[Tuple[int, int]] = None
        self.roi_end: Optional[Tuple[int, int]] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.drawing: bool = False
        
        # Processing state
        self.processing: bool = False
        self.processor: Optional[VideoProcessor] = None
        
        # Create GUI elements
        self._create_widgets()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Folder selection
        ttk.Label(control_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W)
        self.input_folder_var = tk.StringVar(value=self.input_folder)
        self.input_entry = ttk.Entry(control_frame, textvariable=self.input_folder_var, width=50)
        self.input_entry.grid(row=0, column=1, padx=5)
        self.input_browse_btn = ttk.Button(control_frame, text="Browse...", command=self._browse_input_folder)
        self.input_browse_btn.grid(row=0, column=2, padx=5)

        ttk.Label(control_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.output_folder_var = tk.StringVar(value=self.output_folder)
        self.output_entry = ttk.Entry(control_frame, textvariable=self.output_folder_var, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=(5, 0))
        self.output_browse_btn = ttk.Button(control_frame, text="Browse...", command=self._browse_output_folder)
        self.output_browse_btn.grid(row=1, column=2, padx=5, pady=(5, 0))

        # Video selection
        ttk.Label(control_frame, text="Preview Video:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.video_combo = ttk.Combobox(control_frame, width=47, state="readonly")
        self.video_combo.grid(row=2, column=1, padx=5, pady=(5, 0))
        self.video_combo.bind("<<ComboboxSelected>>", self._on_video_selected)
        self.refresh_btn = ttk.Button(control_frame, text="Refresh", command=self._refresh_video_list)
        self.refresh_btn.grid(row=2, column=2, padx=5, pady=(5, 0))
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(main_frame, text="Video Preview - Draw rectangle to select detection area", padding="5")
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas for video preview
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.display_width,
            height=self.display_height,
            bg="black",
            cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind mouse events for drawing
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        
        # ROI info frame
        roi_frame = ttk.Frame(main_frame)
        roi_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.roi_label = ttk.Label(roi_frame, text="ROI: Not selected - Draw a rectangle on the video")
        self.roi_label.grid(row=0, column=0, sticky=tk.W)

        self.clear_roi_btn = ttk.Button(roi_frame, text="Clear ROI", command=self._clear_roi)
        self.clear_roi_btn.grid(row=0, column=1, padx=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(settings_frame, text="Padding Before (sec):").grid(row=0, column=0, padx=5)
        self.padding_before_var = tk.StringVar(value=str(config.PADDING_BEFORE_SECONDS))
        self.padding_before_entry = ttk.Entry(settings_frame, textvariable=self.padding_before_var, width=8)
        self.padding_before_entry.grid(row=0, column=1, padx=5)

        ttk.Label(settings_frame, text="Padding After (sec):").grid(row=0, column=2, padx=5)
        self.padding_after_var = tk.StringVar(value=str(config.PADDING_AFTER_SECONDS))
        self.padding_after_entry = ttk.Entry(settings_frame, textvariable=self.padding_after_var, width=8)
        self.padding_after_entry.grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Motion Sensitivity:").grid(row=0, column=4, padx=5)
        self.sensitivity_var = tk.StringVar(value=str(config.MOTION_SENSITIVITY))
        self.sensitivity_entry = ttk.Entry(settings_frame, textvariable=self.sensitivity_var, width=8)
        self.sensitivity_entry.grid(row=0, column=5, padx=5)

        ttk.Label(settings_frame, text="Merge Gap (sec):").grid(row=0, column=6, padx=5)
        self.merge_gap_var = tk.StringVar(value=str(config.MERGE_GAP_SECONDS))
        self.merge_gap_entry = ttk.Entry(settings_frame, textvariable=self.merge_gap_var, width=8)
        self.merge_gap_entry.grid(row=0, column=7, padx=5)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.process_btn = ttk.Button(
            button_frame,
            text="Process All Videos",
            command=self._start_processing
        )
        self.process_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop",
            command=self._stop_processing,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.open_output_btn = ttk.Button(button_frame, text="Open Output Folder", command=self._open_output_folder)
        self.open_output_btn.grid(row=0, column=2, padx=5)

        # List of widgets to disable during processing
        self.widgets_to_disable = [
            self.input_entry, self.input_browse_btn,
            self.output_entry, self.output_browse_btn,
            self.video_combo, self.refresh_btn,
            self.clear_roi_btn,
            self.padding_before_entry, self.padding_after_entry, self.sensitivity_entry, self.merge_gap_entry,
            self.open_output_btn
        ]
        
    def _browse_input_folder(self):
        """Browse for input folder."""
        folder = filedialog.askdirectory(initialdir=self.input_folder_var.get())
        if folder:
            self.input_folder_var.set(folder)
            self.input_folder = folder
            self._refresh_video_list()
    
    def _browse_output_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get())
        if folder:
            self.output_folder_var.set(folder)
            self.output_folder = folder
    
    def _refresh_video_list(self):
        """Refresh the list of videos in the input folder."""
        self.input_folder = self.input_folder_var.get()
        
        if not os.path.exists(self.input_folder):
            os.makedirs(self.input_folder, exist_ok=True)
            
        video_extensions = ('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV', '.mkv', '.MKV')
        videos = [f for f in os.listdir(self.input_folder) if f.endswith(video_extensions)]
        
        self.video_combo['values'] = videos
        if videos:
            self.video_combo.current(0)
            self._on_video_selected(None)
        else:
            self.video_combo.set("")
            self._clear_canvas()
    
    def _on_video_selected(self, event):
        """Handle video selection from combobox."""
        video_name = self.video_combo.get()
        if video_name:
            video_path = os.path.join(self.input_folder, video_name)
            self._load_video(video_path)

    def _load_video(self, video_path: str):
        """Load a video for preview."""
        self.video_path = video_path
        
        result = get_first_frame(video_path)
        if result is None:
            messagebox.showerror("Error", f"Could not open video: {video_path}")
            return
        
        frame, self.video_width, self.video_height = result
        self.original_frame = frame
        
        # Calculate scaling
        self.scale_x = self.display_width / self.video_width
        self.scale_y = self.display_height / self.video_height
        scale = min(self.scale_x, self.scale_y)
        
        new_width = int(self.video_width * scale)
        new_height = int(self.video_height * scale)
        
        self.scale_x = new_width / self.video_width
        self.scale_y = new_height / self.video_height
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (new_width, new_height))
        
        # Update canvas size
        self.canvas.config(width=new_width, height=new_height)
        
        self._display_frame(display_frame)
        self._clear_roi()
        
        self.status_label.config(text=f"Loaded: {os.path.basename(video_path)} ({self.video_width}x{self.video_height})")
    
    def _display_frame(self, frame):
        """Display a frame on the canvas."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Draw ROI if exists
        if self.roi:
            x1, y1, x2, y2 = self.roi
            # Scale to display coordinates
            dx1 = int(x1 * self.scale_x)
            dy1 = int(y1 * self.scale_y)
            dx2 = int(x2 * self.scale_x)
            dy2 = int(y2 * self.scale_y)
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="green", width=2, tags="roi")
    
    def _clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.photo = None
    
    def _on_mouse_press(self, event):
        """Handle mouse press for ROI drawing."""
        if self.original_frame is None:
            return
        
        self.drawing = True
        self.roi_start = (event.x, event.y)
        self.roi_end = None
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag for ROI drawing."""
        if not self.drawing or self.original_frame is None:
            return
        
        self.roi_end = (event.x, event.y)
        self._draw_temp_rectangle()
    
    def _on_mouse_release(self, event):
        """Handle mouse release for ROI drawing."""
        if not self.drawing or self.original_frame is None:
            return
        
        self.drawing = False
        self.roi_end = (event.x, event.y)
        
        if self.roi_start and self.roi_end:
            # Convert display coordinates to video coordinates
            x1 = int(min(self.roi_start[0], self.roi_end[0]) / self.scale_x)
            y1 = int(min(self.roi_start[1], self.roi_end[1]) / self.scale_y)
            x2 = int(max(self.roi_start[0], self.roi_end[0]) / self.scale_x)
            y2 = int(max(self.roi_start[1], self.roi_end[1]) / self.scale_y)
            
            # Ensure within bounds
            x1 = max(0, min(x1, self.video_width))
            y1 = max(0, min(y1, self.video_height))
            x2 = max(0, min(x2, self.video_width))
            y2 = max(0, min(y2, self.video_height))
            
            # Check minimum size
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.roi = (x1, y1, x2, y2)
                self.roi_label.config(text=f"ROI: ({x1}, {y1}) to ({x2}, {y2}) - Size: {x2-x1}x{y2-y1}")
            else:
                self.roi = None
                self.roi_label.config(text="ROI: Too small - Draw a larger rectangle")
        
        self._redraw_frame()
    
    def _draw_temp_rectangle(self):
        """Draw temporary rectangle while dragging."""
        if self.roi_start and self.roi_end:
            self._redraw_frame()
            self.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1],
                self.roi_end[0], self.roi_end[1],
                outline="yellow", width=2, tags="temp_roi"
            )
    
    def _redraw_frame(self):
        """Redraw the current frame with ROI."""
        if self.original_frame is None:
            return
        
        # Resize frame for display
        new_width = int(self.video_width * self.scale_x)
        new_height = int(self.video_height * self.scale_y)
        display_frame = cv2.resize(self.original_frame, (new_width, new_height))
        
        self._display_frame(display_frame)
    
    def _clear_roi(self):
        """Clear the current ROI."""
        self.roi = None
        self.roi_start = None
        self.roi_end = None
        self.roi_label.config(text="ROI: Not selected - Draw a rectangle on the video")
        self._redraw_frame()
    
    def _start_processing(self):
        """Start processing all videos."""
        if self.roi is None:
            messagebox.showwarning("Warning", "Please draw a region of interest (ROI) first!")
            return
        
        self.input_folder = self.input_folder_var.get()
        self.output_folder = self.output_folder_var.get()
        
        if not os.path.exists(self.input_folder):
            messagebox.showerror("Error", "Input folder does not exist!")
            return
        
        try:
            padding_before = float(self.padding_before_var.get())
            padding_after = float(self.padding_after_var.get())
            sensitivity = float(self.sensitivity_var.get())
            merge_gap = float(self.merge_gap_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid settings values!")
            return
        
        # Update config
        config.MOTION_SENSITIVITY = sensitivity
        
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Disable other widgets during processing
        for widget in self.widgets_to_disable:
            if widget == self.video_combo:
                widget.config(state=tk.DISABLED)
            else:
                widget.config(state=tk.DISABLED)

        # Start processing in a separate thread
        self.processor = VideoProcessor(
            input_folder=self.input_folder,
            output_folder=self.output_folder,
            padding_before=padding_before,
            padding_after=padding_after,
            merge_gap=merge_gap
        )
        
        thread = threading.Thread(target=self._process_videos_thread)
        thread.daemon = True
        thread.start()
    
    def _process_videos_thread(self):
        """Process videos in a separate thread."""
        was_stopped = False
        try:
            def progress_callback(status, video_idx, total_videos, video_name):
                if not self.processing:
                    return
                
                progress = (video_idx / total_videos) * 100 if total_videos > 0 else 0
                self.root.after(0, lambda: self._update_progress(progress, f"{status}: {video_name}"))
            
            results = self.processor.process_all_videos(self.roi, progress_callback)
            
            # Check if we were stopped
            if self.processor.is_stop_requested():
                was_stopped = True
                total_clips = sum(len(clips) for clips in results.values())
                self.root.after(0, lambda: self._processing_stopped(total_clips, len(results)))
            elif self.processing:
                total_clips = sum(len(clips) for clips in results.values())
                self.root.after(0, lambda: self._processing_complete(total_clips, len(results)))
                
        except Exception as e:
            if not was_stopped:
                self.root.after(0, lambda: self._processing_error(str(e)))
        finally:
            self.processing = False
            self.root.after(0, self._reset_ui)
    
    def _update_progress(self, progress: float, status: str):
        """Update progress bar and status label."""
        self.progress_var.set(progress)
        self.status_label.config(text=status)
    
    def _processing_complete(self, total_clips: int, total_videos: int):
        """Handle processing completion."""
        self.progress_var.set(100)
        messagebox.showinfo(
            "Complete",
            f"Processing complete!\n\nProcessed {total_videos} videos\nExported {total_clips} clips"
        )
        self.status_label.config(text=f"Complete: Exported {total_clips} clips from {total_videos} videos")
    
    def _processing_error(self, error: str):
        """Handle processing error."""
        messagebox.showerror("Error", f"Processing error: {error}")
        self.status_label.config(text=f"Error: {error}")
    
    def _processing_stopped(self, total_clips: int, total_videos: int):
        """Handle processing stopped by user."""
        self.status_label.config(text=f"Stopped: Exported {total_clips} clips from {total_videos} videos before stopping")
    
    def _reset_ui(self):
        """Reset UI after processing."""
        self.process_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        # Re-enable widgets
        for widget in self.widgets_to_disable:
            if widget == self.video_combo:
                widget.config(state="readonly")
            else:
                widget.config(state=tk.NORMAL)
    
    def _stop_processing(self):
        """Stop the current processing."""
        # Ask for confirmation before stopping
        confirmed = messagebox.askyesno(
            "Confirm Stop",
            "Are you sure you want to stop the video processing?"
        )

        if confirmed:
            self.processing = False
            if self.processor:
                self.processor.request_stop()
            self.status_label.config(text="Stopping...")
    
    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_folder = self.output_folder_var.get()
        if os.path.exists(output_folder):
            os.startfile(output_folder)
        else:
            messagebox.showinfo("Info", "Output folder does not exist yet.")

    def run(self):
        """Run the application."""
        # Initial refresh of video list
        self._refresh_video_list()
        
        # Start the main loop
        self.root.mainloop()


def create_app() -> VideoPreviewWindow:
    """Create and return the application instance."""
    root = tk.Tk()
    app = VideoPreviewWindow(root)
    return app
