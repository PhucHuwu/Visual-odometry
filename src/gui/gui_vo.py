"""Visual Odometry GUI Application

GUI với real-time camera feed và trajectory visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time

from src.core.camera import CameraInput
from src.core.vo_pipeline import VOPipeline
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger()


class VO_GUI:
    """Visual Odometry GUI Application"""

    def __init__(self, root):
        """Initialize GUI"""
        self.root = root
        self.root.title("Visual Odometry - Real-time Tracking")
        self.root.geometry("1400x800")

        # State
        self.is_running = False
        self.vo_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.trajectory_data = []
        self.source_type = "camera"  # "camera" or "video"
        self.video_path = None

        # Config
        self.config = ConfigLoader('config/default_config.yaml')

        # Setup GUI
        self._setup_ui()

        # Bind cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_ui(self):
        """Setup UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()

        # Right panel - Trajectory plot
        plot_frame = ttk.LabelFrame(main_frame, text="Top View Trajectory", padding="10")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Matplotlib figure
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Camera Trajectory (Top View)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom panel - Controls and stats
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Stats labels
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0.0", font=('Arial', 10))
        self.fps_label.pack(side=tk.LEFT, padx=10)

        self.poses_label = ttk.Label(stats_frame, text="Poses: 0", font=('Arial', 10))
        self.poses_label.pack(side=tk.LEFT, padx=10)

        self.keypoints_label = ttk.Label(stats_frame, text="Keypoints: 0", font=('Arial', 10))
        self.keypoints_label.pack(side=tk.LEFT, padx=10)

        self.algo_status = ttk.Label(stats_frame, text="Algorithm: -", font=('Arial', 10))
        self.algo_status.pack(side=tk.LEFT, padx=10)

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)

        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_vo, width=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_vo, width=10, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Algorithm selection
        ttk.Label(button_frame, text="Algorithm:").pack(side=tk.LEFT, padx=(10, 5))
        self.algo_var = tk.StringVar(value="orb")
        algo_combo = ttk.Combobox(button_frame, textvariable=self.algo_var,
                                  values=["orb", "sift", "fast", "lucas_kanade"],
                                  state="readonly", width=12)
        algo_combo.pack(side=tk.LEFT, padx=5)

        # Source type selection frame
        source_frame = ttk.LabelFrame(control_frame, text="Video Source", padding="10")
        source_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)

        # Radio buttons for source type
        self.source_var = tk.StringVar(value="camera")
        ttk.Radiobutton(source_frame, text="Camera", variable=self.source_var,
                        value="camera", command=self._on_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Video File", variable=self.source_var,
                        value="video", command=self._on_source_change).pack(side=tk.LEFT, padx=5)

        # Camera ID selection
        self.camera_frame = ttk.Frame(source_frame)
        self.camera_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(self.camera_frame, text="ID:").pack(side=tk.LEFT, padx=(0, 5))
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(self.camera_frame, textvariable=self.camera_var,
                                    values=["0", "1"],
                                    state="readonly", width=5)
        camera_combo.pack(side=tk.LEFT)

        # Video file selection
        self.video_file_frame = ttk.Frame(source_frame)  # Renamed from self.video_frame to avoid conflict
        ttk.Button(self.video_file_frame, text="Browse...",
                   command=self._browse_video).pack(side=tk.LEFT, padx=5)
        self.video_file_path_label = ttk.Label(self.video_file_frame, text="No file selected",  # Renamed from self.video_label to avoid conflict
                                               foreground="gray")
        self.video_file_path_label.pack(side=tk.LEFT, padx=5)

        # Initially show camera options
        self._on_source_change()

    def _on_source_change(self):
        """Handle change in video source type (camera/video file)"""
        self.source_type = self.source_var.get()
        if self.source_type == "camera":
            self.camera_frame.pack(side=tk.LEFT, padx=10)
            self.video_file_frame.pack_forget()
        else:
            self.camera_frame.pack_forget()
            self.video_file_frame.pack(side=tk.LEFT, padx=10)

    def _browse_video(self):
        """Open file dialog to select a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.video_file_path_label.config(text=f".../{file_path.split('/')[-1]}", foreground="black")
        else:
            self.video_path = None
            self.video_file_path_label.config(text="No file selected", foreground="gray")

    def start_vo(self):
        """Start VO processing"""
        if self.is_running:
            return

        # Validate source
        if self.source_var.get() == "video" and not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first!")
            return

        self.is_running = True
        self.trajectory_data = []

        # Update buttons
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # Update algo status
        self.algo_status.config(text=f"Algorithm: {self.algo_var.get().upper()}")

        # Start VO thread
        self.vo_thread = threading.Thread(target=self._run_vo, daemon=True)
        self.vo_thread.start()

        # Start GUI update loop
        self._update_gui()

    def stop_vo(self):
        """Stop VO processing"""
        self.is_running = False

        # Update buttons
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        logger.info("VO stopped by user")

    def _run_vo(self):
        """Run VO pipeline in background thread"""
        try:
            # Update config
            self.config.config['algorithm']['type'] = self.algo_var.get()

            # Initialize camera or video
            if self.source_var.get() == "camera":
                camera_id = int(self.camera_var.get())
                camera = CameraInput(source=camera_id)
            else:
                camera = CameraInput(source=self.video_path)

            # Initialize VO pipeline
            vo = VOPipeline(self.config)

            frame_count = 0
            start_time = time.time()

            while self.is_running:
                ret, frame = camera.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    break

                # Process frame
                vo.process_frame(frame)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 0.001)

                # Get stats
                stats = vo.get_stats()
                stats['fps'] = fps
                stats['keyframes'] = vo.keyframe_count

                # Queue frame and stats for GUI update
                try:
                    self.frame_queue.put_nowait({
                        'frame': frame,
                        'keypoints': vo.prev_keypoints if vo.prev_keypoints else [],
                        'trajectory': vo.get_trajectory(),
                        'stats': stats
                    })
                except queue.Full:
                    pass  # Skip if queue full

            # Cleanup
            camera.release()

            # Save trajectory
            if vo.get_trajectory().shape[0] > 0:
                vo.save_trajectory('output/trajectory_gui.txt')
                logger.info(f"Trajectory saved: {vo.get_trajectory().shape[0]} poses")

        except Exception as e:
            logger.error(f"VO thread error: {e}")
            self.is_running = False
            self.root.after(0, lambda: messagebox.showerror("Error", f"VO Error: {e}"))

    def _update_gui(self):
        """Update GUI with latest data"""
        if not self.is_running:
            return

        try:
            # Get latest data from queue
            data = self.frame_queue.get_nowait()

            # Update video frame
            frame = data['frame']
            keypoints = data['keypoints']

            # Draw keypoints
            frame_display = frame.copy()
            for kp in keypoints[:50]:  # Limit to 50 for performance
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame_display, (x, y), 3, (0, 255, 0), -1)

            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)

            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep reference

            # Update trajectory plot
            trajectory = data['trajectory']
            if len(trajectory) > 1:
                self.ax.clear()
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
                self.ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
                self.ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='Current')
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                self.ax.set_title('Camera Trajectory (Top View)')
                self.ax.grid(True, alpha=0.3)
                self.ax.legend()
                self.ax.set_aspect('equal')
                self.canvas.draw()

            # Update stats
            stats = data['stats']
            self.fps_label.config(text=f"FPS: {stats['fps']:.1f}")
            self.poses_label.config(text=f"Poses: {stats['trajectory_length']}")
            self.keypoints_label.config(text=f"Keypoints: {len(keypoints)}")

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(30, self._update_gui)  # ~33 FPS GUI update

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.vo_thread and self.vo_thread.is_alive():
            self.vo_thread.join(timeout=2)
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = VO_GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
