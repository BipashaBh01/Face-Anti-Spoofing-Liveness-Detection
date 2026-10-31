# ============================================================================
#  CHL_capture.py — saves recording frames and assembles them into a video
# ============================================================================
#  During recording, each frame is saved to a temporary subfolder as a
#  numbered image. When recording finishes, the numbered frames are joined
#  into a single MP4 video file.
#
#  Saving numbered frames first, then assembling, keeps the process simple
#  to debug: every captured frame can be opened and inspected in the
#  folder, and resuming later only needs to add higher-numbered frames.
# ============================================================================

from __future__ import annotations

# Path handles folder and file locations.
from pathlib import Path

# shutil removes a whole folder and its contents in one call.
import shutil

# OpenCV writes the image frames and assembles the video.
import cv2

# NumPy holds each frame as an array.
import numpy as np


# ============================================================================
#  CAPTURE SESSION
# ============================================================================

class CaptureSession:
    """Holds the frames for one recording and writes the final video."""

    def __init__(self, capture_dir: Path, session_id: str, fps: float = 8.0):
        """Prepare a fresh folder for this recording's frames."""

        # The main folder that holds all captures.
        self.capture_dir = Path(capture_dir)

        # A unique name for this one recording's subfolder.
        self.session_id = session_id

        # Frames per second used when building the final video.
        self.fps = fps

        # The subfolder for this recording's numbered frames.
        self.frames_dir = self.capture_dir / session_id / "frames"

        # Counts how many frames have been saved so far.
        self.frame_count = 0

        # Create the frames folder, including any missing parent folders.
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        """Save one frame as the next numbered image in the folder."""

        # Move to the next frame number.
        self.frame_count += 1

        # Build the file name, zero-padded so files sort in order.
        # Example: frame_000001.jpg
        file_name = f"frame_{self.frame_count:06d}.jpg"

        # Full path for this frame image.
        file_path = self.frames_dir / file_name

        # Write the frame to disk as a JPEG image.
        cv2.imwrite(str(file_path), frame_bgr)

    # ------------------------------------------------------------------------

    def assemble_video(self) -> Path | None:
        """Join all saved frames into one MP4 video and return its path."""

        # Collect every saved frame file, sorted by name so order is kept.
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))

        # Stop early if no frames were saved.
        if not frame_files:
            return None

        # Read the first frame to learn the video width and height.
        first = cv2.imread(str(frame_files[0]))

        # Stop if that frame failed to load.
        if first is None:
            return None

        # Take the height and width from the first frame.
        height, width = first.shape[:2]

        # Path for the finished video, next to the frames folder.
        video_path = self.capture_dir / self.session_id / "recording.mp4"

        # Choose the MP4 video format.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Open the video writer with the size and frame rate.
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

        # Add every frame to the video in order.
        for file_path in frame_files:

            # Load the frame image.
            frame = cv2.imread(str(file_path))

            # Skip any frame that failed to load.
            if frame is None:
                continue

            # Write the frame into the video.
            writer.write(frame)

        # Close the video file so it is saved properly.
        writer.release()

        # Return the path of the finished video.
        return video_path

    # ------------------------------------------------------------------------

    def delete(self) -> None:
        """Remove this recording's whole folder, frames and video."""

        # The top folder for this recording.
        session_folder = self.capture_dir / self.session_id

        # Delete it and everything inside, ignoring errors if already gone.
        shutil.rmtree(session_folder, ignore_errors=True)