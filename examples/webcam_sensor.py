"""Webcam sensor node example.

This example demonstrates:
1. A sensor node that captures frames from a webcam using OpenCV
2. Using the built-in ImageCodec for efficiently serializing image frames
3. Using Timer for controlled frame rates
4. Recording and playing back webcam data

Run with: uv run --extra opencv python -m examples.webcam_sensor

Example usage:
    # Record 5 seconds of webcam footage
    python -m examples.webcam_sensor record

    # Playback recorded footage (realtime)
    python -m examples.webcam_sensor playback

    # Playback at 2x speed
    python -m examples.webcam_sensor playback --speed 2.0
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Annotated

from tinman import In, Out, run, playback
from tinman.codecs import Image, ImageCodec
from tinman.timer import Timer


# =============================================================================
# Sensor Node
# =============================================================================


class WebcamSensor:
    """Webcam capture node.

    Captures frames from a webcam at a specified frame rate and publishes
    them to the "camera" channel.

    Example:
        >>> sensor = WebcamSensor(device_id=0, fps=30.0, duration=5.0)
        >>> await run([sensor.run, frame_processor])
    """

    def __init__(
        self,
        device_id: int = 0,
        fps: float = 30.0,
        duration: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """Initialize webcam sensor.

        Args:
            device_id: Camera device index (0 for default camera)
            fps: Target frames per second
            duration: How long to capture (None = indefinitely)
            width: Desired frame width (None = camera default)
            height: Desired frame height (None = camera default)
        """
        self.device_id = device_id
        self.fps = fps
        self.duration = duration
        self.width = width
        self.height = height

    async def run(
        self,
        output: Annotated[Out[Image], "camera", ImageCodec()],
        timer: Timer,
    ) -> None:
        """Capture and publish webcam frames.

        Uses Timer.periodic() for consistent frame timing that works
        correctly in both real-time and playback modes.
        """
        # Import cv2 here to make it optional
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for webcam capture. "
                "Install with: pip install opencv-python"
            )

        cap = cv2.VideoCapture(self.device_id)

        # Configure camera resolution if specified
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device_id}")

        # Get event loop for running blocking calls in executor
        loop = asyncio.get_running_loop()

        try:
            seq = 0
            start_time = timer.time_ns()
            end_time = (
                start_time + int(self.duration * 1_000_000_000)
                if self.duration
                else None
            )

            print(f"📷 Webcam started (device={self.device_id}, fps={self.fps})")

            # Use periodic timer for consistent frame rate
            async for tick_time in timer.periodic(1.0 / self.fps):
                # Check duration limit
                if end_time and tick_time >= end_time:
                    break

                # Capture frame - offload blocking I/O to thread pool
                ret, image = await loop.run_in_executor(None, cap.read)
                if not ret:
                    print("⚠️  Failed to capture frame, retrying...")
                    continue

                frame = Image(
                    timestamp_ns=tick_time,
                    seq=seq,
                    data=image,
                )

                await output.publish(frame)
                seq += 1

                if seq % int(self.fps) == 0:
                    elapsed = (tick_time - start_time) / 1_000_000_000
                    print(f"📷 Captured {seq} frames ({elapsed:.1f}s)")

            print(f"📷 Webcam stopped ({seq} frames total)")

        finally:
            cap.release()


# =============================================================================
# Processing Nodes
# =============================================================================


async def frame_display(
    input: Annotated[In[Image], "camera"],
) -> None:
    """Display frames in a window using pyglet (faster than OpenCV).
    
    Uses pyglet with vsync=False for maximum frame rate during playback.
    """
    try:
        import pyglet
        from pyglet import gl
    except ImportError:
        raise ImportError(
            "Pyglet is required for display. "
            "Install with: pip install pyglet"
        )

    print("🖥️  Display started")
    frame_count = 0
    loop = asyncio.get_running_loop()
    
    # Will create window sized to first frame
    window = None

    def prepare_frame(image):
        """Convert BGR numpy array to RGB bytes (runs in thread pool)."""
        # OpenCV uses BGR, pyglet needs RGB, and flip vertically for OpenGL
        rgb = image[::-1, :, ::-1].copy()
        return rgb.tobytes(), image.shape[1], image.shape[0]

    async for frame in input:
        # Offload color conversion to thread pool
        rgb_bytes, width, height = await loop.run_in_executor(
            None, prepare_frame, frame.data
        )
        
        # Create window sized to first frame (accounting for HiDPI)
        if window is None:
            # Get pixel ratio from a temporary window
            temp = pyglet.window.Window(visible=False)
            pixel_ratio = temp.get_pixel_ratio()
            temp.close()
            
            # Create window at frame size in screen points
            win_width = int(width / pixel_ratio)
            win_height = int(height / pixel_ratio)
            window = pyglet.window.Window(win_width, win_height, vsync=False, caption="Webcam")
        
        # Create texture and draw
        image_data = pyglet.image.ImageData(width, height, 'RGB', rgb_bytes)
        texture = image_data.get_texture()
        
        # Draw scaled to window size
        window.clear()
        
        # Scale texture to fill window (accounts for HiDPI)
        texture.blit(0, 0, width=window.width, height=window.height)
        
        window.flip()
        window.dispatch_events()
        
        frame_count += 1

    if window:
        window.close()
    print(f"🖥️  Display stopped ({frame_count} frames shown)")


async def frame_stats(
    input: Annotated[In[Image], "camera"],
) -> None:
    """Print frame statistics without displaying."""
    print("📊 Stats collector started")
    frame_count = 0
    total_pixels = 0
    first_time: int | None = None
    last_time: int = 0

    async for frame in input:
        if first_time is None:
            first_time = frame.timestamp_ns
        last_time = frame.timestamp_ns

        frame_count += 1
        total_pixels += frame.data.shape[0] * frame.data.shape[1]

        if frame_count % 30 == 0:
            elapsed = (last_time - first_time) / 1_000_000_000 if first_time else 0
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(
                f"📊 Image {frame.seq}: "
                f"{frame.data.shape[1]}x{frame.data.shape[0]}, "
                f"actual FPS: {actual_fps:.1f}"
            )

    if first_time is not None and last_time > first_time:
        elapsed = (last_time - first_time) / 1_000_000_000
        avg_fps = frame_count / elapsed
        print(f"📊 Summary: {frame_count} frames, {elapsed:.1f}s, avg {avg_fps:.1f} FPS")


# =============================================================================
# Main
# =============================================================================


async def record_webcam(log_dir: Path, duration: float, fps: float) -> None:
    """Record webcam footage to log files."""
    sensor = WebcamSensor(fps=fps, duration=duration)
    await run([sensor.run, frame_stats], log_dir=log_dir)
    print(f"\n✅ Recording saved to {log_dir}")


async def playback_webcam(log_dir: Path, speed: float, show_display: bool) -> None:
    """Playback recorded webcam footage."""
    nodes = [frame_stats]
    if show_display:
        nodes.append(frame_display)

    await playback(nodes, playback_dir=log_dir, speed=speed)


def main():
    parser = argparse.ArgumentParser(description="Webcam sensor example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record webcam footage")
    record_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("webcam_logs"),
        help="Directory to save logs",
    )
    record_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds",
    )
    record_parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target frame rate",
    )

    # Playback command
    playback_parser = subparsers.add_parser("playback", help="Playback recorded footage")
    playback_parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("webcam_logs"),
        help="Directory containing logs",
    )
    playback_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = realtime, inf = fast-forward)",
    )
    playback_parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show video display",
    )

    # Demo command (record + playback)
    demo_parser = subparsers.add_parser("demo", help="Record then playback")
    demo_parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Recording duration in seconds",
    )

    args = parser.parse_args()

    if args.command == "record":
        asyncio.run(record_webcam(args.log_dir, args.duration, args.fps))

    elif args.command == "playback":
        speed = float("inf") if args.speed == float("inf") else args.speed
        asyncio.run(playback_webcam(args.log_dir, speed, not args.no_display))

    elif args.command == "demo":
        from tempfile import TemporaryDirectory

        async def demo():
            with TemporaryDirectory() as tmpdir:
                log_dir = Path(tmpdir)
                print("=" * 60)
                print("Recording webcam...")
                print("=" * 60)
                await record_webcam(log_dir, args.duration, fps=15.0)

                print("\n" + "=" * 60)
                print("Playing back at 2x speed...")
                print("=" * 60)
                await playback_webcam(log_dir, speed=2.0, show_display=True)

        asyncio.run(demo())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
