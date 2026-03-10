"""
Single-image Daheng capture helper (no GUI).
"""
import os
import time

import cv2
import gxipy as gx


def _process_color_frame(raw_image) -> "cv2.typing.MatLike":
    """
    Convert a raw gxipy Bayer image into a BGR OpenCV image.
    """
    numpy_image = raw_image.get_numpy_array()
    if numpy_image is None:
        raise RuntimeError("Camera returned an empty frame array.")
    return cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_BG2BGR)


def take_single_image(
    save_dir: str = "output",
    filename: str | None = None,
    exposure_time_us: float = 80_000,
    gain_db: float = 1.0,
    width: int = 2048,
    height: int = 1536,
    timeout_ms: int = 1000,
    warmup_frames: int = 1,
) -> str:
    """
    Capture one image immediately, save it to disk, and return the saved path.
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"frame_{int(time.time())}.png"
    save_path = os.path.join(save_dir, filename)

    device_manager = gx.DeviceManager()
    dev_num, _ = device_manager.update_device_list()
    if dev_num == 0:
        raise RuntimeError("No Daheng camera found.")

    cam = device_manager.open_device_by_index(1)
    if cam is None:
        raise RuntimeError("Could not open Daheng camera.")

    try:
        cam.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG8)
        cam.Width.set(width)
        cam.Height.set(height)
        cam.OffsetX.set(0)
        cam.OffsetY.set(0)
        cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
        cam.GainAuto.set(gx.GxAutoEntry.OFF)
        cam.ExposureTime.set(float(exposure_time_us))
        cam.Gain.set(float(gain_db))
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        cam.stream_on()

        # Optional throwaway frames so exposure/gain settings settle before save.
        for _ in range(max(0, warmup_frames)):
            cam.data_stream[0].get_image(timeout=timeout_ms)

        raw_image = cam.data_stream[0].get_image(timeout=timeout_ms)
        if raw_image is None:
            raise RuntimeError("Timed out waiting for image from camera.")
        if raw_image.get_status() != gx.GxFrameStatusList.SUCCESS:
            raise RuntimeError("Camera returned a frame with non-success status.")

        frame = _process_color_frame(raw_image)
        if not cv2.imwrite(save_path, frame):
            raise RuntimeError(f"Failed to write image to: {save_path}")

        return save_path
    finally:
        try:
            cam.stream_off()
        except Exception:
            pass
        cam.close_device()


if __name__ == "__main__":
    output_path = take_single_image()
    print(f"Saved image: {output_path}")
