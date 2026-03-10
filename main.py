"""
Interactive Daheng capture script with live preview, frame saving, and video capture.
"""
import gxipy as gx
import cv2
import numpy as np
import os
import time

# --- USER CONFIGURATION ---

# 2. Camera Parameters
EXPOSURE_TIME = 80_000           # Exposure time in microseconds (e.g., 20_000 -> 20ms)
GAIN = 1.0                      # Gain in dB (e.g., 10.0)
FRAME_RATE = 30                # Target frames per second (e.g., 20)
RESOLUTION_W = 2048             # Frame width in pixels
RESOLUTION_H = 1536             # Frame height in pixels

# 3. Saving Options
SAVE_PATH = 'output'            # Directory to save images/videos

# 4. Live Sharpness Indicator
SHARPNESS_ENABLED = True        # Toggle sharpness indicator on live preview
SHARPNESS_DOWNSCALE = 0.5       # Scale factor for sharpness calc (0 < value <= 1)
SHARPNESS_EVERY_N_FRAMES = 1    # Compute sharpness every N frames
SHARPNESS_EMA_ALPHA = 0.2       # Smoothing for displayed sharpness (0-1)

# --- END OF CONFIGURATION ---


def process_color_frame(raw_image):
    """
    Converts a raw gxipy image to a displayable BGR color format for OpenCV.
    """
    numpy_image = raw_image.get_numpy_array()
    if numpy_image is None:
        return None

    # Debayer the raw image to BGR. Use COLOR_BAYER_BG2BGR if colors are swapped.
    color_image = cv2.cvtColor(numpy_image, cv2.COLOR_BAYER_BG2BGR)
    return color_image

def compute_sharpness_value(frame, downscale):
    """
    Returns a focus/clarity metric using the variance of the Laplacian.
    Higher values generally indicate a sharper image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if downscale < 1.0:
        new_w = max(1, int(gray.shape[1] * downscale))
        new_h = max(1, int(gray.shape[0] * downscale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def main():
    """
    Main function to run the interactive camera viewer and controller.
    """
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"✅ Output will be saved to '{SAVE_PATH}/'")

    # Discover and connect to the first Daheng camera detected on the bus.
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("❌ Error: No Daheng camera found.")
        return

    cam = device_manager.open_device_by_index(1)
    if cam is None:
        print("❌ Error: Could not open camera.")
        return
        
    print(f"✅ Successfully opened camera: {cam.DeviceModelName.get()} ({cam.DeviceSerialNumber.get()})")

    try:
        # --- Configure Camera ---
        print("\n⚙️ Configuring camera...")
        cam.PixelFormat.set(gx.GxPixelFormatEntry.BAYER_RG8)
        print(f"- Mode: Color (Bayer RG8)")
        cam.Width.set(RESOLUTION_W)
        cam.Height.set(RESOLUTION_H)
        cam.OffsetX.set(0)
        cam.OffsetY.set(0)
        print(f"- Resolution: {cam.Width.get()}x{cam.Height.get()}")
        cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
        cam.GainAuto.set(gx.GxAutoEntry.OFF)
        cam.ExposureTime.set(EXPOSURE_TIME)
        cam.Gain.set(GAIN)
        print(f"- Exposure: {cam.ExposureTime.get()} us")
        print(f"- Gain: {cam.Gain.get()} dB")
        cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
        cam.AcquisitionFrameRate.set(FRAME_RATE)
        print(f"- Frame Rate: {cam.AcquisitionFrameRate.get()} fps")
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
        
        # --- Start Streaming ---
        cam.stream_on()
        print("\n🚀 Camera stream started.")

        # --- Interactive Loop ---
        print("\n--- Controls ---")
        print("  's' - Save a single frame")
        print("  'r' - Start/Stop video recording")
        print("  'q' - Quit")
        print("----------------")
        
        is_recording = False
        video_writer = None
        frame_index = 0
        sharpness_ema = None

        while True:
            raw_image = cam.data_stream[0].get_image(timeout=100)
            if raw_image and raw_image.get_status() == gx.GxFrameStatusList.SUCCESS:
                frame = process_color_frame(raw_image)
                if frame is None:
                    continue

                # Work on a copy so the saved/recorded frame stays untouched.
                display_frame = frame.copy() # Create a copy for drawing text

                # Add recording status text to the display frame
                if is_recording:
                    cv2.putText(display_frame, 'RECORDING', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Add sharpness indicator to the display frame
                if SHARPNESS_ENABLED and frame_index % SHARPNESS_EVERY_N_FRAMES == 0:
                    sharpness_now = compute_sharpness_value(frame, SHARPNESS_DOWNSCALE)
                    if sharpness_ema is None:
                        sharpness_ema = sharpness_now
                    else:
                        sharpness_ema = (
                            SHARPNESS_EMA_ALPHA * sharpness_now
                            + (1 - SHARPNESS_EMA_ALPHA) * sharpness_ema
                        )
                if SHARPNESS_ENABLED and sharpness_ema is not None:
                    cv2.putText(
                        display_frame,
                        f"Sharpness: {sharpness_ema:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                # Show the live preview window.
                cv2.imshow('Interactive Camera Control', display_frame)

                # --- Handle Keyboard Input ---
                # A 1ms wait lets OpenCV process key presses without blocking acquisition.
                key = cv2.waitKey(1) & 0xFF

                # Quit Button ('q')
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
                # Save Frame Button ('s')
                elif key == ord('s'):
                    filename = f"frame_{int(time.time())}.png"
                    save_full_path = os.path.join(SAVE_PATH, filename)
                    cv2.imwrite(save_full_path, frame)
                    print(f"📸 Frame saved to: {save_full_path}")

                # Record Video Button ('r')
                elif key == ord('r'):
                    # If not currently recording, start a new recording
                    if not is_recording:
                        is_recording = True
                        filename = f"video_{int(time.time())}.avi"
                        save_full_path = os.path.join(SAVE_PATH, filename)
                        fourcc = cv2.VideoWriter.fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(save_full_path, fourcc, FRAME_RATE, (RESOLUTION_W, RESOLUTION_H))
                        print(f"🔴 Started recording to: {save_full_path}")
                    # If already recording, stop it
                    else:
                        is_recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                            print("⏹️ Stopped recording.")
                
                # Write frame to video if recording is enabled
                if is_recording and video_writer:
                    video_writer.write(frame)

                frame_index += 1

        # Final check to release video writer if the loop was exited while recording
        if video_writer:
            video_writer.release()
            print("⏹️ Stopped recording.")
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- Cleanup ---
        print("\nCleaning up and closing camera...")
        cam.stream_off()
        cam.close_device()
        print("Done.")


if __name__ == "__main__":
    main()
