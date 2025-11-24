"""
Interactive Daheng capture script with live preview, frame saving, and video capture.
"""
import gxipy as gx
import cv2
import numpy as np
import os
import time

# --- USER CONFIGURATION ---

# 1. Color Mode
# Note: This interactive version is designed for color capture.
IS_COLOR = True

# 2. Camera Parameters
EXPOSURE_TIME = 40_000           # Exposure time in microseconds (e.g., 20000 -> 20ms)
GAIN = 5.0                      # Gain in dB (e.g., 10.0)
FRAME_RATE = 125                # Target frames per second (e.g., 20)
RESOLUTION_W = 2048             # Frame width in pixels
RESOLUTION_H = 1536             # Frame height in pixels

# 3. Saving Options
SAVE_PATH = 'output'            # Directory to save images/videos

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
