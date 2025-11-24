"""
Batch capture helper that records bracketed frames and creates an HDR output.
"""
import os
import cv2
import numpy as np
import gxipy as gx  # <- gxipy import
import time  # Added for timing the script
import datetime  # Import datetime for timestamping

# Suppress OpenCV warnings by setting the environment variable
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

def capture_images_with_exposures(device: gx.Device, exposure_times, output_folder: str):
    """
    Captures images with the specified exposure times and saves them to the output folder.

    Args:
        device (gx.Device): The connected camera device.
        exposure_times (list of int): List of exposure times in microseconds.
        output_folder (str): Path to the folder where images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Generate a timestamp for unique naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Capture one frame per requested exposure value.
        for i, exposure in enumerate(exposure_times):
            device.ExposureTime.set(float(exposure))
            # Start acquisition stream
            device.stream_on()
            print(f"Capturing image with exposure time: {exposure} µs")

            # Optionally: grab and discard one frame to let exposure settle
            _ = device.data_stream[0].get_image()

            raw_image = device.data_stream[0].get_image()
            if raw_image is None:
                print("Failed to capture image.")
                continue

            rgb_image = raw_image.convert("RGB")
            image_data = rgb_image.get_numpy_array()
            # Update the image path to include the timestamp
            image_path = os.path.join(output_folder, f"image_{timestamp}_{i+1}_{exposure}.png")
            cv2.imwrite(image_path, image_data)
            print(f"Saved image: {image_path}")
            device.stream_off()
    finally:
        # Make sure streaming ends even if an exception occurs.
        device.stream_off()


def combine_images_to_hdr(image_folder: str, output_path: str):
    """
    Combines images in the specified folder into an HDR image and saves it.

    Args:
        image_folder (str): Path to the folder containing images.
        output_path (str): Path to save the HDR image.
    """
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith('.png')
    ]
    if not image_files:
        print("No PNG images found to combine.")
        return

    # Filter out files that do not match the expected naming convention
    valid_image_files = []
    for f in image_files:
        try:
            # Attempt to extract the exposure time from the filename
            int(os.path.basename(f).split('_')[-1].split('.')[0])
            valid_image_files.append(f)
        except ValueError:
            print(f"Skipping invalid file: {f}")

    if not valid_image_files:
        print("No valid PNG images found to combine.")
        return

    valid_image_files.sort()  # assumes naming: image_1_..., image_2_..., etc.

    images = [cv2.imread(f) for f in valid_image_files]

    # Extract exposure times (µs) from filenames and convert to seconds
    exposure_times_us = np.array(
        [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in valid_image_files],
        dtype=np.float32
    )
    exposure_times = exposure_times_us / 1e6  # µs → seconds

    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=exposure_times)  #type: ignore
 
    # Tonemap (Drago)
    tonemap = cv2.createTonemapDrago(1.0, 0.7)
    ldr = tonemap.process(hdr)
    
    # ---------------------------------------------------------------------------- #
    ldr = np.nan_to_num(ldr, nan=0.0, posinf=255, neginf=0)
    ldr = np.clip(ldr * 255, 0, 255).astype('uint8')

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, ldr)
    print(f"HDR image saved to: {output_path}")

def combine_images_exposure_fusion(image_folder, output_path):
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith('.png')
    ]
    image_files.sort()
    images = [cv2.imread(f) for f in image_files]

    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)  #type: ignore

    fusion_8bit = np.clip(fusion * 255, 0, 255).astype('uint8')

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, fusion_8bit)
    print(f"Exposure fusion image saved to: {output_path}")

def main():
    start_time = time.time()  # Start timing

    # Example exposure times in microseconds
    exposure_times = [40_000,500_000, 1_000_000]  # 10ms, 40ms, 160ms, 640ms, 1s
    output_folder = "output_images"
    # Generate a timestamp for unique naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Update the HDR output path to include the timestamp
    hdr_output_path = os.path.join(output_folder, f"hdr_image_{timestamp}.png")  # Save HDR image to output_images

    device_manager = gx.DeviceManager()
    dev_num, _ = device_manager.update_device_list()
    if dev_num == 0:
        print("No device found. Please check the connection and try again.")
        return

    # Most gx examples use 1-based index, adjust if needed
    device = device_manager.open_device_by_index(1)

    try:
        capture_images_with_exposures(device, exposure_times, output_folder)  # type: ignore
        combine_images_to_hdr(output_folder, hdr_output_path)
        combine_images_exposure_fusion(output_folder, os.path.join(output_folder, f"fusion_image_{timestamp}.png"))
    finally:
        if device is not None:
            device.close_device()

    end_time = time.time()  # End timing
    print(f"Total script duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
