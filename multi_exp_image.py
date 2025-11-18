import os
import cv2
import numpy as np
import gxipy as gx  # <- gxipy import
import time  # Added for timing the script

def capture_images_with_exposures(device: gx.Device, exposure_times, output_folder: str):
    """
    Captures images with the specified exposure times and saves them to the output folder.

    Args:
        device (gx.Device): The connected camera device.
        exposure_times (list of int): List of exposure times in microseconds.
        output_folder (str): Path to the folder where images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)



    try:
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
            image_path = os.path.join(output_folder, f"image_{i+1}_{exposure}.png")
            cv2.imwrite(image_path, image_data)
            print(f"Saved image: {image_path}")
            device.stream_off()
    finally:
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

    image_files.sort()  # assumes naming: image_1_..., image_2_..., etc.

    images = [cv2.imread(f) for f in image_files]

    # Extract exposure times (µs) from filenames and convert to seconds
    exposure_times_us = np.array(
        [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in image_files],
        dtype=np.float32
    )
    exposure_times = exposure_times_us / 1e6  # µs → seconds

    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=exposure_times)  #type: ignore
    
    # Tonemap (Drago)
    tonemap = cv2.createTonemapDrago(1.0, 0.7)
    ldr = tonemap.process(hdr)
    ldr = np.clip(ldr * 255, 0, 255).astype('uint8')

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, ldr)
    print(f"HDR image saved to: {output_path}")


def main():
    start_time = time.time()  # Start timing

    # Example exposure times in microseconds
    exposure_times = [10000, 40000, 160000, 640000, 1_000_000]  # 10ms, 40ms, 160ms, 640ms, 1s
    output_folder = "output_images"
    hdr_output_path = os.path.join(output_folder, "hdr_image.png")  # Save HDR image to output_images

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
    finally:
        if device is not None:
            device.close_device()

    end_time = time.time()  # End timing
    print(f"Total script duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
