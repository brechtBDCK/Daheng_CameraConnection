import cv2
import numpy as np
import glob
import os
import json

# --- CONFIGURATION ---
# Dimensions of internal corners (width, height)
CHECKERBOARD = (7, 10) 
# Size of one square in your defined unit (mm)
SQUARE_SIZE = 12.0 
# Image directory path
IMAGE_DIR = 'calibration_images/*.png' 

def calibrate_camera():
    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points based on real-world dimensions
    # Coordinates like (0,0,0), (1,0,0), (2,0,0) ...., (7,0,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(IMAGE_DIR)

    if not images:
        print("No images found. Please check your directory path.")
        return

    print(f"Found {len(images)} images. Processing...")

    found_count = 0
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Error reading image {fname}. Skipping...")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Flags help with adapting to lighting or finding the board in the image
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +cv2.CALIB_CB_FAST_CHECK) #type: ignore

        if ret == True:
            found_count += 1
            objpoints.append(objp)

            # Refine corner locations to sub-pixel accuracy (Critical for good calibration)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Optional: Draw and display the corners to verify
            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(100) # Wait 100ms
        else:
            print(f"Pattern not found in {fname}")

    cv2.destroyAllWindows()

    if found_count < 1:
        print("Could not detect corners in any images.")
        return

    print(f"Calibrating based on {found_count} valid images...")

    # --- CALIBRATION ---
    cameraMatrix = np.zeros((3, 3), dtype=np.float64)
    distCoeffs = np.zeros((5, 1), dtype=np.float64)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], cameraMatrix, distCoeffs)

    # --- OUTPUT ---
    print("\n--- Intrinsic Parameters ---")
    print(f"Reprojection Error: {ret:.4f} (Lower is better, ideally < 1.0)")

    print("\nCamera Matrix (K):")
    print(mtx)
    print("\nFormat:")
    print("[[fx,  0, cx],")
    print(" [ 0, fy, cy],")
    print(" [ 0,  0,  1]]")

    print("\nDistortion Coefficients (D):")
    print(dist)
    print("(k1, k2, p1, p2, k3)")

    # Extract fx, fy, cx, cy from the camera matrix
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    # Get image dimensions
    h, w = gray.shape[:2]

    # Flatten the distortion coefficients array
    dist = dist.flatten()

    # Save calibration data to JSON
    calibration_data = {
        "reprojection_error": ret,
        "camera_matrix": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        },
        "distortion_coefficients": {
            "k1": dist[0] if len(dist) > 0 else None,
            "k2": dist[1] if len(dist) > 1 else None,
            "p1": dist[2] if len(dist) > 2 else None,
            "p2": dist[3] if len(dist) > 3 else None,
            "k3": dist[4] if len(dist) > 4 else None
        },
        "image_width": w,
        "image_height": h
    }

    with open("camera_intrinsics_daheng.json", "w") as json_file:
        json.dump(calibration_data, json_file, indent=4)

    print("\nSaved calibration data to 'camera_intrinsics_daheng.json'.")

if __name__ == "__main__":
    calibrate_camera()
