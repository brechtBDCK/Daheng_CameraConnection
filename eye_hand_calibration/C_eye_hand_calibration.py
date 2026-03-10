import os
import glob
import json
import cv2
import numpy as np


# ============================================================
# USER SETTINGS - EDIT THESE
# ============================================================

IMAGES_DIR = "eye_hand_calibration/samples/images"
POSES_DIR = "eye_hand_calibration/samples/robot_poses"
INTRINSICS_JSON = "camera_intrinsics_daheng_6mmLens.json"

# ChArUco board settings: MUST match your real printed board exactly
SQUARES_X = 4                 # number of chessboard squares in X
SQUARES_Y = 4                 # number of chessboard squares in Y
SQUARE_LENGTH = 0.033         # meters
MARKER_LENGTH = 0.024         # meters
ARUCO_DICT_NAME = "DICT_4X4_50"   # example only - change if needed

# If your board was generated with older OpenCV (< 4.6) and even row counts,
# you may need legacy pattern compatibility.
USE_LEGACY_CHARUCO_PATTERN = True

# If your .npz already stores ^bT_tool0 (base <- tool0), leave True.
# If it stores ^tool0T_b instead, set False and the script will invert it.
ROBOT_POSE_IS_BASE_T_TOOL = True

# Robot translations from FANUC are usually in mm. ChArUco solvePnP uses meters,
# so convert robot translation to meters by default.
ROBOT_TRANSLATION_SCALE = 1e-3

# If no filename-based image<->pose match is found, pair by sorted index as fallback.
ALLOW_INDEX_PAIRING_FALLBACK = True

# How many detected ChArUco corners required before solving PnP
MIN_CHARUCO_CORNERS = 6

# Extra debug reporting and outlier list length.
PRINT_DIAGNOSTICS = True
TOP_K_OUTLIERS = 5



# ============================================================
# BASIC TRANSFORM HELPERS
# ============================================================

def T_from_R_t(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def rotation_angle_deg(R):
    val = (np.trace(R) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))

def scalar_stats(values):
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "count": 0,
            "min": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }

def transform_motion_stats(T_list):
    trans = []
    rots = []
    n = len(T_list)
    for i in range(n):
        Ti_inv = invert_T(T_list[i])
        for j in range(i + 1, n):
            dT = Ti_inv @ T_list[j]
            trans.append(np.linalg.norm(dT[:3, 3]))
            rots.append(rotation_angle_deg(dT[:3, :3]))
    return scalar_stats(trans), scalar_stats(rots)

def rotation_quality(R):
    I = np.eye(3, dtype=np.float64)
    ortho_err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
    det = float(np.linalg.det(R))
    return det, ortho_err


# ============================================================
# INTRINSICS LOADING
# ============================================================

def _reshape_camera_matrix(v):
    if isinstance(v, dict):
        # Format written by calibration_checkerboard.py
        if all(k in v for k in ("fx", "fy", "cx", "cy")):
            fx = float(v["fx"])
            fy = float(v["fy"])
            cx = float(v["cx"])
            cy = float(v["cy"])
            return np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        if "data" in v:
            return _reshape_camera_matrix(v["data"])

    arr = np.array(v, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    arr = arr.reshape(-1)
    if arr.size == 9:
        return arr.reshape(3, 3)
    raise ValueError(f"Could not reshape camera matrix from shape {arr.shape}")

def _reshape_dist(v):
    if isinstance(v, dict):
        # Format written by calibration_checkerboard.py
        ordered = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        vals = [v[k] for k in ordered if k in v and v[k] is not None]
        if vals:
            return np.array(vals, dtype=np.float64)
        if "data" in v:
            return _reshape_dist(v["data"])

    arr = np.array(v, dtype=np.float64).reshape(-1)
    if arr.size in (4, 5, 8, 12, 14):
        return arr
    raise ValueError(f"Unexpected distortion coefficient size: {arr.size}")

def load_intrinsics_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Common direct formats
    if "K" in data:
        K = _reshape_camera_matrix(data["K"])
    elif "camera_matrix" in data:
        cm = data["camera_matrix"]
        if isinstance(cm, dict) and "data" in cm:
            K = _reshape_camera_matrix(cm["data"])
        else:
            K = _reshape_camera_matrix(cm)
    else:
        raise KeyError("Could not find camera matrix in JSON. Expected 'K' or 'camera_matrix'.")

    if "dist" in data:
        dist = _reshape_dist(data["dist"])
    elif "dist_coeff" in data:
        dist = _reshape_dist(data["dist_coeff"])
    elif "distortion_coefficients" in data:
        dc = data["distortion_coefficients"]
        if isinstance(dc, dict) and "data" in dc:
            dist = _reshape_dist(dc["data"])
        else:
            dist = _reshape_dist(dc)
    else:
        raise KeyError("Could not find distortion coefficients in JSON.")

    return K, dist


# ============================================================
# ROBOT POSE LOADING
# ============================================================

def load_pose_npz(path, pose_is_base_t_tool):
    data = np.load(path)

    candidate_keys = [
        "T_base_tool0",
        "T_base_tool",
        "T_base_ee",
        "T",
        "pose",
        "transform",
    ]

    key = None
    for k in candidate_keys:
        if k in data:
            key = k
            break

    if key is None:
        # If there is only one array in the npz, use that
        files = list(data.files)
        if len(files) == 1:
            key = files[0]
        else:
            raise KeyError(
                f"Could not determine pose key in {path}. "
                f"Available keys: {list(data.files)}"
            )

    T = np.array(data[key], dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(f"{path} key '{key}' is not 4x4, got {T.shape}")

    if not pose_is_base_t_tool:
        T = invert_T(T)

    T = T.copy()
    T[:3, 3] *= float(ROBOT_TRANSLATION_SCALE)

    return T, key


# ============================================================
# CHARUCO BOARD SETUP
# ============================================================

def get_aruco_dictionary(dict_name):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary name: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

def create_charuco_board():
    dictionary = get_aruco_dictionary(ARUCO_DICT_NAME)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )

    # Compatibility for some boards generated with older OpenCV
    if USE_LEGACY_CHARUCO_PATTERN and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    return board


# ============================================================
# TARGET POSE DETECTION: ^cT_t
# ============================================================

def detect_target_pose_charuco(image_path, K, dist, board):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV build has no cv2.aruco module.")

    if not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("Your OpenCV build has no cv2.aruco.CharucoDetector.")

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None, {
            "reason": f"too_few_charuco_corners ({0 if charuco_ids is None else len(charuco_ids)})"
        }

    # solvePnP fails if the detected ChArUco corners are collinear
    if hasattr(board, "checkCharucoCornersCollinear"):
        if board.checkCharucoCornersCollinear(charuco_ids):
            return None, {"reason": "charuco_corners_collinear"}

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    if obj_points is None or img_points is None or len(obj_points) < MIN_CHARUCO_CORNERS:
        return None, {"reason": "not_enough_matched_points"}

    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not ok:
        return None, {"reason": "solvePnP_failed"}

    R, _ = cv2.Rodrigues(rvec)
    T_cam_target = T_from_R_t(R, tvec.reshape(3))

    # Optional reprojection error
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    img_points_2d = img_points.reshape(-1, 2)
    reproj_err = np.sqrt(np.mean(np.sum((projected - img_points_2d) ** 2, axis=1)))

    info = {
        "num_charuco": int(len(charuco_ids)),
        "num_matched": int(len(obj_points)),
        "reproj_rmse_px": float(reproj_err),
    }
    return T_cam_target, info


# ============================================================
# SAMPLE LOADING
# ============================================================

def find_pose_file_for_image(stem, poses_dir):
    candidates = [
        os.path.join(poses_dir, stem + ".npz"),
        os.path.join(poses_dir, stem + "_robot_pose.npz"),
        os.path.join(poses_dir, stem.replace("image", "pose") + ".npz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def list_image_files(images_dir):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)

def list_pose_files(poses_dir):
    return sorted(glob.glob(os.path.join(poses_dir, "*.npz")))

def pair_images_and_poses(image_files, poses_dir):
    pairs = []
    unmatched_images = []

    all_pose_files = list_pose_files(poses_dir)
    used_pose_files = set()

    # First pass: explicit filename-based matching.
    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        pose_path = find_pose_file_for_image(stem, poses_dir)
        if pose_path is None:
            unmatched_images.append(img_path)
            continue
        pairs.append((img_path, pose_path))
        used_pose_files.add(pose_path)

    # Fallback: if filenames don't align, pair remaining images/poses by sorted order.
    if unmatched_images and ALLOW_INDEX_PAIRING_FALLBACK:
        remaining_pose_files = [p for p in all_pose_files if p not in used_pose_files]
        if len(remaining_pose_files) == len(unmatched_images):
            for img_path, pose_path in zip(unmatched_images, remaining_pose_files):
                pairs.append((img_path, pose_path))
            unmatched_images = []

    return pairs, unmatched_images

def load_samples(images_dir, poses_dir, K, dist, board, pose_is_base_t_tool):
    image_files = list_image_files(images_dir)
    image_pose_pairs, unmatched_images = pair_images_and_poses(image_files, poses_dir)

    T_base_tool_list = []
    T_cam_target_list = []
    accepted = []
    rejected = []

    for img_path in unmatched_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        rejected.append((stem, "missing_pose_file"))

    for img_path, pose_path in image_pose_pairs:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        try:
            T_base_tool, pose_key = load_pose_npz(pose_path, pose_is_base_t_tool)
        except Exception as e:
            rejected.append((stem, f"pose_load_error: {e}"))
            continue

        try:
            T_cam_target, info = detect_target_pose_charuco(img_path, K, dist, board)
        except Exception as e:
            rejected.append((stem, f"charuco_error: {e}"))
            continue

        if T_cam_target is None:
            rejected.append((stem, info["reason"])) #type: ignore
            continue

        T_base_tool_list.append(T_base_tool)
        T_cam_target_list.append(T_cam_target)
        accepted.append((stem, os.path.basename(pose_path), pose_key, info))

    return T_base_tool_list, T_cam_target_list, accepted, rejected


# ============================================================
# HAND-EYE
# ============================================================

def solve_handeye(T_base_tool_list, T_cam_target_list, method):
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for T_bg, T_ct in zip(T_base_tool_list, T_cam_target_list):
        # OpenCV expects:
        # R_gripper2base, t_gripper2base == ^bT_g
        # R_target2cam,   t_target2cam   == ^cT_t
        R_gripper2base.append(T_bg[:3, :3])
        t_gripper2base.append(T_bg[:3, 3])

        R_target2cam.append(T_ct[:3, :3])
        t_target2cam.append(T_ct[:3, 3])

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )

    T_tool_camera = np.eye(4, dtype=np.float64)
    T_tool_camera[:3, :3] = R_cam2gripper
    T_tool_camera[:3, 3] = t_cam2gripper.reshape(3)

    return T_tool_camera


# ============================================================
# VALIDATION
# ============================================================

def validate(T_base_tool_list, T_cam_target_list, T_tool_camera, sample_labels=None):
    T_base_target_list = []

    for T_bg, T_ct in zip(T_base_tool_list, T_cam_target_list):
        # ^bT_t = ^bT_g * ^gT_c * ^cT_t
        T_bt = T_bg @ T_tool_camera @ T_ct
        T_base_target_list.append(T_bt)

    positions = np.array([T[:3, 3] for T in T_base_target_list])
    pos_mean = positions.mean(axis=0)
    pos_std = positions.std(axis=0)

    R0 = T_base_target_list[0][:3, :3]
    angs = [0.0]
    for T in T_base_target_list[1:]:
        dR = R0.T @ T[:3, :3]
        angs.append(rotation_angle_deg(dR))

    angs = np.array(angs, dtype=np.float64)
    pos_errs = np.linalg.norm(positions - pos_mean.reshape(1, 3), axis=1)

    # Pairwise target consistency (should be close to zero for a static board).
    pair_pos_err = []
    pair_ori_err = []
    for i in range(len(T_base_target_list)):
        Ti_inv = invert_T(T_base_target_list[i])
        for j in range(i + 1, len(T_base_target_list)):
            dT = Ti_inv @ T_base_target_list[j]
            pair_pos_err.append(np.linalg.norm(dT[:3, 3]))
            pair_ori_err.append(rotation_angle_deg(dT[:3, :3]))

    labels = sample_labels or [f"#{i+1}" for i in range(len(T_base_target_list))]
    worst_pos_idx = np.argsort(pos_errs)[::-1][: min(TOP_K_OUTLIERS, len(pos_errs))]
    worst_ori_idx = np.argsort(angs)[::-1][: min(TOP_K_OUTLIERS, len(angs))]

    metrics = {
        "target_pos_mean_m": pos_mean,
        "target_pos_std_m": pos_std,
        "target_pos_std_norm_m": float(np.linalg.norm(pos_std)),
        "target_ori_mean_deg": float(np.mean(angs)),
        "target_ori_max_deg": float(np.max(angs)),
        "target_pos_err_stats_m": scalar_stats(pos_errs),
        "target_ori_err_stats_deg": scalar_stats(angs),
        "target_pair_pos_err_stats_m": scalar_stats(pair_pos_err),
        "target_pair_ori_err_stats_deg": scalar_stats(pair_ori_err),
        "worst_pos_samples": [(labels[i], float(pos_errs[i])) for i in worst_pos_idx],
        "worst_ori_samples": [(labels[i], float(angs[i])) for i in worst_ori_idx],
    }
    return metrics


# ============================================================
# MAIN
# ============================================================

def evaluate_pose_direction(K, dist, board, pose_is_base_t_tool, methods, verbose):
    T_base_tool_list, T_cam_target_list, accepted, rejected = load_samples(
        IMAGES_DIR,
        POSES_DIR,
        K,
        dist,
        board,
        pose_is_base_t_tool=pose_is_base_t_tool,
    )

    if verbose:
        print("\n==============================================")
        print(f"Assumption: ROBOT_POSE_IS_BASE_T_TOOL = {pose_is_base_t_tool}")
        print(f"ROBOT_TRANSLATION_SCALE (to meters) = {ROBOT_TRANSLATION_SCALE}")
        print(f"Accepted samples: {len(accepted)}")
        print(f"Rejected samples: {len(rejected)}")

        print("\nAccepted images (in pairing order):")
        for stem, _, _, _ in accepted:
            print(f"  {stem}")

        if rejected:
            print("\nRejected sample summary:")
            for stem, reason in rejected[:15]:
                print(f"  {stem}: {reason}")
            if len(rejected) > 15:
                print(f"  ... and {len(rejected)-15} more")

    if len(accepted) < 8:
        raise RuntimeError(
            f"Too few valid samples ({len(accepted)}). Aim for at least ~15-20 good samples."
        )

    if verbose:
        reproj = [item[3]["reproj_rmse_px"] for item in accepted]
        num_charuco = [item[3]["num_charuco"] for item in accepted]
        cam_dist = [np.linalg.norm(T[:3, 3]) for T in T_cam_target_list]
        robot_positions = np.array([T[:3, 3] for T in T_base_tool_list], dtype=np.float64)
        robot_span = robot_positions.max(axis=0) - robot_positions.min(axis=0)
        robot_motion_t, robot_motion_r = transform_motion_stats(T_base_tool_list)
        cam_motion_t, cam_motion_r = transform_motion_stats(T_cam_target_list)

        print("\nInput diagnostics:")
        print("  reproj_rmse_px stats    =", scalar_stats(reproj))
        print("    Meaning: per-image CharUco reprojection RMSE in pixels (lower is better).")
        print("  num_charuco stats       =", scalar_stats(num_charuco))
        print("    Meaning: how many CharUco corners were used per image (higher is better).")
        print("  cam_target_dist_m stats =", scalar_stats(cam_dist))
        print("    Meaning: camera-to-board distance range across captures.")
        print("  robot_xyz_span_m        =", robot_span)
        print("    Meaning: workspace coverage of robot tool positions.")
        print("  robot motion trans m    =", robot_motion_t)
        print("  robot motion rot deg    =", robot_motion_r)
        print("    Meaning: pairwise robot motion magnitude; too small reduces observability.")
        print("  cam-target motion m     =", cam_motion_t)
        print("  cam-target rot deg      =", cam_motion_r)
        print("    Meaning: pairwise camera-board motion from PnP; should be diverse.")

    results = {}
    failed_methods = []
    sample_labels = [item[0] for item in accepted]

    for name, method in methods.items():
        try:
            T_tool_camera = solve_handeye(T_base_tool_list, T_cam_target_list, method)
            metrics = validate(
                T_base_tool_list,
                T_cam_target_list,
                T_tool_camera,
                sample_labels=sample_labels,
            )
            results[name] = (T_tool_camera, metrics)
        except Exception as e:
            failed_methods.append((name, str(e)))

    best_name = None
    best_score = np.inf
    for name, (_, metrics) in results.items():
        score = metrics["target_pos_std_norm_m"]
        if score < best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("No hand-eye method succeeded.")

    best_T, best_metrics = results[best_name]

    if verbose:
        if failed_methods:
            print("\nFailed methods:")
            for name, err in failed_methods:
                print(f"  {name}: {err}")

        detR, ortho_err = rotation_quality(best_T[:3, :3])
        cam_offset = float(np.linalg.norm(best_T[:3, 3]))
        print(f"\n=== BEST ({best_name}) ===")
        print("T_tool_camera (^gT_c) =")
        print(best_T)
        print(f"g->c translation norm m = {cam_offset:.6f}")
        print(f"R det / orthogonality   = {detR:.6f} / {ortho_err:.3e}")
        print("Validation:")
        print("  target_pos_mean_m      =", best_metrics["target_pos_mean_m"])
        print("    Meaning: estimated board origin in robot base (mean over all captures).")
        print("  target_pos_std_m       =", best_metrics["target_pos_std_m"])
        print("    Meaning: axis-wise spread of estimated board position (lower is better).")
        print("  target_pos_std_norm_m  =", best_metrics["target_pos_std_norm_m"])
        print("    Meaning: single position-consistency score used for solver ranking.")
        print("  target_ori_mean_deg    =", best_metrics["target_ori_mean_deg"])
        print("  target_ori_max_deg     =", best_metrics["target_ori_max_deg"])
        print("    Meaning: board orientation consistency in base frame (lower is better).")
        print("  target_pos_err_stats_m =", best_metrics["target_pos_err_stats_m"])
        print("    Meaning: per-image distance-to-mean distribution.")
        print("  target_ori_err_stats_deg =", best_metrics["target_ori_err_stats_deg"])
        print("    Meaning: per-image orientation-to-reference distribution.")
        print("  target_pair_pos_err_stats_m =", best_metrics["target_pair_pos_err_stats_m"])
        print("    Meaning: pairwise board-position disagreement (static board should be small).")
        print("  target_pair_ori_err_stats_deg =", best_metrics["target_pair_ori_err_stats_deg"])
        print("    Meaning: pairwise board-orientation disagreement.")
        print("  worst_pos_samples      =", best_metrics["worst_pos_samples"])
        print("  worst_ori_samples      =", best_metrics["worst_ori_samples"])
        print("    Meaning: highest-error images to inspect/remove first.")
    return {
        "pose_is_base_t_tool": pose_is_base_t_tool,
        "accepted": accepted,
        "rejected": rejected,
        "results": results,
        "failed_methods": failed_methods,
        "best_name": best_name,
        "best_T": best_T,
        "best_metrics": best_metrics,
    }


def main():
    K, dist = load_intrinsics_json(INTRINSICS_JSON)
    print("Loaded intrinsics.")

    board = create_charuco_board()

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    chosen_pose_dir = ROBOT_POSE_IS_BASE_T_TOOL

    final = evaluate_pose_direction(
        K,
        dist,
        board,
        pose_is_base_t_tool=chosen_pose_dir,
        methods=methods,
        verbose=True,
    )

    best_T = final["best_T"]
    best_name = final["best_name"]
    print(f"\nFinal selection: {best_name}")

    np.savez(
        "handeye_result.npz",
        T_tool_camera=best_T,
        method=best_name,
        K=K,
        dist=dist,
        robot_pose_is_base_t_tool=chosen_pose_dir,
    )
    print("\nSaved result to handeye_result.npz")


if __name__ == "__main__":
    main()
