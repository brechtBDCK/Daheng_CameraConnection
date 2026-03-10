"""
Build base-to-tool transformation matrices from measured robot cartesian poses.

Input poses are expected in FANUC style:
- Position: X/Y/Z in mm
- Orientation: W/P/R in degrees

The resulting matrix is a homogeneous transform T_base_tool (4x4).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

POSE_KEYS = ("X", "Y", "Z", "W", "P", "R")
_POSE_LINE_PATTERN = re.compile(r"^\s*Pose\s*#\s*(\d+)\s*:\s*(.+?)\s*$")
_VALUE_PATTERN = re.compile(r"([XYZWPR])\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _rotation_matrix_from_wpr(w_deg: float, p_deg: float, r_deg: float) -> np.ndarray:
    """
    Convert FANUC W/P/R (degrees) to a 3x3 rotation matrix.

    FANUC W/P/R here is interpreted as:
    - W: rotation about X
    - P: rotation about Y
    - R: rotation about Z

    with composition:
    R = Rz(R) @ Ry(P) @ Rx(W)
    """
    w, p, r = np.deg2rad([w_deg, p_deg, r_deg])
    cw, sw = np.cos(w), np.sin(w)  # Rx(W)
    cp, sp = np.cos(p), np.sin(p)  # Ry(P)
    cr, sr = np.cos(r), np.sin(r)  # Rz(R)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cw, -sw],
            [0.0, sw, cw],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return rz @ ry @ rx


def make_transformation_matrix(measured_coords: Mapping[str, float]) -> np.ndarray:
    """
    Build the 4x4 T_base_tool matrix from one measured cartesian pose.

    measured_coords must contain X/Y/Z/W/P/R.
    """
    missing = [k for k in POSE_KEYS if k not in measured_coords]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"Missing pose keys: {missing_txt}")

    x = float(measured_coords["X"])
    y = float(measured_coords["Y"])
    z = float(measured_coords["Z"])
    w = float(measured_coords["W"])
    p = float(measured_coords["P"])
    r = float(measured_coords["R"])

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_matrix_from_wpr(w, p, r)
    transform[:3, 3] = np.array([x, y, z], dtype=float)
    return transform


def parse_pose_line(line: str) -> tuple[int, dict[str, float]] | None:
    """Parse one line like: Pose #1:X: ..., Y: ..., Z: ..., W: ..., P: ..., R: ..."""
    stripped = line.strip()
    if not stripped:
        return None

    match = _POSE_LINE_PATTERN.match(stripped)
    if not match:
        raise ValueError(f"Invalid pose line: {line.rstrip()}")

    pose_idx = int(match.group(1))
    payload = match.group(2)

    values = {key: float(value) for key, value in _VALUE_PATTERN.findall(payload)}
    missing = [k for k in POSE_KEYS if k not in values]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"Pose #{pose_idx} is missing: {missing_txt}")

    return pose_idx, {k: values[k] for k in POSE_KEYS}


def load_cartesian_poses(path: str | Path) -> list[tuple[int, dict[str, float]]]:
    """Load all cartesian poses from robot_position_cartesian.txt style file."""
    pose_file = Path(path)
    poses: list[tuple[int, dict[str, float]]] = []

    with pose_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed = parse_pose_line(line)
            except ValueError as exc:
                raise ValueError(f"{pose_file}:{line_number}: {exc}") from exc
            if parsed is None:
                continue
            poses.append(parsed)

    if not poses:
        raise ValueError(f"No poses found in {pose_file}")
    return poses


def save_pose_matrices(
    poses: Iterable[tuple[int, Mapping[str, float]]],
    output_dir: str | Path,
) -> list[Path]:
    """
    Save pose matrices as 0001_robot_pose.npz, 0002_robot_pose.npz, ...

    Each .npz contains a single 4x4 matrix named 'T_base_tool'.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for pose_idx, pose in poses:
        matrix = make_transformation_matrix(pose)
        output_path = destination / f"{pose_idx:04d}_robot_pose.npz"
        np.savez(output_path, T_base_tool=matrix)
        written.append(output_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert measured robot cartesian poses into base-to-tool "
            "4x4 transformation matrices."
        )
    )
    parser.add_argument(
        "--input",
        default="robot_position_cartesian.txt",
        help="Path to pose text file (default: robot_position_cartesian.txt).",
    )
    parser.add_argument(
        "--output-dir",
        default="eye_hand_calibration/samples/robot_poses",
        help="Directory to write 0001_robot_pose.npz files.",
    )
    args = parser.parse_args()

    poses = load_cartesian_poses(args.input)
    written = save_pose_matrices(poses, args.output_dir)
    print(f"Wrote {len(written)} base-to-tool matrices to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
