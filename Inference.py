import os
import shutil
from datetime import datetime

import cv2
import numpy as np
import torch

from ML.Models import UNetSmall
from ML.Sensor_maps import make_sensor_maps

BASE_DIR = os.path.expanduser("~/Desktop/DataControl")

PENDING_FOLDER = os.path.join(BASE_DIR, "Pending")
STRENGTH_GRIDS_FOLDER = os.path.join(BASE_DIR, "Strength_grids")
ARCHIVE_FOLDER = os.path.join(BASE_DIR, "Archive")
LOGS_FOLDER = os.path.join(BASE_DIR, "Logs")
MAPS_FOLDER = os.path.join(BASE_DIR, "Maps")

WEIGHTS_PATH = os.path.join(BASE_DIR, "ML", "BestWeights.pt")

BLOCK_SIZE = 32

M_SCALE_DIVISOR = 10_000.0
C_SCALE_DIVISOR = 1_000.0


def load_model(device):
    model = UNetSmall(in_channels=5, out_channels=3, base_channels=16).to(device)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    return model


def ensure_folders():
    os.makedirs(MAPS_FOLDER, exist_ok=True)
    os.makedirs(STRENGTH_GRIDS_FOLDER, exist_ok=True)
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)


def crop_to_multiple_of_16(gray):
    h, w = gray.shape

    new_h = (h // 16) * 16
    new_w = (w // 16) * 16

    if new_h < 16 or new_w < 16:
        raise ValueError(f"Image too small after multiple-of-16 crop: original shape={gray.shape}")

    if new_h == h and new_w == w:
        return gray, False

    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2

    cropped = gray[start_h:start_h + new_h, start_w:start_w + new_w]
    return cropped, True


def read_grayscale(folder_path):
    files = os.listdir(folder_path)
    png_files = [f for f in files if f.lower().endswith(".png")]

    if len(png_files) == 0:
        return None, None

    image_name = png_files[0]
    image_path = os.path.join(folder_path, image_name)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None

    gray = gray.astype(np.float32) / 255.0
    original_shape = gray.shape

    gray, was_changed = crop_to_multiple_of_16(gray)

    if was_changed:
        print(f"Cropped image from {original_shape} to {gray.shape}")
    else:
        print(f"Image already valid: {gray.shape}")

    return gray, image_path


@torch.no_grad()
def predict_maps_from_gray(model, gray, device):
    x = make_sensor_maps(gray).unsqueeze(0).to(device)   # (1, 5, H, W)
    output = model(x).squeeze(0).cpu().numpy()           # (3, H, W)

    cohesion_map = output[0]
    friction_map = output[1]
    density_map = output[2]

    return friction_map, cohesion_map, density_map


def compute_strength_maps(friction_deg_map, cohesion_map, density_map, lunar_g=1.62):
    friction_deg_map = np.clip(friction_deg_map, 5.0, 60.0)
    cohesion_map = np.clip(cohesion_map, 0.0, None)
    density_map = np.clip(density_map, 0.0, None)

    phi_rad = np.deg2rad(friction_deg_map)

    tan_phi = np.tan(phi_rad)
    tan_phi = np.clip(tan_phi, 0.01, 10.0)

    exp_term = np.exp(np.clip(np.pi * tan_phi, 0.0, 20.0))
    nq_map = exp_term * (np.tan(np.deg2rad(45.0) + phi_rad / 2.0) ** 2)
    nc_map = (nq_map - 1.0) / tan_phi
    n_gamma_map = 2.0 * (nq_map + 1.0) * tan_phi

    gamma_map = density_map * lunar_g

    m_map = 0.5 * gamma_map * n_gamma_map
    c_map = cohesion_map * nc_map

    m_map = np.nan_to_num(m_map, nan=0.0, posinf=1e9, neginf=0.0)
    c_map = np.nan_to_num(c_map, nan=0.0, posinf=1e9, neginf=0.0)

    return m_map, c_map


def block_reduce_mean(arr, block_size):
    h, w = arr.shape

    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size

    arr = arr[:h_trim, :w_trim]
    arr = arr.reshape(h_trim // block_size, block_size, w_trim // block_size, block_size)

    return arr.mean(axis=(1, 3))


def make_heatmap(m_map):
    m_map = np.nan_to_num(m_map, nan=0.0, posinf=0.0, neginf=0.0)

    m_vis = np.log1p(m_map)

    p2 = np.percentile(m_vis, 2)
    p98 = np.percentile(m_vis, 98)

    if abs(p98 - p2) < 1e-8:
        norm = np.zeros_like(m_vis, dtype=np.float32)
    else:
        norm = (m_vis - p2) / (p98 - p2)

    norm = np.clip(norm, 0.0, 1.0)

    norm = norm ** 0.85

    heat_u8 = (norm * 255).astype(np.uint8)

    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    hsv = cv2.cvtColor(heat, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.8, 0, 255).astype(np.uint8)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)

    heat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return heat


def overlay_heatmap(gray, heat):
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    gray_edges = cv2.Laplacian(gray_u8, cv2.CV_8U, ksize=3)
    gray_edges_bgr = cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2BGR)

    heat_strong = cv2.convertScaleAbs(heat, alpha=1.08, beta=0)

    base = cv2.addWeighted(gray_bgr, 0.55, gray_edges_bgr, 0.35, 0.0)

    overlay = cv2.addWeighted(base, 0.65, heat_strong, 0.75, 0.0)
    return overlay


def write_matrix_as_text(f, title, arr, fmt="%.2f", col_width=10):
    f.write(f"=== {title} ===\n")
    for row in arr:
        line = " ".join(f"{val:{col_width}.2f}" for val in row)
        f.write(line + "\n")
    f.write("\n")


def save_strength_grid(timestamp, base_name, m_map, c_map):
    m_small = block_reduce_mean(m_map, BLOCK_SIZE) / M_SCALE_DIVISOR
    c_small = block_reduce_mean(c_map, BLOCK_SIZE) / C_SCALE_DIVISOR

    grid_path = os.path.join(STRENGTH_GRIDS_FOLDER, f"{timestamp}_{base_name}_grid.txt")

    with open(grid_path, "w") as f:
        f.write(f"BLOCK_SIZE = {BLOCK_SIZE}\n")
        f.write(f"M values divided by {int(M_SCALE_DIVISOR)}\n")
        f.write(f"C values divided by {int(C_SCALE_DIVISOR)}\n")
        f.write(f"M grid shape: {m_small.shape}\n")
        f.write(f"C grid shape: {c_small.shape}\n\n")

        write_matrix_as_text(f, "M MAP (slope, scaled)", m_small, col_width=9)
        write_matrix_as_text(f, "C MAP (intercept, scaled)", c_small, col_width=9)

    return grid_path


def save_overlay(timestamp, base_name, overlay):
    overlay_path = os.path.join(MAPS_FOLDER, f"{timestamp}_{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    return overlay_path


def save_log_stack(timestamp, base_name, cohesion_map, friction_map, density_map):
    c_small = block_reduce_mean(cohesion_map, BLOCK_SIZE)
    f_small = block_reduce_mean(friction_map, BLOCK_SIZE)
    d_small = block_reduce_mean(density_map, BLOCK_SIZE)

    log_path = os.path.join(LOGS_FOLDER, f"{timestamp}_{base_name}_log.txt")

    with open(log_path, "w") as f:
        f.write(f"BLOCK_SIZE = {BLOCK_SIZE}\n")
        f.write(f"Log grid shape: {c_small.shape}\n\n")

        write_matrix_as_text(f, "COHESION", c_small, col_width=9)
        write_matrix_as_text(f, "FRICTION", f_small, col_width=9)
        write_matrix_as_text(f, "DENSITY", d_small, col_width=9)

    return log_path


def move_original_to_archive(image_path, timestamp, base_name):
    archive_path = os.path.join(ARCHIVE_FOLDER, f"{timestamp}_{base_name}_original.png")
    shutil.move(image_path, archive_path)
    return archive_path


def main():
    ensure_folders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gray, image_path = read_grayscale(PENDING_FOLDER)
    if gray is None:
        print("No valid grayscale PNG found in Pending.")
        return

    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model = load_model(device)

    friction_map, cohesion_map, density_map = predict_maps_from_gray(model, gray, device)

    m_map, c_map = compute_strength_maps(
        friction_deg_map=friction_map,
        cohesion_map=cohesion_map,
        density_map=density_map,
        lunar_g=1.62
    )

    heat = make_heatmap(m_map)
    overlay = overlay_heatmap(gray, heat)

    overlay_path = save_overlay(timestamp, base_name, overlay)
    grid_path = save_strength_grid(timestamp, base_name, m_map, c_map)
    log_path = save_log_stack(timestamp, base_name, cohesion_map, friction_map, density_map)
    archive_path = move_original_to_archive(image_path, timestamp, base_name)

    print("Done.")
    print("Overlay:", overlay_path)
    print("Grid:", grid_path)
    print("Log:", log_path)
    print("Archived original:", archive_path)


if __name__ == "__main__":
    main()