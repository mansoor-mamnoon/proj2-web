import numpy as np
from pathlib import Path
from scipy.signal import convolve2d

from io_utils import load_gray, image_saver
from filters import gaussian_kernel, derivative_of_gauss_kernel, zero_to_one_normalizer, x_diff, y_diff
from edges import gradient_image_xy, gradient_magnitude_image

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, QOUT = ROOT/"data", ROOT/"out", ROOT/"out"/"q1_3"

def _write_number(path, val):
    path.parent.mkdir(parents=True, exist_ok=True)  # I keep scalar metrics for inspection
    with open(path, "w") as f:
        f.write(f"{val:.8e}\n")

def _core(a, m=2):
    return a[m:-m, m:-m] if a.shape[0] > 2*m and a.shape[1] > 2*m else a  # I compare cores to avoid boundary artifacts

def _signed_vis(arr: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(arr)))  # I map signed arrays to [0,1] for viewing only
    if m == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return 0.5 + 0.5 * (arr / m)

def _center_pad(kernel: np.ndarray, target: int) -> np.ndarray:
    out = np.zeros((target, target), dtype=np.float32)  # I center small kernels for visualization
    kh, kw = kernel.shape
    y0 = target//2 - kh//2
    x0 = target//2 - kw//2
    out[y0:y0+kh, x0:x0+kw] = kernel
    return out

def main():
    QOUT.mkdir(parents=True, exist_ok=True)

    img = load_gray(str(DATA/"cameraman.png"))
    size, sigma = 9, 1.5  # I use a moderate blur level as per spec examples

    G = gaussian_kernel(size, sigma)
    image_saver(str(QOUT/"gaussian_kernel.png"), zero_to_one_normalizer(G))  # I show the smoothing kernel

    Dx, Dy = x_diff(), y_diff()
    image_saver(str(QOUT/"dx_kernel.png"), _signed_vis(_center_pad(Dx, size)))  # I visualize finite diff
    image_saver(str(QOUT/"dy_kernel.png"), _signed_vis(_center_pad(Dy, size)))

    kx, ky = derivative_of_gauss_kernel(size, sigma)
    image_saver(str(QOUT/"dog_kx.png"), _signed_vis(kx))  # I also show DoG kernels
    image_saver(str(QOUT/"dog_ky.png"), _signed_vis(ky))

    img_blur = convolve2d(img, G, mode="same", boundary="fill", fillvalue=0)  # I blur first then take finite diffs
    gx_b, gy_b = gradient_image_xy(img_blur)

    kx, ky = derivative_of_gauss_kernel(size, sigma)  # I compute DoG once more for clarity of flow
    image_saver(str(QOUT/"dog_kx.png"), zero_to_one_normalizer(kx))  # I normalize this version for a different view
    image_saver(str(QOUT/"dog_ky.png"), zero_to_one_normalizer(ky))
    gx_dog = convolve2d(img, kx, mode="same", boundary="fill", fillvalue=0)
    gy_dog = convolve2d(img, ky, mode="same", boundary="fill", fillvalue=0)

    gm_b_raw = gradient_magnitude_image(img_blur, gx_b, gy_b, normalize_for_display=False)
    gm_d_raw = gradient_magnitude_image(img, gx_dog, gy_dog, normalize_for_display=False)
    M = max(gm_b_raw.max(), gm_d_raw.max(), 1e-8)  # I normalize jointly for fair visual thresholds
    gm_b_disp, gm_d_disp = gm_b_raw / M, gm_d_raw / M

    image_saver(str(QOUT/"cameraman_blur.png"), img_blur)
    image_saver(str(QOUT/"cameraman_blur_gradmag.png"), gm_b_disp)
    image_saver(str(QOUT/"cameraman_dog_gradmag.png"),  gm_d_disp)
    for t in (0.10, 0.15, 0.20, 0.25):
        image_saver(str(QOUT/f"cameraman_blur_edges_t{int(t*100):03d}.png"), (gm_b_disp >= t).astype(np.float32))
        image_saver(str(QOUT/f"cameraman_dog_edges_t{int(t*100):03d}.png"),  (gm_d_disp >= t).astype(np.float32))

    gx_blur_then_dx = convolve2d(img_blur, x_diff(), mode="same", boundary="fill", fillvalue=0)
    gy_blur_then_dy = convolve2d(img_blur, y_diff(), mode="same", boundary="fill", fillvalue=0)

    gx_diff = np.max(np.abs(_core(gx_dog) - _core(gx_blur_then_dx)))
    gy_diff = np.max(np.abs(_core(gy_dog) - _core(gy_blur_then_dy)))
    gm_diff = np.max(np.abs(_core(gm_b_raw) - _core(gm_d_raw)))
    _write_number(QOUT/"diff_gx_core.txt", gx_diff)  # I record scalar diffs for verification
    _write_number(QOUT/"diff_gy_core.txt", gy_diff)
    _write_number(QOUT/"diff_gradmag_core.txt", gm_diff)

    print("Part 1.3 outputs written to", QOUT)

if __name__ == "__main__":
    main()
