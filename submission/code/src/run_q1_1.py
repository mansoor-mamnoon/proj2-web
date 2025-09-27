import time  # I use time.perf_counter for timings
import numpy as np
from pathlib import Path
from scipy.signal import convolve2d

from io_utils import load_gray, image_saver
from filters import box_filter, x_diff, y_diff
from conv import conv_4_loops, conv_2_loops

ROOT = Path(__file__).resolve().parents[3]  # I go two levels up from src/ to proj2website
DATA = ROOT / "assets" / "data"            # I point to proj2website/assets/data
QOUT = ROOT / "submission" / "out" / "q1_1"  # I put outputs under submission/out/q1_1

def write_number(path: Path, val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{float(val):.8e}\n")

def signed_display(arr: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(arr)))  # I normalize by max abs so signed outputs are visible
    if m == 0:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return 0.5 + 0.5 * (arr / m)

def benchmark(func, *args, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return float(np.median(times))

def main():
    QOUT.mkdir(parents=True, exist_ok=True)

    selfie_path = next((x for x in [DATA/"selfie.jpg", DATA/"selfie.png"] if x.exists()), None)
    if selfie_path is None:
        raise FileNotFoundError("put selfie.jpg or selfie.png in assets/data/")

    img = load_gray(str(selfie_path))

    B9 = box_filter(9)
    Dx, Dy = x_diff(), y_diff()

    box4, box2 = conv_4_loops(img, B9), conv_2_loops(img, B9)
    dx4, dy4   = conv_4_loops(img, Dx), conv_4_loops(img, Dy)
    dx2, dy2   = conv_2_loops(img, Dx), conv_2_loops(img, Dy)

    boxs = convolve2d(img, B9[::-1, ::-1], mode="same", boundary="fill", fillvalue=0)
    dxs  = convolve2d(img, Dx[::-1, ::-1], mode="same", boundary="fill", fillvalue=0)
    dys  = convolve2d(img, Dy[::-1, ::-1], mode="same", boundary="fill", fillvalue=0)

    diffs = {
        "box_4_vs_scipy": np.max(np.abs(box4 - boxs)),
        "box_2_vs_scipy": np.max(np.abs(box2 - boxs)),
        "dx_4_vs_scipy":  np.max(np.abs(dx4 - dxs)),
        "dx_2_vs_scipy":  np.max(np.abs(dx2 - dxs)),
        "dy_4_vs_scipy":  np.max(np.abs(dy4 - dys)),
        "dy_2_vs_scipy":  np.max(np.abs(dy2 - dys)),
        "box_4_vs_2":     np.max(np.abs(box4 - box2)),
        "dx_4_vs_2":      np.max(np.abs(dx4 - dx2)),
        "dy_4_vs_2":      np.max(np.abs(dy4 - dy2)),
    }
    for k, v in diffs.items():
        write_number(QOUT / f"{k}.txt", v)

    runtimes = {
        "time_box_4": benchmark(conv_4_loops, img, B9),
        "time_box_2": benchmark(conv_2_loops, img, B9),
        "time_box_scipy": benchmark(convolve2d, img, B9[::-1, ::-1], "same", "fill", 0),
    }
    for k, v in runtimes.items():
        write_number(QOUT / f"{k}.txt", v)

    image_saver(str(QOUT/"selfie_box9_4loops.png"), box4)
    image_saver(str(QOUT/"selfie_box9_2loops.png"), box2)
    image_saver(str(QOUT/"selfie_box9_scipy.png"), boxs)
    image_saver(str(QOUT/"selfie_dx_4loops.png"), signed_display(dx4))
    image_saver(str(QOUT/"selfie_dx_2loops.png"), signed_display(dx2))
    image_saver(str(QOUT/"selfie_dx_scipy.png"),  signed_display(dxs))
    image_saver(str(QOUT/"selfie_dy_4loops.png"), signed_display(dy4))
    image_saver(str(QOUT/"selfie_dy_2loops.png"), signed_display(dy2))
    image_saver(str(QOUT/"selfie_dy_scipy.png"),  signed_display(dys))

    print("Runtimes (ms):", runtimes)  # I print them here so I can see quickly
    print("1.1 outputs written to", QOUT)

if __name__ == "__main__":
    main()
