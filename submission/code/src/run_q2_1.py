import numpy as np
from pathlib import Path
from scipy.signal import convolve2d

from io_utils import load_color, image_saver
from filters import gaussian_kernel
from freq import image_shapener, low_pass_filter, is_color, apply_func_to_each_channel

ROOT = Path(__file__).resolve().parents[1]
DATA, QOUT = ROOT/"data", ROOT/"out"/"q2_1"

def _ensure(p):
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")  # I fail early if inputs are absent

def _write_number(path, val):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{val:.8e}\n")  # I store MSE as a scalar

def _signed_vis(arr):
    m = np.max(np.abs(arr)) or 1e-8  # to prevent division by zero
    return 0.5 + 0.5*(arr / m)

def _run_unsharp(name, stem, size=9, sigma=1.5,
                 strengths_small=(0.25, 0.5, 1.0, 1.5),
                 strengths_big=(2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15)):
    p = DATA/name; _ensure(p)
    img = load_color(str(p)); outdir = QOUT/stem; outdir.mkdir(parents=True, exist_ok=True)

    low, high, sharp1 = image_shapener(img, size=size, sigma=sigma, amount=1.0) 
    image_saver(str(outdir/"input.png"), img)
    image_saver(str(outdir/"low.png"),   low)
    image_saver(str(outdir/"high.png"),  _signed_vis(high))
    image_saver(str(outdir/"strength1.0.png"), np.clip(sharp1, 0, 1))

    for s in list(strengths_small) + list(strengths_big):  # I sweep representative strengths
        _, _, sharp = image_shapener(img, size=size, sigma=sigma, amount=s)
        image_saver(str(outdir/f"sharp_strength{s}.png"), np.clip(sharp, 0, 1))

def _single_conv_unsharp(img, size=9, sigma=1.5, strength=2.0):
    G = gaussian_kernel(size, sigma)  # I use a δ − αG form
    k = np.zeros_like(G); k[G.shape[0]//2, G.shape[1]//2] = 1.0
    K = (1.0 + strength)*k - strength*G
    if is_color(img):
        return apply_func_to_each_channel(img, lambda ch: convolve2d(ch, K, mode="same", boundary="fill", fillvalue=0))
    else:
        return convolve2d(img, K, mode="same", boundary="fill", fillvalue=0)

def main():
    QOUT.mkdir(parents=True, exist_ok=True)

    _run_unsharp("taj.jpg",             "taj")
    _run_unsharp("cityscape.png",       "cityscape")
    _run_unsharp("blurry_portrait.png", "portrait")

    taj = load_color(str(DATA/"taj.jpg"))
    taj_single = _single_conv_unsharp(taj, size=9, sigma=1.5, strength=2.0)  # I show a single-pass equivalent
    image_saver(str(QOUT/"taj"/"single_conv_strength2.png"), np.clip(taj_single, 0, 1))

    sharp0 = load_color(str(DATA/"sharp_image_to_blur.png"))
    size_eval, sigma_eval = 31, 4.5  # I apply a stronger blur for a clearer test
    blurred = low_pass_filter(sharp0, size=size_eval, sigma=sigma_eval)

    evaldir = QOUT/"evaluation"; evaldir.mkdir(parents=True, exist_ok=True)
    image_saver(str(evaldir/"original.png"), sharp0)
    image_saver(str(evaldir/"blurred.png"),  blurred)
    for s in (0.25, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 15, 30, 60, 120):
        _, _, sharpened = image_shapener(blurred, size=9, sigma=1.5, amount=s)
        image_saver(str(evaldir/f"resharp_strength{s}.png"), np.clip(sharpened, 0, 1))
        mse = np.mean((np.clip(sharpened,0,1) - np.clip(sharp0,0,1))**2)
        _write_number(evaldir/f"mse_strength{s}.txt", mse)  # I keep numeric comparisons

    print("Part 2.1 outputs written to", QOUT)

if __name__ == "__main__":
    main()
