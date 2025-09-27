import numpy as np
from pathlib import Path
from scipy.signal import convolve2d 

from io_utils import load_gray, image_saver
from edges import gradient_image_xy, gradient_magnitude_image, binarize_edges, signed_image_viewer

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, QOUT = ROOT/"data", ROOT/"out", ROOT/"out"/"q1_2"

def main():
    QOUT.mkdir(parents=True, exist_ok=True)  # I create the output folder once

    cam = load_gray(str(DATA/"cameraman.png"))  # I use the standard test image

    gx, gy = gradient_image_xy(cam)  # I keep derivative kernels inside edges.* for consistency
    image_saver(str(QOUT/"cameraman_dx.png"), signed_image_viewer(gx))
    image_saver(str(QOUT/"cameraman_dy.png"), signed_image_viewer(gy))

    gm = gradient_magnitude_image(cam, gx, gy, normalize_for_display=True)  # I request [0,1] here
    image_saver(str(QOUT/"cameraman_gradmag.png"), gm)

    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        image_saver(str(QOUT/f"cameraman_edges_bin_t{int(t*100):03d}.png"), binarize_edges(gm, t))  # I sweep a small threshold set

    CHOSEN_T = 0.20  # I keep a single representative threshold for the final edge map
    image_saver(str(QOUT/"cameraman_edges_bin.png"), binarize_edges(gm, CHOSEN_T))

    print("Part 1.2 outputs written to", QOUT) 

if __name__ == "__main__":
    main()
