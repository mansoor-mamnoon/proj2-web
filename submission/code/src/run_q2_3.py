import numpy as np
from pathlib import Path

from io_utils import load_color, image_saver
from freq import image_matcher
from stacks import (
    gaussian_stack, laplacian_stack, tile_stacked_sideways,
    make_soft_vertical_mask, multiband_blend
)
from filters import zero_to_one_normalizer

ROOT = Path(__file__).resolve().parents[1]
DATA, QOUT = ROOT/"data", ROOT/"out"/"q2_3"

def main():
    QOUT.mkdir(parents=True, exist_ok=True)  

    ap, orp = DATA/"apple.jpeg", DATA/"orange.jpeg"
    if not ap.exists() or not orp.exists():
        raise FileNotFoundError("Put apple.jpeg and orange.jpeg in data/")  # fail clearly if assets are missing

    A0, B0 = load_color(str(ap)), load_color(str(orp))
    A, B   = image_matcher(A0, B0)  # I match sizes to make stacks consistent

    levels, size, sigma = 7, 31, 5.0
    GA = gaussian_stack(A, levels, size=size, sigma=sigma)
    GB = gaussian_stack(B, levels, size=size, sigma=sigma)
    LA = laplacian_stack(A, levels, size=size, sigma=sigma)
    LB = laplacian_stack(B, levels, size=size, sigma=sigma)

    image_saver(str(QOUT/"apple_gauss_stack.png"),  tile_stacked_sideways(GA))
    image_saver(str(QOUT/"apple_lap_stack.png"),    tile_stacked_sideways(LA))
    image_saver(str(QOUT/"orange_gauss_stack.png"), tile_stacked_sideways(GB))
    image_saver(str(QOUT/"orange_lap_stack.png"),   tile_stacked_sideways(LB))

    H, W = A.shape[:2]; mask = make_soft_vertical_mask(H, W)  # I use a cosine ramp mask
    GM = gaussian_stack(mask, levels, size=size, sigma=sigma)

    def mask_like(m, img_level): return m if img_level.ndim == 2 else np.repeat(m[..., None], 3, axis=2)  # I broadcast to RGB when needed
    def vis(arr): return zero_to_one_normalizer(arr)  # I keep gray stacks in [0,1]
    def vis_color(arr):
        m = np.max(np.abs(arr)); return (0.5 + 0.5*(arr/m)) if m > 0 else np.full_like(arr, 0.5)

    rows_gray, rows_color = [], []
    letters = iter(list("abcdefghijkl"))
    for lvl in (0, 2, 4):
        gm = mask_like(GM[lvl], LA[lvl])
        left   = gm * LA[lvl]
        middle = (1.0 - gm) * LB[lvl]
        right  = left + middle
        a = next(letters); image_saver(str(QOUT/f"{a}.png"),       vis(left));   image_saver(str(QOUT/f"{a}_color.png"),   vis_color(left))
        b = next(letters); image_saver(str(QOUT/f"{b}.png"),       vis(middle)); image_saver(str(QOUT/f"{b}_color.png"),   vis_color(middle))
        c = next(letters); image_saver(str(QOUT/f"{c}.png"),       vis(right));  image_saver(str(QOUT/f"{c}_color.png"),   vis_color(right))
        rows_gray.append(np.concatenate([vis(left), vis(middle), vis(right)], axis=1))  # I assemble gray grid row
        rows_color.append(np.concatenate([vis_color(left), vis_color(middle), vis_color(right)], axis=1))

    out, GA2, GB2, GM2, LA2, LB2, Lout = multiband_blend(A, B, mask, levels=levels, size=size, sigma=sigma)
    image_saver(str(QOUT/"oraple_result.png"), out)  # I save the final blend

    m3 = mask_like(mask, A)
    row4_left  = np.clip(A * m3, 0, 1)
    row4_mid   = np.clip(B * (1.0 - m3), 0, 1)
    row4_right = np.clip(out, 0, 1)
    j = next(letters); image_saver(str(QOUT/f"{j}.png"), row4_left);  image_saver(str(QOUT/f"{j}_color.png"), row4_left)
    k = next(letters); image_saver(str(QOUT/f"{k}.png"), row4_mid);   image_saver(str(QOUT/f"{k}_color.png"), row4_mid)
    l = next(letters); image_saver(str(QOUT/f"{l}.png"), row4_right); image_saver(str(QOUT/f"{l}_color.png"), row4_right)

    rows_gray.append(np.concatenate([row4_left, row4_mid, row4_right], axis=1))
    rows_color.append(np.concatenate([row4_left, row4_mid, row4_right], axis=1))
    grid_gray  = np.concatenate(rows_gray, axis=0);  image_saver(str(QOUT/"szeliski_342_grid.png"),       grid_gray)
    grid_color = np.concatenate(rows_color, axis=0); image_saver(str(QOUT/"szeliski_342_grid_color.png"), grid_color)

    print("Part 2.3 outputs written to", QOUT) 
    print("Saved a.png … l.png, *_color.png, szeliski_342_grid.png, szeliski_342_grid_color.png, and oraple_result.png.")

if __name__ == "__main__":
    main()
