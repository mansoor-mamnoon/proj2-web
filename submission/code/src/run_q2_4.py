from pathlib import Path
import numpy as np

from io_utils import load_color, image_saver
from freq import image_matcher
from stacks import (
    multiband_blend,
    make_soft_vertical_mask,
    tile_stacked_sideways,
    mask_circle,
    mask_horiz,
    
)
from align_image_code import align_images

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
QOUT = ROOT / "out" / "q2_4"

WEBROOT = Path(__file__).resolve().parents[3]  
ASSETS  = WEBROOT / "assets"
DATA_WEB = ASSETS / "data"
QOUT_WEB = ASSETS / "run_q2_4"

def blend_and_save_to_assets(a_name, b_name, stem, mask_fn, levels=7, size=31, sigma=5.0, save_process=True):
    a_path, b_path = DATA_WEB/a_name, DATA_WEB/b_name; ensure(a_path); ensure(b_path)
    A0, B0 = load_color(str(a_path)), load_color(str(b_path))
    A, B = image_matcher(A0, B0)
    H, W = A.shape[:2]; M = mask_fn(H, W)

    outdir = QOUT_WEB/stem; mkdir(outdir)
    image_saver(str(outdir/"mask.png"), M)

    out, GA, GB, GM, LA, LB, Lout = multiband_blend(A, B, M, levels=levels, size=size, sigma=sigma)
    image_saver(str(outdir/"result.png"), out)
    if save_process:
        M3 = np.repeat(M[..., None], 3, axis=2)
        image_saver(str(outdir/"masked_A.png"), A * M3)            # top image under mask
        image_saver(str(outdir/"masked_B.png"), B * (1.0 - M3))    # bottom image under inverse
        image_saver(str(outdir/"lap_outputs_tile.png"), tile_stacked_sideways(Lout))


def ensure(p):
    if not p.exists():
        raise FileNotFoundError(f"missing {p}") 

def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)

def blend_and_save(a_name, b_name, stem, mask_fn, levels=7, size=31, sigma=5.0, save_process=False):
    a_path, b_path = DATA/a_name, DATA/b_name; ensure(a_path); ensure(b_path)
    A0, B0 = load_color(str(a_path)), load_color(str(b_path))
    A, B = image_matcher(A0, B0)  # I size match to avoid shape issues
    H, W = A.shape[:2]; M = mask_fn(H, W)

    outdir = QOUT/stem; mkdir(outdir)
    image_saver(str(outdir/"mask.png"), M)  # I save the mask for reference in the webpage

    out, GA, GB, GM, LA, LB, Lout = multiband_blend(A, B, M, levels=levels, size=size, sigma=sigma)
    image_saver(str(outdir/"result.png"), out)
    if save_process:
        M3 = np.repeat(M[..., None], 3, axis=2)
        image_saver(str(outdir/"masked_A.png"), A * M3)
        image_saver(str(outdir/"masked_B.png"), B * (1.0 - M3))
        image_saver(str(outdir/"lap_outputs_tile.png"), tile_stacked_sideways(Lout))  # I tile Laplacian outputs

def blend_and_save_aligned_vertical(a_name, b_name, stem,
                                     levels=7, size=31, sigma=5.0,
                                     ramp=120, crop_ratio=0.20, save_process=True):
    a_path, b_path = DATA/a_name, DATA/b_name; ensure(a_path); ensure(b_path)
    A_col, B_col = load_color(str(a_path)), load_color(str(b_path))
    A_aligned, B_aligned = align_images(A_col, B_col)  # interactive alignment

    cut = int(crop_ratio * A_aligned.shape[0]) 
    if cut > 0:
        A_aligned = A_aligned[cut:, :]
        B_aligned = B_aligned[cut:, :]

    H, W = A_aligned.shape[:2]; M = make_soft_vertical_mask(H, W)
    outdir = QOUT/stem; mkdir(outdir)
    image_saver(str(outdir/"aligned_A.png"), A_aligned)
    image_saver(str(outdir/"aligned_B.png"), B_aligned)
    image_saver(str(outdir/"mask.png"), M)

    out, GA, GB, GM, LA, LB, Lout = multiband_blend(A_aligned, B_aligned, M, levels=levels, size=size, sigma=sigma)
    image_saver(str(outdir/"result.png"), out)
    if save_process:
        M3 = np.repeat(M[..., None], 3, axis=2)
        image_saver(str(outdir/"masked_A.png"), A_aligned * M3)
        image_saver(str(outdir/"masked_B.png"), B_aligned * (1.0 - M3))
        image_saver(str(outdir/"lap_outputs_tile.png"), tile_stacked_sideways(Lout))

def main():
    mkdir(QOUT) 

    blend_and_save_to_assets(
        "sunset.png", "city_night.png", "sunset_city_horizontal",
        mask_fn=lambda h, w: mask_horiz(h, w, ramp=120),   # ramp ↑ = softer band
        levels=7, size=31, sigma=5.0, save_process=True
    )

    blend_and_save(
        "apple.jpeg", "orange.jpeg", "oraple_vertical",
        mask_fn=lambda h, w: make_soft_vertical_mask(h, w),
        levels=7, size=31, sigma=5.0, save_process=False
    )

    blend_and_save(
        "cat.png", "dog.png", "cat_dog_irregular",
        mask_fn=lambda h, w: mask_circle(h, w, radius=None, ramp=20),
        levels=7, size=31, sigma=5.0, save_process=True
    )

    blend_and_save_aligned_vertical(
        "me_before.jpeg", "me_after.jpeg", "me_before_after_vertical_aligned",
        levels=7, size=31, sigma=5.0, ramp=120, crop_ratio=0.20, save_process=True
    )

    print("Part 2.4 outputs written to", QOUT)  

if __name__ == "__main__":
    main()
