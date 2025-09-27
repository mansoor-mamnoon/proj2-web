from pathlib import Path
from io_utils import load_color, image_saver
from align_image_code import align_images
from freq import to_gray_1ch, hybrid_maker_aligned, fft_log_gray

ROOT = Path(__file__).resolve().parents[1]
DATA, QOUT = ROOT/"data", ROOT/"out"/"q2_2"

def _ensure(p):
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")  

def _run_pair(a_name, b_name, stem, low_size, low_sigma, high_size, high_sigma, do_fft=False):
    a_path, b_path = DATA/a_name, DATA/b_name; _ensure(a_path); _ensure(b_path)
    outdir = QOUT/stem; outdir.mkdir(parents=True, exist_ok=True)  # I keep outputs grouped per pair

    A = load_color(str(a_path)); B = load_color(str(b_path))
    A_al_c, B_al_c = align_images(A, B)  # I align in color then convert to gray
    A_al = to_gray_1ch(A_al_c); B_al = to_gray_1ch(B_al_c)

    image_saver(str(outdir/"a_aligned.png"), A_al)
    image_saver(str(outdir/"b_aligned.png"), B_al)

    A_low, B_high, hybrid, B_high_dbg = hybrid_maker_aligned(
        image_low=A_al, image_high_aligned=B_al,
        low_size=low_size, low_sigma=low_sigma,
        high_size=high_size, high_sigma=high_sigma
    )  # I keep explicit params to make results reproducible

    image_saver(str(outdir/"low.png"),  A_low)
    image_saver(str(outdir/"high.png"), B_high)
    image_saver(str(outdir/"hybrid.png"), hybrid)

    if do_fft:
        image_saver(str(outdir/"a_fft.png"),      fft_log_gray(A_al))  # I provide frequency views on one pair
        image_saver(str(outdir/"b_fft.png"),      fft_log_gray(B_al))
        image_saver(str(outdir/"a_low_fft.png"),  fft_log_gray(A_low))
        image_saver(str(outdir/"b_high_fft.png"), fft_log_gray(B_high_dbg))
        image_saver(str(outdir/"hybrid_fft.png"), fft_log_gray(hybrid))

def main():
    QOUT.mkdir(parents=True, exist_ok=True)

    _run_pair("cat.png",      "lion.png",    "cat_lion",      low_size=21, low_sigma=4, high_size=13, high_sigma=3)
    _run_pair("einstein.png", "marilyn.png", "einstein_mar",  low_size=21, low_sigma=4, high_size=17, high_sigma=2)
    _run_pair("person.png",   "old_man.png", "person_oldman", low_size=21, low_sigma=4, high_size=13, high_sigma=2, do_fft=True)
    _run_pair("derek.jpg",    "nutmeg.jpg",  "derek_nutmeg",  low_size=21, low_sigma=4, high_size=13, high_sigma=3)

    print("Part 2.2 outputs written to", QOUT)  

if __name__ == "__main__":
    main()
