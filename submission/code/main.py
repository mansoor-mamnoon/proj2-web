# main.py
import sys
import importlib
from pathlib import Path
from time import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

questions = ["q1_1", "q1_2", "q1_3", "q2_1", "q2_2", "q2_3", "q2_4"]

def run_one(mod_name: str) -> bool:
    """
    Import mod_name and call its main().
    Returns True if it succeeded, False if it failed.
    """
    try:
        mod = importlib.import_module(mod_name)     # import the file
        fn = getattr(mod, "main")                   # every part exposes main() in src files I have question scripts in
    except Exception as e:
        print(f"[import error] {mod_name}: {e}")
        return False

    try:
        t0 = time()
        fn()                                       
        dt = time() - t0
        print(f"[done] {mod_name} in {dt:.2f}s")
        return True
    except Exception as e:
        print(f"[run error] {mod_name}: {e}")
        return False

def main(argv):
    parts = argv[1:] if len(argv) > 1 else questions

    known = set(questions)
    unknown = [p for p in parts if p not in known]
    if unknown:
        print("[error] Unknown modules:", ", ".join(unknown))
        print("        Expected one or more of:", ", ".join(questions))
        sys.exit(2)

    print("[info] Running parts:", " ".join(parts))
    any_fail = False

    for name in parts:
        print(f"\n=== {name} ===")
        ok = run_one(name)
        if not ok:
            any_fail = True

    print("\n[summary]")
    if any_fail:
        print("Some parts failed. Check the messages above.")
        sys.exit(1)
    else:
        print("All requested parts finished successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)
