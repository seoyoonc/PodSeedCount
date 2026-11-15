r"""Interactive review in the browser: check, correct and confirm seed counts.

  python scripts/review.py                      images from input_images/, results in output/
  python scripts/review.py D:\scans\batch3 -o D:\scans\batch3_results --port 8001

Opens http://localhost:8000. Every change is saved immediately (output/review/), so you can stop
(Ctrl+C) and continue later. "Save results" in the page writes the report, tables and annotated
images with your corrections into the output folder.
"""
import argparse
import logging
import sys
import threading
import webbrowser

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from seedcounter.config import DEFAULT_CONFIG, load_config
from seedcounter.review.server import serve
from seedcounter.review.session import ReviewSession

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default=str(ROOT / "input_images"), help="folder with .tif images")
    ap.add_argument("-o", "--output", default=str(ROOT / "output"), help="output folder (default: output)")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="settings file")
    ap.add_argument("--gt", default=str(ROOT / "ground_truth"), help="ground-truth folder used by 'Learn from reviews'")
    ap.add_argument("--keys", default=str(ROOT / "config" / "keybindings.yaml"),
                    help="keyboard shortcuts file (default: config/keybindings.yaml)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this computer (default); 0.0.0.0 = also others on your network")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        cfg = load_config(a.config)
        session = ReviewSession(a.input, a.output, cfg, a.config, a.gt)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if session.keys:
        print(f"Preparing {len(session.keys)} images (analysed images are reused from {a.output}/review/cache) ...")
    else:
        print(f"No images in {session.input_dir} yet: add them with 'Add images' in the page.")
    if session.keys:
        session.load_all((lambda xs: tqdm(xs, desc="Images", unit="img")) if tqdm else None)
    try:
        httpd = serve(session, a.host, a.port, a.keys)
    except OSError as exc:
        print(f"Error: cannot use port {a.port} ({exc}). Is the review already running? "
              f"Try another port, e.g. --port {a.port + 1}", file=sys.stderr)
        return 1
    url = f"http://localhost:{a.port}"
    print(f"\nReview is running at {url}\nChanges are saved as you go. Press Ctrl+C here to stop.\n")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your changes are saved; run this script again to continue.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
