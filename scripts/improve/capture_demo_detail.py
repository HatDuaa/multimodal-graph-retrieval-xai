"""Render the detail panel of the demo's top hit for a query, as the app shows it, to a JPEG.

Asks a running demo (python app/demo_app.py ...) for the top hit through its /run API, wraps the returned detail HTML
in the app's own CSS and click script, optionally pins one table row (the state after clicking it), and screenshots
the page with headless Chrome. The page holds exactly the markup the app sends; only the Gradio frame is left out.

    python scripts/improve/capture_demo_detail.py --port 7861 --name frisbee \
        --query "Two dogs playing with a frisbee on the grass" --pin 3
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.demo_app import CSS  # noqa: E402
from src.explain.match_view import FOCUS_JS  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def page(detail: str, pin: int | None) -> str:
    click = f"document.querySelector('#detail tr[data-row=\"{pin}\"] td:nth-child(2)').click();" if pin else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; color: #1f2937; background: #fff; margin: 0; }}
{CSS}
html, body {{ height: auto !important; overflow: visible !important; }}
#detail {{ width: 650px; padding: 10px 14px; }}
#detail .mx-fig {{ position: static; }}
#detail .mx-image {{ max-height: 380px; }}
</style></head><body><div id="detail">{detail}</div>
<script>const element = document.querySelector('#detail');{FOCUS_JS}{click}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--query", required=True)
    ap.add_argument("--name", required=True, help="output file stem")
    ap.add_argument("--pin", type=int, default=None, help="table row to pin, as after a click on it")
    ap.add_argument("--out", default=str(ROOT / "experiments/improve/demo/node-matching"))
    ap.add_argument("--chrome", default=CHROME)
    args = ap.parse_args()

    from gradio_client import Client
    detail = Client(f"http://127.0.0.1:{args.port}/", verbose=False).predict(
        args.query, "clip_graph", 12, None, api_name="/run")[2]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        html, png = Path(tmp) / "detail.html", Path(tmp) / "detail.png"
        html.write_text(page(detail, args.pin), encoding="utf-8")
        subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--screenshot={png}",
                        "--window-size=680,3000", "--force-device-scale-factor=1.5", "--virtual-time-budget=2000",
                        html.as_uri()], check=True, capture_output=True)
        image = Image.open(png).convert("RGB")
    # crop the white space under the panel
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    image = image.crop((0, 0, image.width, min(image.height, box[3] + 20)))
    target = out / f"{args.name}.jpg"
    image.save(target, quality=85, optimize=True)
    print(target, image.size, target.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
