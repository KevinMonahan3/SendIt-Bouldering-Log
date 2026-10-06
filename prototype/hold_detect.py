"""
hold_detect.py - find all climbing holds of one colour on a still frame.

Issue #10 (SendIt FYP). Classic computer vision, no AI training:
    still frame -> blur -> BGR to HSV -> colour mask -> clean up -> contours -> filter by area -> boxes

Usage (from the prototype/ folder, with the venv active):
    python hold_detect.py ../footage/stills/2026-10-12_blue_V3_f00041.jpg

Two windows:
    "SendIt sliders"       sliders, the hue colour bar (bright part = selected colours)
                           and a short guide to what each slider does
    "SendIt view"          left: your photo with each detected hold outlined and numbered
                           right: the mask (white = pixels inside the colour range)

Controls:
    click a hold (view)    shows its H, S, V values and marks its hue on the colour bar
    p                      print the settings and a ready-to-paste CSV row
    s                      save the result image and mask to results/
    q or Esc               quit
"""
import sys
from pathlib import Path

import cv2
import numpy as np

SLIDERS = "SendIt sliders"               # window with the sliders and colour bar
VIEW = "SendIt view"                     # window with the result and mask
SLIDER_PANEL_WIDTH = 460
MAX_WIDTH = 640                        # each of the two images is shrunk to fit this box
MAX_HEIGHT = 560
RESULTS_DIR = Path(__file__).parent / "results"

# (slider name, max value, starting value)
# Names are short because Windows cuts off long slider labels.
TRACKBARS = [
    ("1 Hue min", 179, 0),             # colour range start
    ("2 Hue max", 179, 179),           # colour range end (min > max = wraps round, for red)
    ("3 Sat min", 255, 80),            # how vivid: raise to drop walls/greys/white
    ("4 Sat max", 255, 255),
    ("5 Bright min", 255, 50),         # how bright: raise to drop shadows/black
    ("6 Bright max", 255, 255),
    ("7 Clean-up", 10, 2),             # morphology kernel = 2 * value + 1 (0 = off)
    ("8 Min size", 100, 5),            # smallest hold kept, in 0.01% of the image
]

# Colours used for drawing (BGR)
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
PANEL_BG = (40, 40, 40)


# ---------------------------------------------------------------------------
# The pipeline: small functions, one job each (easy to test and to port to Dart)
# ---------------------------------------------------------------------------

def load_frame(path, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Read an image and shrink it to fit inside max_width x max_height."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def to_hsv(frame, blur_size=5):
    """Blur slightly to remove noise, then convert BGR -> HSV."""
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def make_mask(hsv, lower, upper):
    """White where the pixel is inside the HSV range, black everywhere else.

    If H low > H high the range wraps around the end of the hue circle
    (needed for red, which sits near both 0 and 179).
    """
    h_lo, s_lo, v_lo = lower
    h_hi, s_hi, v_hi = upper

    if h_lo <= h_hi:
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

    # Wrap-around: [h_lo .. 179] OR [0 .. h_hi]
    mask_top = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([179, s_hi, v_hi]))
    mask_bottom = cv2.inRange(hsv, np.array([0, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))
    return cv2.bitwise_or(mask_top, mask_bottom)


def clean_mask(mask, kernel_size=5):
    """Opening removes small specks; closing fills small holes (e.g. chalk) inside holds."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def find_holds(mask, min_area):
    """Find the outline of each white blob and keep the ones big enough to be holds.

    Returns (contours, boxes) where each box is (x, y, w, h).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holds = [c for c in contours if cv2.contourArea(c) >= min_area]
    boxes = [cv2.boundingRect(c) for c in holds]
    return holds, boxes


def draw_holds(frame, contours, boxes):
    """Outline each hold and number it, so you can count correct/missed/false positives."""
    out = frame.copy()
    cv2.drawContours(out, contours, -1, GREEN, 2)
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        put_label(out, str(i), (x, max(y - 4, 12)), scale=0.45)
    return out


# ---------------------------------------------------------------------------
# Display helpers (only used by the tuner, not needed in the app)
# ---------------------------------------------------------------------------

def put_label(img, text, org, scale=0.5, colour=WHITE):
    """Text on a small dark box so it's readable on any background."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + baseline), BLACK, -1)
    cv2.putText(img, text, org, font, scale, colour, 1, cv2.LINE_AA)


def hue_bar(width, h_lo, h_hi, clicked_h=None, height=26):
    """A strip showing every hue 0-179; the selected range is bright, the rest dimmed."""
    hues = np.linspace(0, 179, width).astype(np.uint8)
    hsv_strip = np.zeros((height, width, 3), np.uint8)
    hsv_strip[:, :, 0] = hues
    hsv_strip[:, :, 1] = 255
    hsv_strip[:, :, 2] = 255
    bar = cv2.cvtColor(hsv_strip, cv2.COLOR_HSV2BGR)

    if h_lo <= h_hi:
        selected = (hues >= h_lo) & (hues <= h_hi)
    else:                                   # wrap-around (red)
        selected = (hues >= h_lo) | (hues <= h_hi)
    bar[:, ~selected] = (bar[:, ~selected] * 0.25).astype(np.uint8)

    if clicked_h is not None:               # mark the hue of the last clicked pixel
        x = int(clicked_h / 179 * (width - 1))
        cv2.line(bar, (x, 0), (x, height - 1), WHITE, 2)
    return bar


def info_panel(width, lines, height=None):
    """A dark strip with a few lines of text."""
    line_h = 20
    height = height or line_h * len(lines) + 10
    panel = np.full((height, width, 3), PANEL_BG, np.uint8)
    for i, text in enumerate(lines):
        y = 18 + i * line_h
        if isinstance(text, tuple):         # (name, description) -> two columns
            put_label(panel, text[0], (8, y), scale=0.45, colour=YELLOW)
            put_label(panel, text[1], (120, y), scale=0.45)
        elif text:                          # skip blank lines
            put_label(panel, text, (8, y), scale=0.45)
    return panel


def slider_panel(width, settings, clicked):
    """Image shown under the sliders: the hue colour bar plus a guide to each slider."""
    h_lo, h_hi = settings["h_lo"], settings["h_hi"]
    wrap = " (wraps round - red)" if h_lo > h_hi else ""
    title = info_panel(width, [f"Hue range: {h_lo}-{h_hi}{wrap}"])
    bar = hue_bar(width, h_lo, h_hi, clicked["hsv"][0] if clicked else None, height=34)

    # Tick marks under the bar so you can read hue values off it
    ticks = np.full((18, width, 3), PANEL_BG, np.uint8)
    for hue in range(0, 180, 30):
        x = int(hue / 179 * (width - 1))
        cv2.line(ticks, (x, 0), (x, 4), WHITE, 1)
        put_label(ticks, str(hue), (min(x + 2, width - 26), 15), scale=0.38)

    if clicked is not None:
        ch, cs, cv = clicked["hsv"]
        click_text = f"Last click: H={ch}  S={cs}  V={cv}  (white line on bar)"
    else:
        click_text = "Click a hold in the view window to see its H S V"

    guide = info_panel(width, [
        click_text,
        "",
        ("1-2 Hue", "which colour - use the bar above"),
        ("3 Sat min", "raise to remove wall / grey / white"),
        ("4 Sat max", "leave at 255"),
        ("5 Bright min", "raise to remove shadows; lower if"),
        ("", "holds in shadow go missing"),
        ("6 Bright max", "leave at 255 (lower for glare)"),
        ("7 Clean-up", "raise to remove specks; lower if"),
        ("", "holds merge or footholds vanish"),
        ("8 Min size", f"smallest hold kept ({settings['min_area']:.0f}px now)"),
    ])
    return np.vstack([title, bar, ticks, guide])


def build_view(result, mask, settings, n_holds, clicked):
    """Main view: [result | mask] side by side, with a short info strip underneath."""
    h, w = result.shape[:2]
    left = result.copy()
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    put_label(left, f"RESULT: {n_holds} holds", (8, 22), scale=0.6, colour=YELLOW)
    put_label(right, "MASK (white = in range)", (8, 22), scale=0.6, colour=YELLOW)

    if clicked is not None:                 # small cross where you clicked
        cx, cy = clicked["xy"]
        cv2.drawMarker(left, (cx, cy), YELLOW, cv2.MARKER_CROSS, 14, 2)

    images = np.hstack([left, np.full((h, 6, 3), PANEL_BG, np.uint8), right])
    info = info_panel(images.shape[1], [
        f"Hue {settings['h_lo']}-{settings['h_hi']}  Sat {settings['s_lo']}-{settings['s_hi']}  "
        f"Bright {settings['v_lo']}-{settings['v_hi']}  Clean-up {settings['kernel']}  "
        f"Min size {settings['min_area']:.0f}px",
        "click a hold = H S V    p = print CSV row    s = save    q = quit",
    ])
    return np.vstack([images, info])


# ---------------------------------------------------------------------------
# The interactive tuner: two windows
#   "SendIt sliders" - the sliders, colour bar and a guide to each slider
#   "SendIt view"    - result and mask side by side (click holds here)
# ---------------------------------------------------------------------------

def create_windows(view_x):
    cv2.namedWindow(SLIDERS, cv2.WINDOW_AUTOSIZE)
    for name, max_val, start in TRACKBARS:
        cv2.createTrackbar(name, SLIDERS, start, max_val, lambda v: None)
    cv2.namedWindow(VIEW, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(SLIDERS, 0, 0)
    cv2.moveWindow(VIEW, view_x, 0)


def read_settings(image_area):
    v = {name: cv2.getTrackbarPos(name, SLIDERS) for name, _, _ in TRACKBARS}
    return {
        "h_lo": v["1 Hue min"], "h_hi": v["2 Hue max"],
        "s_lo": v["3 Sat min"], "s_hi": v["4 Sat max"],
        "v_lo": v["5 Bright min"], "v_hi": v["6 Bright max"],
        "kernel": 2 * v["7 Clean-up"] + 1 if v["7 Clean-up"] > 0 else 0,
        "min_area": v["8 Min size"] / 10000 * image_area,
    }


def window_closed(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python hold_detect.py <path to still image>")
        sys.exit(1)

    path = Path(sys.argv[1])
    frame = load_frame(path)
    hsv = to_hsv(frame)
    img_h, img_w = frame.shape[:2]
    image_area = img_h * img_w
    print(f"Loaded {path.name}: {img_w}x{img_h} after resizing")

    state = {"clicked": None}

    def on_click(event, x, y, flags, param):
        # Only clicks on the left image (the photo) count
        if event == cv2.EVENT_LBUTTONDOWN and x < img_w and y < img_h:
            h, s, v = (int(n) for n in hsv[y, x])
            state["clicked"] = {"xy": (x, y), "hsv": (h, s, v)}
            print(f"Clicked ({x}, {y}) -> H={h} S={s} V={v}")

    create_windows(view_x=SLIDER_PANEL_WIDTH + 30)
    cv2.setMouseCallback(VIEW, on_click)

    while True:
        s = read_settings(image_area)
        lower = (s["h_lo"], s["s_lo"], s["v_lo"])
        upper = (s["h_hi"], s["s_hi"], s["v_hi"])

        mask = make_mask(hsv, lower, upper)
        mask = clean_mask(mask, s["kernel"])
        contours, boxes = find_holds(mask, s["min_area"])
        result = draw_holds(frame, contours, boxes)

        cv2.imshow(SLIDERS, slider_panel(SLIDER_PANEL_WIDTH, s, state["clicked"]))
        cv2.imshow(VIEW, build_view(result, mask, s, len(boxes), state["clicked"]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("p"):
            print(f"lower={lower} upper={upper} kernel={s['kernel']} "
                  f"min_area={s['min_area']:.0f}px -> {len(boxes)} holds")
            row = (f"{path.stem},COLOUR,{lower[0]},{upper[0]},{lower[1]},{lower[2]},"
                   f"{s['kernel']},{round(s['min_area'])},,,,,")
            print("CSV row:")
            print(row)
        elif key == ord("s"):
            RESULTS_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_result.jpg"), result)
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_mask.jpg"), mask)
            print(f"Saved to {RESULTS_DIR}")

        # Stop if either window was closed with the X button
        if window_closed(VIEW) or window_closed(SLIDERS):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
