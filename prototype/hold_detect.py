"""
hold_detect.py - find all climbing holds of one colour on a still frame.

Issue #10 (SendIt FYP). Classic computer vision, no AI training:
    still frame -> blur -> BGR to HSV -> colour mask -> clean up -> contours -> filter by area -> boxes

Usage (from the prototype/ folder, with the venv active):
    python hold_detect.py ../footage/stills/2026-10-12_blue_V3_f00041.jpg

Two windows:
    "SendIt controls"   six sliders, each with its name above it and a one-line hint below.
                        Drag the white handles, or use the keyboard (below).
                        The colour sliders show the colours they select:
                        bright = kept, dark = removed.
    "SendIt view"       left: your photo with each detected hold outlined and numbered
                        right: the mask (white = pixels inside the colour range)

Mouse in the view window:
    click a hold        shows its H, S, V values; a cyan marker shows where it sits on each slider
                        (in TAP MODE the click also sets the colour sliders for you - issue #11)
    drag                pans the image when zoomed in (Ubuntu); does NOT pick a pixel

Keyboard (either window):
    Up / Down  or Tab   choose which handle to move (the selected handle turns yellow)
    Left / Right        move the selected handle by 1
    a / d               same as Left / Right (if the arrow keys don't work)
    A / D (with Shift)  move by 10
    t                   TAP MODE on/off: click one hold and the colour range is built from it
                        Ctrl+click adds more holds (e.g. start, top and the biggest hold):
                        the range then covers the tapped holds with a smaller margin, max
                        hold size is set from the biggest one and min hold size from the
                        smallest. A plain click clears them and starts again.
    [ / ]               in tap mode: narrower / wider hue tolerance (re-applies the taps)
    p                   print the settings and a ready-to-paste CSV row
    s                   save the result image and mask to results/
    r                   reset the sliders
    q or Esc            quit
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Arrow key codes from cv2.waitKeyEx: (Windows, Linux)
KEYS_LEFT = {2424832, 65361}
KEYS_UP = {2490368, 65362}
KEYS_RIGHT = {2555904, 65363}
KEYS_DOWN = {2621440, 65364}

CONTROLS_WINDOW = "SendIt controls"
VIEW_WINDOW = "SendIt view"
MAX_WIDTH = 640                        # the photo is shrunk to fit this box
MAX_HEIGHT = 560
RESULTS_DIR = Path(__file__).parent / "results"

# Tap-to-sample (issue #11)
TAP_PATCH = 7                          # sample a 7x7 square around the click
TAP_H_TOL = 10                         # hue range = sampled hue +/- this
TAP_S_TOL = 60                         # vividness: keep pixels down to sampled S - this
TAP_V_TOL = 80                         # brightness: keep pixels down to sampled V - this
# With 2+ taps (Ctrl+click) you've measured how much the route's colour varies, so the
# extra margin around the tapped holds can be much smaller than for a single tap.
TAP_MULTI_H_TOL = 3                    # hue: tapped hues +/- this
TAP_MULTI_S_TOL = 30                   # vividness: keep down to the dullest tapped hold - this
TAP_MULTI_V_TOL = 40                   # brightness: keep down to the darkest tapped hold - this
TAP_MAX_MARGIN = 1.15                  # max hold size = biggest tapped hold x this
TAP_MIN_FACTOR = 0.1                   # min hold size = smallest tapped hold x this (footholds)

# Colours used for drawing (BGR)
GREEN = (0, 220, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
GREY = (170, 170, 170)
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


def sample_patch(hsv, x, y, size=TAP_PATCH):
    """Average HSV colour of a small square around (x, y).

    One pixel can be chalk, a shadow or the edge of the hold, so a small patch
    gives a more reliable colour. Hue is an angle on a circle (179 sits next to 0),
    so it is averaged as an angle; otherwise red at 2 and 178 would average to 90 (green).
    S and V use the median, which ignores the odd very light or dark pixel.
    """
    half = size // 2
    img_h, img_w = hsv.shape[:2]
    patch = hsv[max(y - half, 0):min(y + half + 1, img_h),
                max(x - half, 0):min(x + half + 1, img_w)].reshape(-1, 3)

    angles = patch[:, 0].astype(np.float64) * 2 * np.pi / 180      # 0-179 -> 0-2pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    h = int(round(mean_angle * 180 / (2 * np.pi))) % 180
    s = int(np.median(patch[:, 1]))
    v = int(np.median(patch[:, 2]))
    return h, s, v


def range_from_samples(samples, h_tol=TAP_H_TOL, s_tol=TAP_S_TOL, v_tol=TAP_V_TOL):
    """Build one HSV range that covers every tapped hold.

    samples is a list of (h, s, v), one per tapped hold (from sample_patch).
    Hue: the centre is the average of the tapped hues, worked out as an angle like in
    sample_patch, so red holds at 2 and 177 give a centre of 0, not 90. The range then
    reaches out to the tapped hue furthest from the centre, plus h_tol. It wraps round
    (% 180), which make_mask already handles.
    S and V: only a lower limit, below the dullest / darkest tapped hold. Shadow and
    distance make holds duller and darker, never more vivid, so the upper limit stays 255.
    With one sample this gives exactly the same range as before.
    """
    hues = np.array([sample[0] for sample in samples], dtype=np.float64)
    angles = hues * 2 * np.pi / 180
    centre = int(round(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
                       * 180 / (2 * np.pi))) % 180

    # Distance from the centre to each hue, the short way round the circle
    distances = [min(abs(h - centre), 180 - abs(h - centre)) for h in hues]
    half_width = min(int(max(distances)) + h_tol, 89)     # 89 = almost every hue

    lower = ((centre - half_width) % 180,
             max(min(sample[1] for sample in samples) - s_tol, 0),
             max(min(sample[2] for sample in samples) - v_tol, 0))
    upper = ((centre + half_width) % 180, 255, 255)
    return lower, upper


def range_from_sample(h, s, v, h_tol=TAP_H_TOL, s_tol=TAP_S_TOL, v_tol=TAP_V_TOL):
    """Range from a single tapped hold (same as range_from_samples with one sample)."""
    return range_from_samples([(h, s, v)], h_tol, s_tol, v_tol)


def hold_area_at(mask, x, y):
    """Area (in pixels) of the blob under (x, y), or None if (x, y) isn't on a blob.

    Uses the same outer outlines and contourArea as find_holds, so the number can be
    compared directly with its min/max area limits. A tap in a hole of the hold
    (e.g. a bolt hole) still counts, because the outer outline goes round it.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0:
            return cv2.contourArea(c)
    return None


def clean_mask(mask, kernel_size=5):
    """Opening removes small specks; closing fills small holes (e.g. chalk) inside holds."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def find_holds(mask, min_area, max_area=None):
    """Find the outline of each white blob and keep the ones that are hold-sized.

    Blobs smaller than min_area are noise; blobs bigger than max_area are usually
    wall panels or volumes of the same colour. max_area=None means no upper limit.
    Returns (contours, boxes) where each box is (x, y, w, h).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holds = [c for c in contours
             if cv2.contourArea(c) >= min_area
             and (max_area is None or cv2.contourArea(c) <= max_area)]
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
# Drawing helpers (only used by the tuner)
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


def put_text(img, text, org, scale=0.5, colour=WHITE, thickness=1, align="left"):
    """Plain text (for the dark control panel)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if align == "right":
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (org[0] - tw, org[1])
    cv2.putText(img, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def info_panel(width, lines):
    """A dark strip with a few lines of text."""
    panel = np.full((20 * len(lines) + 10, width, 3), PANEL_BG, np.uint8)
    for i, text in enumerate(lines):
        put_label(panel, text, (8, 18 + i * 20), scale=0.45)
    return panel


# ---------------------------------------------------------------------------
# The control panel: sliders drawn by us, so each one sits right under its label
# ---------------------------------------------------------------------------

PANEL_W = 560
TRACK_X0, TRACK_X1 = 24, PANEL_W - 24   # left/right end of every slider track
TRACK_H = 22
ROW_H = 78                              # height of one slider row
TOP = 46                                # space for the title

# Each control: label, hint, max value, starting value(s), track style.
# Two values = a range slider with a min handle and a max handle.
CONTROLS = [
    {"key": "hue", "label": "1  COLOUR (hue)", "max": 179, "start": [0, 179], "style": "hue",
     "hint": "Put the two handles around your route's colour"},
    {"key": "sat", "label": "2  VIVIDNESS (saturation)", "max": 255, "start": [80, 255], "style": "sat",
     "hint": "Drag the left handle right to remove wall / grey / white"},
    {"key": "val", "label": "3  BRIGHTNESS (value)", "max": 255, "start": [50, 255], "style": "val",
     "hint": "Left handle right = remove shadows. Back left if dark holds vanish"},
    {"key": "clean", "label": "4  CLEAN-UP", "max": 10, "start": [2], "style": "plain",
     "hint": "Right = remove specks. Back left if holds merge or footholds vanish"},
    {"key": "size", "label": "5  MIN HOLD SIZE", "max": 100, "start": [5], "style": "plain",
     "hint": "Right = ignore small blobs. Back left if small holds get dropped"},
    {"key": "maxsize", "label": "6  MAX HOLD SIZE", "max": 500, "start": [500], "style": "plain",
     "hint": "Left = ignore big blobs (wall panels). Back right if big holds vanish"},
]
MAXSIZE_STEP = 0.0001       # one step of the max-size slider = 0.01% of the image
                            # (all the way right = no limit; 499 steps = 4.99%)

# Every handle, in the order Up/Down/Tab move through them: (control index, handle index)
HANDLES = [(i, h) for i, c in enumerate(CONTROLS) for h in range(len(c["start"]))]


class ControlPanel:
    def __init__(self):
        self.values = {c["key"]: list(c["start"]) for c in CONTROLS}
        self.dragging = None            # (control index, handle index) while the mouse is held
        self.selected = HANDLES[0]      # handle moved by the keyboard
        self.source = "manual"          # "manual" or "tap" - goes into the CSV notes

    def reset(self):
        self.values = {c["key"]: list(c["start"]) for c in CONTROLS}
        self.source = "manual"

    def apply_range(self, lower, upper):
        """Move the colour handles to a range built by tap-to-sample."""
        self.values["hue"] = [lower[0], upper[0]]
        self.values["sat"] = [lower[1], upper[1]]
        self.values["val"] = [lower[2], upper[2]]
        self.source = "tap"

    # --- geometry -----------------------------------------------------------
    @staticmethod
    def track_top(i):
        return TOP + i * ROW_H + 30

    @staticmethod
    def value_to_x(value, max_value):
        return int(TRACK_X0 + value / max_value * (TRACK_X1 - TRACK_X0))

    @staticmethod
    def x_to_value(x, max_value):
        x = min(max(x, TRACK_X0), TRACK_X1)
        return round((x - TRACK_X0) / (TRACK_X1 - TRACK_X0) * max_value)

    # --- settings used by the pipeline --------------------------------------
    def settings(self, image_area):
        v = self.values
        clean = v["clean"][0]
        return {
            "h_lo": v["hue"][0], "h_hi": v["hue"][1],
            "s_lo": v["sat"][0], "s_hi": v["sat"][1],
            "v_lo": v["val"][0], "v_hi": v["val"][1],
            "kernel": 2 * clean + 1 if clean > 0 else 0,
            "min_area": v["size"][0] / 10000 * image_area,
            "max_area": self.max_area(image_area),
        }

    def max_area(self, image_area):
        v = self.values["maxsize"][0]
        return None if v >= CONTROLS[-1]["max"] else v * MAXSIZE_STEP * image_area

    def set_min_area(self, min_px, image_area):
        """Move the min-size handle so blobs smaller than min_px are ignored."""
        step = int(min_px / (image_area / 10000))                 # round down: keep that hold
        size_max = next(c["max"] for c in CONTROLS if c["key"] == "size")
        self.values["size"] = [max(0, min(step, size_max))]

    def set_max_area(self, max_px, image_area):
        """Move the max-size handle so it allows blobs up to max_px (None = no limit)."""
        top = CONTROLS[-1]["max"]
        if max_px is None:
            self.values["maxsize"] = [top]
        else:
            step = int(np.ceil(max_px / (MAXSIZE_STEP * image_area)))   # round up: keep that hold
            self.values["maxsize"] = [min(step, top)]

    # --- keyboard -----------------------------------------------------------
    def select_next(self, step):
        i = HANDLES.index(self.selected)
        self.selected = HANDLES[(i + step) % len(HANDLES)]

    def nudge(self, amount):
        i, handle = self.selected
        c = CONTROLS[i]
        vals = self.values[c["key"]]
        vals[handle] = min(max(vals[handle] + amount, 0), c["max"])
        if c["style"] in ("hue", "sat", "val"):
            self.source = "manual"      # tap range was changed by hand

    # --- mouse --------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, c in enumerate(CONTROLS):
                top = self.track_top(i)
                if top - 16 <= y <= top + TRACK_H + 16 and TRACK_X0 - 14 <= x <= TRACK_X1 + 14:
                    vals = self.values[c["key"]]
                    handle = 0
                    if len(vals) == 2:      # pick the handle nearest the click
                        d0 = abs(self.value_to_x(vals[0], c["max"]) - x)
                        d1 = abs(self.value_to_x(vals[1], c["max"]) - x)
                        if d1 < d0 or (d1 == d0 and x > self.value_to_x(vals[1], c["max"])):
                            handle = 1
                    self.dragging = (i, handle)
                    self.selected = (i, handle)     # keyboard now moves this handle too
                    self._set(x)
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._set(x)
            else:                           # button was released outside the window
                self.dragging = None
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def _set(self, x):
        i, handle = self.dragging
        c = CONTROLS[i]
        self.values[c["key"]][handle] = self.x_to_value(x, c["max"])
        if c["style"] in ("hue", "sat", "val"):
            self.source = "manual"      # tap range was changed by hand

    # --- drawing ------------------------------------------------------------
    def _hue_centre(self):
        lo, hi = self.values["hue"]
        if lo <= hi:
            return (lo + hi) // 2
        return ((lo + hi + 180) // 2) % 180   # middle of a wrapped (red) range

    def _track_image(self, c):
        """The coloured background of a slider track."""
        n = TRACK_X1 - TRACK_X0 + 1
        ramp = np.linspace(0, c["max"], n)
        hsv = np.zeros((TRACK_H, n, 3), np.uint8)
        hc = self._hue_centre()
        if c["style"] == "hue":
            hsv[..., 0], hsv[..., 1], hsv[..., 2] = ramp, 255, 255
        elif c["style"] == "sat":
            hsv[..., 0], hsv[..., 1], hsv[..., 2] = hc, ramp, 230
        elif c["style"] == "val":
            hsv[..., 0], hsv[..., 1], hsv[..., 2] = hc, 200, ramp
        else:
            return np.full((TRACK_H, n, 3), (90, 90, 90), np.uint8), ramp
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), ramp

    def _value_text(self, c, image_area):
        vals = self.values[c["key"]]
        if c["key"] == "hue":
            text = f"{vals[0]} - {vals[1]}"
            return text + "  (wraps round = red)" if vals[0] > vals[1] else text
        if c["key"] in ("sat", "val"):
            return f"{vals[0]} - {vals[1]}"
        if c["key"] == "clean":
            return "off" if vals[0] == 0 else f"{vals[0]}  (kernel {2 * vals[0] + 1})"
        if c["key"] == "maxsize":
            if vals[0] >= c["max"]:
                return "no limit"
            return f"{vals[0] * MAXSIZE_STEP * 100:.2f}% of image = {vals[0] * MAXSIZE_STEP * image_area:.0f} px"
        px = vals[0] / 10000 * image_area
        return f"{vals[0] / 100:.2f}% of image = {px:.0f} px"

    def draw(self, image_area, clicked_hsv=None, tap_mode=False, tap_samples=None):
        height = TOP + ROW_H * len(CONTROLS) + 92
        panel = np.full((height, PANEL_W, 3), PANEL_BG, np.uint8)
        if tap_mode:
            cv2.rectangle(panel, (0, 0), (PANEL_W, TOP - 8), (0, 90, 120), -1)
            put_text(panel, "TAP MODE: click a hold   Ctrl+click = add a hold", (TRACK_X0, 28), 0.55, YELLOW, 2)
        else:
            put_text(panel, "Drag the white handles", (TRACK_X0, 28), 0.6, WHITE, 1)
            put_text(panel, "bright = kept   dark = removed", (TRACK_X1, 28), 0.45, GREY, align="right")

        for i, c in enumerate(CONTROLS):
            row_top = TOP + i * ROW_H
            top = self.track_top(i)
            vals = self.values[c["key"]]

            # label (left) and current value (right), hint underneath the track
            is_selected_row = self.selected[0] == i
            if is_selected_row:                 # arrow in the margin shows the keyboard row
                put_text(panel, ">", (6, row_top + 20), 0.5, YELLOW, 2)
            put_text(panel, c["label"], (TRACK_X0, row_top + 20), 0.5, YELLOW, 2 if is_selected_row else 1)
            put_text(panel, self._value_text(c, image_area), (TRACK_X1, row_top + 20), 0.5, WHITE,
                     align="right")
            put_text(panel, c["hint"], (TRACK_X0, top + TRACK_H + 18), 0.42, GREY)

            # track: dim the parts that are removed
            track, ramp = self._track_image(c)
            if len(vals) == 2:
                lo, hi = vals
                kept = (ramp >= lo) & (ramp <= hi) if lo <= hi else (ramp >= lo) | (ramp <= hi)
            elif c["key"] == "maxsize":         # max size: blobs up to the handle are kept
                kept = ramp <= vals[0]
                track[:, kept] = (200, 200, 200)
            else:
                kept = ramp <= vals[0]      # plain slider: fill up to the handle
                track[:, kept] = (200, 200, 200)
            track[:, ~kept] = (track[:, ~kept] * 0.3).astype(np.uint8)
            panel[top:top + TRACK_H, TRACK_X0:TRACK_X1 + 1] = track
            cv2.rectangle(panel, (TRACK_X0 - 1, top - 1), (TRACK_X1 + 1, top + TRACK_H), (90, 90, 90), 1)

            # cyan markers: where each tapped hold (or the last clicked pixel) sits on this slider
            markers = tap_samples if tap_samples else ([clicked_hsv] if clicked_hsv is not None else [])
            if c["style"] in ("hue", "sat", "val"):
                for sample in markers:
                    value = sample[("hue", "sat", "val").index(c["style"])]
                    mx = self.value_to_x(value, c["max"])
                    y = top + TRACK_H + 2
                    pts = np.array([[mx, y], [mx - 6, y + 9], [mx + 6, y + 9]], np.int32)
                    cv2.fillPoly(panel, [pts], CYAN)

            # handles (the one the keyboard moves is yellow)
            for h, v in enumerate(vals):
                hx = self.value_to_x(v, c["max"])
                fill = YELLOW if self.selected == (i, h) else WHITE
                cv2.rectangle(panel, (hx - 5, top - 6), (hx + 5, top + TRACK_H + 6), fill, -1)
                cv2.rectangle(panel, (hx - 5, top - 6), (hx + 5, top + TRACK_H + 6), BLACK, 1)

        # footer: last click and keys
        y = TOP + ROW_H * len(CONTROLS) + 18
        if tap_samples and len(tap_samples) > 1:
            hues = ", ".join(str(sample[0]) for sample in tap_samples)
            put_text(panel, f"{len(tap_samples)} holds tapped: H = {hues}   (cyan markers)",
                     (TRACK_X0, y), 0.5, CYAN)
        elif clicked_hsv is not None:
            h, s, v = clicked_hsv
            what = "Tap sample" if self.source == "tap" else "Last click"
            put_text(panel, f"{what}: H={h}  S={s}  V={v}   (cyan markers)", (TRACK_X0, y), 0.5, CYAN)
        else:
            put_text(panel, "Click a hold in the view window to see where it sits on each slider",
                     (TRACK_X0, y), 0.45, CYAN)
        put_text(panel, "Up/Down or Tab = pick handle (yellow)   Left/Right or a/d = move 1   A/D = 10",
                 (TRACK_X0, y + 28), 0.42, GREY)
        put_text(panel, "t = tap mode   [ ] = tap hue +/-   p = CSV row   s = save   r = reset   q = quit",
                 (TRACK_X0, y + 50), 0.42, GREY)
        return panel


# ---------------------------------------------------------------------------
# The view window: result and mask side by side
# ---------------------------------------------------------------------------

def build_view(result, mask, settings, n_holds, clicked, tap_mode=False, taps=None):
    h, w = result.shape[:2]
    left = result.copy()
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    put_label(left, f"RESULT: {n_holds} holds", (8, 22), scale=0.6, colour=YELLOW)
    put_label(right, "MASK (white = in range)", (8, 22), scale=0.6, colour=YELLOW)

    if clicked is not None:                 # small cross where you clicked
        cx, cy = clicked["xy"]
        cv2.drawMarker(left, (cx, cy), CYAN, cv2.MARKER_CROSS, 14, 2)
        cv2.drawMarker(right, (cx, cy), CYAN, cv2.MARKER_CROSS, 14, 2)

    half = TAP_PATCH // 2
    for n, tap in enumerate(taps or [], start=1):   # each tapped hold: sampled square + its number
        tx, ty = tap["xy"]
        cv2.rectangle(left, (tx - half, ty - half), (tx + half, ty + half), CYAN, 1)
        if len(taps) > 1:
            put_label(left, f"T{n}", (tx + half + 2, ty + 4), scale=0.4, colour=CYAN)

    images = np.hstack([left, np.full((h, 6, 3), PANEL_BG, np.uint8), right])
    info = info_panel(images.shape[1], [
        f"Hue {settings['h_lo']}-{settings['h_hi']}  Sat {settings['s_lo']}-{settings['s_hi']}  "
        f"Bright {settings['v_lo']}-{settings['v_hi']}  Clean-up {settings['kernel']}  "
        f"Size {settings['min_area']:.0f}-"
        f"{'any' if settings['max_area'] is None else format(settings['max_area'], '.0f')}px",
        ("TAP MODE: click = one hold   Ctrl+click = add a hold   (t = off)" if tap_mode
         else "click a hold = show H S V    t = tap mode    drag = move when zoomed in"),
    ])
    return np.vstack([images, info])


class ViewClicks:
    """Tell a click (sample the pixel) apart from a drag (pan when zoomed in).

    The pixel is only sampled when the button is released and the mouse barely moved,
    so dragging to pan never picks a new colour.
    """
    MAX_MOVES = 3                           # mouse-move events allowed during a click
    MAX_DISTANCE = 5                        # pixels

    def __init__(self, hsv, img_w, img_h, gap=6):
        self.hsv, self.img_w, self.img_h, self.gap = hsv, img_w, img_h, gap
        self.press = None
        self.clicked = None
        self.new_click = False              # main loop checks this for tap mode

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press = {"xy": (x, y), "moves": 0, "ctrl": bool(flags & cv2.EVENT_FLAG_CTRLKEY)}
        elif event == cv2.EVENT_MOUSEMOVE and self.press and flags & cv2.EVENT_FLAG_LBUTTON:
            self.press["moves"] += 1
        elif event == cv2.EVENT_LBUTTONUP and self.press:
            px, py = self.press["xy"]
            still = (self.press["moves"] <= self.MAX_MOVES
                     and abs(x - px) <= self.MAX_DISTANCE and abs(y - py) <= self.MAX_DISTANCE)
            # Ctrl held at press or release counts as Ctrl+click
            ctrl = self.press["ctrl"] or bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            self.press = None
            if still:
                self._sample(x, y, ctrl)

    def _sample(self, x, y, ctrl=False):
        if x >= self.img_w + self.gap:      # clicked on the mask side: same pixel
            x -= self.img_w + self.gap
        if 0 <= x < self.img_w and 0 <= y < self.img_h:
            h, s, v = (int(n) for n in self.hsv[y, x])
            self.clicked = {"xy": (x, y), "hsv": (h, s, v), "add": ctrl}
            self.new_click = True
            print(f"Clicked ({x}, {y}) -> H={h} S={s} V={v}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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

    controls = ControlPanel()
    clicks = ViewClicks(hsv, img_w, img_h)

    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(CONTROLS_WINDOW, 0, 0)
    cv2.moveWindow(VIEW_WINDOW, PANEL_W + 30, 0)
    cv2.setMouseCallback(CONTROLS_WINDOW, controls.on_mouse)
    cv2.setMouseCallback(VIEW_WINDOW, clicks.on_mouse)

    tap_mode = False
    h_tol = TAP_H_TOL                       # hue tolerance for one tap ([ ] changes it)
    multi_h_tol = TAP_MULTI_H_TOL           # hue tolerance for 2+ taps ([ ] changes it)
    taps = []                               # tapped holds: {"xy": (x, y), "hsv": (h, s, v)}
    sizes_from_taps = False                 # True while min/max size were set by Ctrl+click taps
    start_sizes = None                      # min/max slider values before the multi-tap

    def apply_taps():
        """Build the colour range from every tapped hold; with 2+ taps also set min/max size."""
        nonlocal sizes_from_taps, start_sizes
        samples = [tap["hsv"] for tap in taps]
        if len(taps) == 1:
            lower, upper = range_from_samples(samples, h_tol=h_tol)
        else:
            lower, upper = range_from_samples(samples, h_tol=multi_h_tol,
                                              s_tol=TAP_MULTI_S_TOL, v_tol=TAP_MULTI_V_TOL)
        controls.apply_range(lower, upper)
        print(f"{len(taps)} tap(s): " + "  ".join(f"T{n} H={h} S={s} V={v}"
                                                for n, (h, s, v) in enumerate(samples, start=1)))
        print(f"  -> lower={lower} upper={upper} "
              f"(hue tolerance {h_tol if len(taps) == 1 else multi_h_tol})")

        if len(taps) >= 2:
            # Min and max hold size from the smallest and biggest tapped holds, measured
            # on the new mask (so they match what find_holds will see)
            if not sizes_from_taps:         # remember the hand-set sizes to restore later
                start_sizes = (list(controls.values["size"]), list(controls.values["maxsize"]))
            s = controls.settings(image_area)
            new_mask = clean_mask(make_mask(hsv, lower, upper), s["kernel"])
            areas = [hold_area_at(new_mask, *tap["xy"]) for tap in taps]
            for n, area in enumerate(areas, start=1):
                if area is None:
                    print(f"  T{n} is not on a detected blob - try tapping the middle of the hold")
            found = [area for area in areas if area is not None]
            if found:
                biggest, smallest = max(found), min(found)
                controls.set_max_area(biggest * TAP_MAX_MARGIN, image_area)
                controls.set_min_area(smallest * TAP_MIN_FACTOR, image_area)
                sizes_from_taps = True
                print(f"  tapped holds {smallest:.0f}-{biggest:.0f}px -> hold size "
                      f"{smallest * TAP_MIN_FACTOR:.0f}-{biggest * TAP_MAX_MARGIN:.0f}px")
        elif sizes_from_taps:               # back to one tap: put the sizes back as they were
            controls.values["size"], controls.values["maxsize"] = start_sizes
            sizes_from_taps = False

    while True:
        # Tap-to-sample: click = start again with this hold, Ctrl+click = add this hold
        if clicks.new_click:
            clicks.new_click = False
            if tap_mode:
                x, y = clicks.clicked["xy"]
                tap = {"xy": (x, y), "hsv": sample_patch(hsv, x, y)}
                taps = taps + [tap] if clicks.clicked["add"] else [tap]
                clicks.clicked["hsv"] = tap["hsv"]
                apply_taps()

        s = controls.settings(image_area)
        lower = (s["h_lo"], s["s_lo"], s["v_lo"])
        upper = (s["h_hi"], s["s_hi"], s["v_hi"])

        mask = make_mask(hsv, lower, upper)
        mask = clean_mask(mask, s["kernel"])
        contours, boxes = find_holds(mask, s["min_area"], s["max_area"])
        result = draw_holds(frame, contours, boxes)

        clicked_hsv = clicks.clicked["hsv"] if clicks.clicked else None
        tap_samples = [tap["hsv"] for tap in taps] if controls.source == "tap" else None
        shown_taps = taps if controls.source == "tap" else None
        cv2.imshow(CONTROLS_WINDOW, controls.draw(image_area, clicked_hsv, tap_mode, tap_samples))
        cv2.imshow(VIEW_WINDOW, build_view(result, mask, s, len(boxes), clicks.clicked, tap_mode,
                                           shown_taps))

        raw = cv2.waitKeyEx(30)             # full key code, needed for the arrow keys
        if raw in KEYS_UP:
            controls.select_next(-1)
            continue
        if raw in KEYS_DOWN:
            controls.select_next(+1)
            continue
        if raw in KEYS_LEFT:
            controls.nudge(-1)
            continue
        if raw in KEYS_RIGHT:
            controls.nudge(+1)
            continue

        key = raw & 0xFF                    # letters: only the last 8 bits matter
        if key in (ord("q"), 27):
            break
        elif key == 9:                      # Tab
            controls.select_next(+1)
        elif key in (ord("a"), ord("d")):
            controls.nudge(-1 if key == ord("a") else +1)
        elif key in (ord("A"), ord("D")):
            controls.nudge(-10 if key == ord("A") else +10)
        elif key == ord("t"):
            tap_mode = not tap_mode
            print("Tap mode", "ON: click one hold" if tap_mode else "OFF")
        elif key in (ord("["), ord("]")) and tap_mode and taps:
            change = -1 if key == ord("[") else 1
            if len(taps) == 1:
                h_tol = max(1, min(89, h_tol + 2 * change))
            else:
                multi_h_tol = max(0, min(89, multi_h_tol + change))
            apply_taps()
        elif key == ord("r"):
            controls.reset()
            taps = []
            sizes_from_taps = False
        elif key == ord("p"):
            max_text = "none" if s["max_area"] is None else str(round(s["max_area"]))
            if controls.source != "tap":
                method = "manual"
            elif len(taps) > 1:
                method = f"multi-tap x{len(taps)} (hue tol {multi_h_tol})"
            else:
                method = f"tap-to-sample (hue +/-{h_tol})"
            print(f"lower={lower} upper={upper} kernel={s['kernel']} "
                  f"min_area={s['min_area']:.0f}px max_area={max_text} -> {len(boxes)} holds")
            row = (f"{path.stem},COLOUR,{lower[0]},{upper[0]},{lower[1]},{lower[2]},"
                   f"{s['kernel']},{round(s['min_area'])},{max_text},,,,,{method}")
            print("CSV row:")
            print(row)
        elif key == ord("s"):
            RESULTS_DIR.mkdir(exist_ok=True)
            if controls.source != "tap":
                tag = "manual"
            else:
                tag = "tap" if len(taps) <= 1 else f"tap{len(taps)}"   # tap3 = three taps
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_{tag}_result.jpg"), result)
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_{tag}_mask.jpg"), mask)
            print(f"Saved to {RESULTS_DIR}")

        # Stop if either window was closed with the X button
        if window_closed(VIEW_WINDOW) or window_closed(CONTROLS_WINDOW):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()