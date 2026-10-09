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
    [ / ]               in tap mode: narrower / wider hue tolerance (re-applies the last tap)
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
 
 
def range_from_sample(h, s, v, h_tol=TAP_H_TOL, s_tol=TAP_S_TOL, v_tol=TAP_V_TOL):
    """Build the HSV range around a sampled colour.
 
    Hue: sampled hue +/- h_tol. It wraps round (% 180): a red sample at H=5 gives
    175 -> 15, which make_mask already handles as a wrap-around range.
    S and V: only a lower limit. Shadow and distance make parts of a hold duller and
    darker than the tapped spot, but never more vivid, so the upper limit stays 255.
    (A symmetric +/- range cut off the most vivid parts of holds in testing.)
    """
    lower = ((h - h_tol) % 180, max(s - s_tol, 0), max(v - v_tol, 0))
    upper = ((h + h_tol) % 180, 255, 255)
    return lower, upper
 
 
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
    {"key": "maxsize", "label": "6  MAX HOLD SIZE", "max": 100, "start": [100], "style": "plain",
     "hint": "Left = ignore big blobs (wall panels). Back right if big holds vanish"},
]
MAXSIZE_STEP = 0.001        # one step of the max-size slider = 0.1% of the image
                            # (all the way right = no limit)
 
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
            return f"{vals[0] * MAXSIZE_STEP * 100:.1f}% of image = {vals[0] * MAXSIZE_STEP * image_area:.0f} px"
        px = vals[0] / 10000 * image_area
        return f"{vals[0] / 100:.2f}% of image = {px:.0f} px"
 
    def draw(self, image_area, clicked_hsv=None, tap_mode=False):
        height = TOP + ROW_H * len(CONTROLS) + 92
        panel = np.full((height, PANEL_W, 3), PANEL_BG, np.uint8)
        if tap_mode:
            cv2.rectangle(panel, (0, 0), (PANEL_W, TOP - 8), (0, 90, 120), -1)
            put_text(panel, "TAP MODE: click a hold in the view window", (TRACK_X0, 28), 0.6, YELLOW, 2)
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
 
            # cyan marker: where the last clicked pixel sits on this slider
            if clicked_hsv is not None and c["style"] in ("hue", "sat", "val"):
                value = clicked_hsv[("hue", "sat", "val").index(c["style"])]
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
        if clicked_hsv is not None:
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
 
def build_view(result, mask, settings, n_holds, clicked, tap_mode=False):
    h, w = result.shape[:2]
    left = result.copy()
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    put_label(left, f"RESULT: {n_holds} holds", (8, 22), scale=0.6, colour=YELLOW)
    put_label(right, "MASK (white = in range)", (8, 22), scale=0.6, colour=YELLOW)
 
    if clicked is not None:                 # small cross where you clicked
        cx, cy = clicked["xy"]
        cv2.drawMarker(left, (cx, cy), CYAN, cv2.MARKER_CROSS, 14, 2)
        cv2.drawMarker(right, (cx, cy), CYAN, cv2.MARKER_CROSS, 14, 2)
        if clicked.get("patch"):            # the square that was averaged
            half = TAP_PATCH // 2
            cv2.rectangle(left, (cx - half, cy - half), (cx + half, cy + half), CYAN, 1)
 
    images = np.hstack([left, np.full((h, 6, 3), PANEL_BG, np.uint8), right])
    info = info_panel(images.shape[1], [
        f"Hue {settings['h_lo']}-{settings['h_hi']}  Sat {settings['s_lo']}-{settings['s_hi']}  "
        f"Bright {settings['v_lo']}-{settings['v_hi']}  Clean-up {settings['kernel']}  "
        f"Size {settings['min_area']:.0f}-"
        f"{'any' if settings['max_area'] is None else format(settings['max_area'], '.0f')}px",
        ("TAP MODE ON: click one hold to set the colour   (t = off)" if tap_mode
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
            self.press = {"xy": (x, y), "moves": 0}
        elif event == cv2.EVENT_MOUSEMOVE and self.press and flags & cv2.EVENT_FLAG_LBUTTON:
            self.press["moves"] += 1
        elif event == cv2.EVENT_LBUTTONUP and self.press:
            px, py = self.press["xy"]
            still = (self.press["moves"] <= self.MAX_MOVES
                     and abs(x - px) <= self.MAX_DISTANCE and abs(y - py) <= self.MAX_DISTANCE)
            self.press = None
            if still:
                self._sample(x, y)
 
    def _sample(self, x, y):
        if x >= self.img_w + self.gap:      # clicked on the mask side: same pixel
            x -= self.img_w + self.gap
        if 0 <= x < self.img_w and 0 <= y < self.img_h:
            h, s, v = (int(n) for n in self.hsv[y, x])
            self.clicked = {"xy": (x, y), "hsv": (h, s, v)}
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
    h_tol = TAP_H_TOL
    last_tap = None                         # sampled (h, s, v) of the last tap
 
    def apply_tap():
        lower, upper = range_from_sample(*last_tap, h_tol=h_tol)
        controls.apply_range(lower, upper)
        print(f"Tap sample H={last_tap[0]} S={last_tap[1]} V={last_tap[2]} "
              f"-> lower={lower} upper={upper} (hue +/-{h_tol})")
 
    while True:
        # Tap-to-sample: a new click in tap mode builds the colour range from that hold
        if clicks.new_click:
            clicks.new_click = False
            if tap_mode:
                x, y = clicks.clicked["xy"]
                last_tap = sample_patch(hsv, x, y)
                clicks.clicked["hsv"] = last_tap
                clicks.clicked["patch"] = True
                apply_tap()
 
        s = controls.settings(image_area)
        lower = (s["h_lo"], s["s_lo"], s["v_lo"])
        upper = (s["h_hi"], s["s_hi"], s["v_hi"])
 
        mask = make_mask(hsv, lower, upper)
        mask = clean_mask(mask, s["kernel"])
        contours, boxes = find_holds(mask, s["min_area"], s["max_area"])
        result = draw_holds(frame, contours, boxes)
 
        clicked_hsv = clicks.clicked["hsv"] if clicks.clicked else None
        cv2.imshow(CONTROLS_WINDOW, controls.draw(image_area, clicked_hsv, tap_mode))
        cv2.imshow(VIEW_WINDOW, build_view(result, mask, s, len(boxes), clicks.clicked, tap_mode))
 
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
        elif key in (ord("["), ord("]")) and tap_mode and last_tap:
            h_tol = max(1, min(89, h_tol + (-2 if key == ord("[") else 2)))
            apply_tap()
        elif key == ord("r"):
            controls.reset()
        elif key == ord("p"):
            max_text = "none" if s["max_area"] is None else str(round(s["max_area"]))
            method = f"tap-to-sample (hue +/-{h_tol})" if controls.source == "tap" else "manual"
            print(f"lower={lower} upper={upper} kernel={s['kernel']} "
                  f"min_area={s['min_area']:.0f}px max_area={max_text} -> {len(boxes)} holds")
            row = (f"{path.stem},COLOUR,{lower[0]},{upper[0]},{lower[1]},{lower[2]},"
                   f"{s['kernel']},{round(s['min_area'])},{max_text},,,,,{method}")
            print("CSV row:")
            print(row)
        elif key == ord("s"):
            RESULTS_DIR.mkdir(exist_ok=True)
            tag = "tap" if controls.source == "tap" else "manual"
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_{tag}_result.jpg"), result)
            cv2.imwrite(str(RESULTS_DIR / f"{path.stem}_{tag}_mask.jpg"), mask)
            print(f"Saved to {RESULTS_DIR}")
 
        # Stop if either window was closed with the X button
        if window_closed(VIEW_WINDOW) or window_closed(CONTROLS_WINDOW):
            break
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()

