"""Simple bounding-box labeller for Where's Waldo pages, with zoom.

Opens every image in a folder, lets you drag a box around Waldo, and saves the
annotation in YOLO format (one .txt per image, class 0 = waldo). It remembers
what you have done, so you can label in several sittings.

Run:
    python scripts/label_waldo.py
    python scripts/label_waldo.py --images /path/to/imgs --labels /path/to/labels

Controls (also shown in the window):
    drag left mouse   draw a box (you can draw several per page)
    mouse wheel       zoom in / out at the cursor
    right-drag        pan around when zoomed in
    0                 reset zoom to fit the screen
    n / Enter         save boxes and go to the next image
    x                 mark this page as "no Waldo" (empty label) and go next
    s                 skip without saving (revisit later)
    b                 go back to the previous image
    u                 undo the last box
    c                 clear all boxes
    q / Esc           quit (progress is saved per image)
"""

import argparse
import shutil
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageTk

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ZOOM_STEP = 1.25
MAX_DISP = 9000          # cap the rendered image side (memory guard)


class Labeller:
    def __init__(self, root, images, labels_dir, reject_dir, view_w, view_h):
        self.root = root
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.reject_dir = reject_dir
        self.images = images
        self.view_w, self.view_h = view_w, view_h

        self.idx = self._first_unlabelled()
        self.boxes = []            # committed boxes in ORIGINAL pixel coords
        self.start = None          # drag start in canvas coords
        self.temp_id = None
        self.fit_scale = 1.0       # scale that fits the image to the viewport
        self.zoom = 1.0            # user zoom on top of fit_scale
        self.pil = None
        self.tkimg = None

        self.status = tk.Label(root, anchor="w", font=("Menlo", 12), bg="#111",
                               fg="#eee", padx=8, pady=6, justify="left")
        self.status.pack(fill="x")

        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(frame, bg="#222", highlightthickness=0,
                                cursor="crosshair", width=view_w, height=view_h)
        hbar = tk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        vbar = tk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", self.on_wheel)          # mac / windows
        self.canvas.bind("<Button-4>", self.on_wheel)            # linux up
        self.canvas.bind("<Button-5>", self.on_wheel)            # linux down
        root.bind("<Key>", self.on_key)

        self.load()

    # ---- coord helpers ---------------------------------------------------
    def disp_scale(self):
        return self.fit_scale * self.zoom

    def to_canvas(self, ox, oy):
        s = self.disp_scale()
        return ox * s, oy * s

    def to_orig(self, cx, cy):
        s = self.disp_scale()
        return cx / s, cy / s

    def label_path(self, i):
        return self.labels_dir / (self.images[i].stem + ".txt")

    def _first_unlabelled(self):
        for i in range(len(self.images)):
            if not self.label_path(i).exists():
                return i
        return 0

    # ---- image / label ---------------------------------------------------
    def load(self):
        self.boxes, self.start, self.temp_id = [], None, None
        path = self.images[self.idx]
        try:
            self.pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"cannot open {path.name}: {e}; skipping")
            self.advance(1)
            return
        ow, oh = self.pil.size
        self.fit_scale = min(self.view_w / ow, self.view_h / oh, 1.0)
        self.zoom = 1.0
        lp = self.label_path(self.idx)
        if lp.exists():
            for ln in lp.read_text().strip().splitlines():
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, bw, bh = map(float, p)
                    self.boxes.append([(cx - bw / 2) * ow, (cy - bh / 2) * oh,
                                       (cx + bw / 2) * ow, (cy + bh / 2) * oh])
        self.render()
        self.update_status()

    def render(self):
        ow, oh = self.pil.size
        s = self.disp_scale()
        dw, dh = max(1, int(ow * s)), max(1, int(oh * s))
        self.tkimg = ImageTk.PhotoImage(self.pil.resize((dw, dh)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)
        self.canvas.config(scrollregion=(0, 0, dw, dh))
        self.redraw_boxes()

    def redraw_boxes(self):
        self.canvas.delete("box")
        for ox1, oy1, ox2, oy2 in self.boxes:
            x1, y1 = self.to_canvas(ox1, oy1)
            x2, y2 = self.to_canvas(ox2, oy2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#39ff14", width=2, tags="box")

    def update_status(self):
        done = sum(1 for i in range(len(self.images)) if self.label_path(i).exists())
        name = self.images[self.idx].name
        self.status.config(
            text=f"[{self.idx + 1}/{len(self.images)}]  labelled:{done}   {name}   "
                 f"boxes:{len(self.boxes)}   zoom:{self.zoom:.1f}x\n"
                 f"drag=box  wheel=zoom  right-drag=pan  0=fit  "
                 f"n/Enter=save+next  x=no-waldo  s=skip  b=back  u=undo  c=clear  d=delete  q=quit")

    # ---- mouse -----------------------------------------------------------
    def on_down(self, e):
        self.start = (self.canvas.canvasx(e.x), self.canvas.canvasy(e.y))

    def on_drag(self, e):
        if self.start is None:
            return
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        if self.temp_id:
            self.canvas.delete(self.temp_id)
        self.temp_id = self.canvas.create_rectangle(
            self.start[0], self.start[1], cx, cy, outline="#ffd400", width=2)

    def on_up(self, e):
        if self.start is None:
            return
        x1, y1 = self.start
        x2, y2 = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        self.start = None
        if self.temp_id:
            self.canvas.delete(self.temp_id)
            self.temp_id = None
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return
        ox1, oy1 = self.to_orig(min(x1, x2), min(y1, y2))
        ox2, oy2 = self.to_orig(max(x1, x2), max(y1, y2))
        self.boxes.append([ox1, oy1, ox2, oy2])
        self.redraw_boxes()
        self.update_status()

    def on_wheel(self, e):
        # zoom keeping the point under the cursor fixed
        up = (getattr(e, "delta", 0) > 0) or getattr(e, "num", None) == 4
        factor = ZOOM_STEP if up else 1 / ZOOM_STEP
        ow, oh = self.pil.size
        new_zoom = self.zoom * factor
        if max(ow, oh) * self.fit_scale * new_zoom > MAX_DISP:
            return
        if new_zoom < 1.0:
            new_zoom = 1.0
        # original point currently under the cursor
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        oxp, oyp = self.to_orig(cx, cy)
        self.zoom = new_zoom
        self.render()
        # scroll so that same original point stays under the cursor
        s = self.disp_scale()
        dw, dh = ow * s, oh * s
        self.canvas.xview_moveto(max(0.0, (oxp * s - e.x) / dw))
        self.canvas.yview_moveto(max(0.0, (oyp * s - e.y) / dh))
        self.update_status()

    # ---- keys ------------------------------------------------------------
    def on_key(self, e):
        k = e.keysym.lower()
        if k in ("n", "return"):
            if not self.boxes:
                self.status.config(text="No box drawn. Draw one, press X for 'no Waldo', or S to skip.")
                return
            self.save()
            self.advance(1)
        elif k == "x":
            self.save(empty=True)
            self.advance(1)
        elif k == "s":
            self.advance(1)
        elif k == "b":
            self.advance(-1)
        elif k == "u":
            if self.boxes:
                self.boxes.pop()
                self.redraw_boxes()
                self.update_status()
        elif k == "c":
            self.boxes = []
            self.redraw_boxes()
            self.update_status()
        elif k in ("0", "equal", "plus", "minus"):
            if k == "0":
                self.zoom = 1.0
            elif k == "minus":
                self.zoom = max(1.0, self.zoom / ZOOM_STEP)
            else:
                self.zoom = self.zoom * ZOOM_STEP
            self.render()
            self.update_status()
        elif k in ("d", "delete", "backspace"):
            self.delete_current()
        elif k in ("q", "escape"):
            self.root.quit()

    # ---- delete ----------------------------------------------------------
    def delete_current(self):
        # move the image (and its label, if any) into the rejected folder
        self.reject_dir.mkdir(parents=True, exist_ok=True)
        path = self.images[self.idx]
        try:
            shutil.move(str(path), str(self.reject_dir / path.name))
        except Exception as e:
            print(f"could not move {path.name}: {e}")
            return
        lp = self.label_path(self.idx)
        if lp.exists():
            shutil.move(str(lp), str(self.reject_dir / lp.name))
        print(f"rejected {path.name} -> {self.reject_dir}")
        self.images.pop(self.idx)
        if not self.images:
            print("no images left")
            self.root.quit()
            return
        self.idx %= len(self.images)      # next image now sits at this index
        self.load()

    # ---- save / navigate -------------------------------------------------
    def save(self, empty=False):
        ow, oh = self.pil.size
        lines = []
        if not empty:
            for ox1, oy1, ox2, oy2 in self.boxes:
                cx = (ox1 + ox2) / 2 / ow
                cy = (oy1 + oy2) / 2 / oh
                bw = abs(ox2 - ox1) / ow
                bh = abs(oy2 - oy1) / oh
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        self.label_path(self.idx).write_text("\n".join(lines) + ("\n" if lines else ""))

    def advance(self, step):
        self.idx = (self.idx + step) % len(self.images)
        self.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/Users/boss/Downloads/waldo_book_pages")
    ap.add_argument("--labels", default="/Users/boss/Downloads/waldo_book_pages_labels")
    ap.add_argument("--rejected", default="/Users/boss/Downloads/waldo_book_pages_rejected")
    args = ap.parse_args()

    img_dir = Path(args.images)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"no images found in {img_dir}")

    root = tk.Tk()
    root.title("Waldo labeller")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    app = Labeller(root, images, Path(args.labels), Path(args.rejected),
                   view_w=sw - 80, view_h=sh - 200)
    # macOS often opens Tk windows behind everything - force it to the front
    root.geometry(f"+20+40")
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    root.focus_force()
    print(f"{len(images)} images | labels -> {args.labels}")
    print("drag=box  wheel=zoom  right-drag=pan  0=fit  n=save+next  x=no-waldo  s=skip  b=back  u=undo  c=clear  d=delete  q=quit")
    root.mainloop()
    done = sum(1 for i in range(len(images)) if app.label_path(i).exists())
    print(f"saved. {done}/{len(images)} images have labels in {args.labels}")


if __name__ == "__main__":
    main()
