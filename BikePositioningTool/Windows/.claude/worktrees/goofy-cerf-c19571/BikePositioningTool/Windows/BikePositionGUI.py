"""
BikePositionGUI.py - Windows control panel for the bike fit tool
================================================================

A dark, modern Tkinter front-end that wraps BikePositionCV.py so you don't need
the command line. Pick your options, start a live fit session (the camera window
opens in its own console), and open the analysis dashboard (builds the HTML
report and opens it in your browser).

    python BikePositionGUI.py

Optional polish: drop a photo at  assets/gui_bg.jpg  and it becomes the header
banner. (Pillow handles it; if Pillow or the image is missing, a flat banner is
used instead.) Nothing else to install - Tkinter ships with Python on Windows.
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

HERE = os.path.dirname(os.path.abspath(__file__))
BIKECV = os.path.join(HERE, "BikePositionCV.py")
CONFIG = os.path.join(HERE, "gui_config.json")
ASSETS = os.path.join(HERE, "assets")
DEFAULT_OUTDIR = os.path.abspath(os.path.join(os.getcwd(), "captures"))
NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

# Dark "matte" palette (tuned to the black-frame photos).
BG = "#101215"        # window background
CARD = "#191c20"      # section background
FIELD = "#0b0d0f"     # entry/input background
FG = "#eaecef"        # primary text
SUB = "#9aa0a6"       # secondary text
BORDER = "#2a2f35"    # hairline borders
ACC = "#6aa3ff"       # primary accent
ACC_TXT = "#08111f"   # text on accent
DISABLED = "#5b6168"
WIN_W = 540
BANNER_H = 150


def list_input_devices():
    """[(index, name), ...] of microphones, or [] if sounddevice is missing."""
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


def make_banner(width, height):
    """A PhotoImage banner: the bike photo (assets/gui_bg.jpg) darkened with the
    app title drawn on top. Falls back to None if Pillow/image is missing."""
    try:
        from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
    except Exception:
        return None
    path = os.path.join(ASSETS, "gui_bg.jpg")
    if os.path.isfile(path):
        img = ImageOps.fit(Image.open(path).convert("RGB"), (width, height),
                           Image.LANCZOS)
    else:
        img = Image.new("RGB", (width, height), (24, 27, 31))

    # Left-weighted dark scrim so the title is always legible over the photo.
    scrim = Image.new("L", (width, 1))
    for x in range(width):
        scrim.putpixel((x, 0), int(225 * max(0.0, 1.0 - x / (width * 0.9))))
    scrim = scrim.resize((width, height))
    img = Image.composite(Image.new("RGB", (width, height), (0, 0, 0)), img, scrim)

    d = ImageDraw.Draw(img)

    def font(size, bold=False):
        names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
        for n in names:
            try:
                return ImageFont.truetype(n, size)
            except Exception:
                continue
        return ImageFont.load_default()

    d.text((22, height - 70), "Bike Position Tool", font=font(28, True),
           fill=(245, 246, 248))
    d.text((24, height - 32),
           "side-on road-bike fit  -  measure - record - analyse",
           font=font(13), fill=(176, 182, 188))
    return ImageTk.PhotoImage(img)


class App:
    def __init__(self, root):
        self.root = root
        self.proc = None
        cfg = self._load_config()
        root.title("Bike Position Tool")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._init_style()

        # --- state ---------------------------------------------------------
        self.side = tk.StringVar(value=cfg.get("side", "right"))
        self.source = tk.StringVar(value=cfg.get("source", "auto"))
        self.outdir = tk.StringVar(value=cfg.get("outdir", DEFAULT_OUTDIR))
        self.save = tk.BooleanVar(value=cfg.get("save", True))
        self.name = tk.StringVar(value="")
        self.note = tk.StringVar(value="")
        self.voice = tk.BooleanVar(value=cfg.get("voice", False))
        self.mic = tk.StringVar(value="System default")

        # --- banner --------------------------------------------------------
        self._banner = make_banner(WIN_W, BANNER_H)
        if self._banner is not None:
            tk.Label(root, image=self._banner, bd=0, bg=BG).pack(fill="x")
        else:
            band = tk.Frame(root, bg="#0c0d0f", height=BANNER_H)
            band.pack(fill="x")
            band.pack_propagate(False)
            tk.Label(band, text="Bike Position Tool", bg="#0c0d0f", fg=FG,
                     font=("Segoe UI", 20, "bold")).place(x=22, y=BANNER_H - 64)
            tk.Label(band, text="side-on road-bike fit", bg="#0c0d0f", fg=SUB,
                     font=("Segoe UI", 10)).place(x=24, y=BANNER_H - 28)

        main = ttk.Frame(root, padding=(16, 14, 16, 14))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- camera --------------------------------------------------------
        cam = self._section(main, "Camera", 0)
        ttk.Label(cam, text="Drive side facing camera",
                  style="CardSub.TLabel").grid(row=0, column=0, sticky="w")
        sb = ttk.Frame(cam, style="Card.TFrame")
        sb.grid(row=0, column=1, columnspan=2, sticky="e")
        ttk.Radiobutton(sb, text="Right", variable=self.side, value="right",
                        style="Card.TRadiobutton").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(sb, text="Left", variable=self.side, value="left",
                        style="Card.TRadiobutton").pack(side="left")
        ttk.Label(cam, text="Source", style="CardSub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(cam, textvariable=self.source).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(cam, text="'auto' finds the IMX415, or a camera index / video path",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=3,
                                            sticky="w", pady=(4, 0))

        # --- saving --------------------------------------------------------
        out = self._section(main, "Saving", 1)
        ttk.Checkbutton(out, text="Save video, snapshot & report",
                        variable=self.save, style="Card.TCheckbutton").grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(out, text="Captures folder", style="CardSub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(out, textvariable=self.outdir).grid(
            row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(out, text="Browse", command=self._browse, width=8).grid(
            row=1, column=2, sticky="e", padx=(8, 0), pady=(10, 0))

        # --- run labelling -------------------------------------------------
        run = self._section(main, "This run (optional)", 2)
        ttk.Label(run, text="Name", style="CardSub.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Entry(run, textvariable=self.name).grid(
            row=0, column=1, columnspan=2, sticky="ew")
        ttk.Label(run, text="Notes", style="CardSub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(run, textvariable=self.note).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(run, text="e.g. '5mm saddle lowered'  - saved with the run",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=3,
                                            sticky="w", pady=(4, 0))

        # --- voice ---------------------------------------------------------
        voice = self._section(main, "Hands-free voice control", 3)
        ttk.Checkbutton(voice, text="Enable voice (say 'start' / 'stop')",
                        variable=self.voice, command=self._toggle_voice,
                        style="Card.TCheckbutton").grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(voice, text="Microphone", style="CardSub.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        self.mic_combo = ttk.Combobox(voice, textvariable=self.mic,
                                      state="readonly")
        self.mic_combo.grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(voice, text="Refresh", command=self._refresh_mics,
                   width=8).grid(row=1, column=2, sticky="e", padx=(8, 0),
                                 pady=(10, 0))
        self._refresh_mics()
        self._toggle_voice()

        # --- actions -------------------------------------------------------
        actions = ttk.Frame(main, style="TFrame")
        actions.grid(row=4, column=0, sticky="ew", pady=(14, 6))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.start_btn = ttk.Button(actions, text="▶  Start Fit Session",
                                    style="Accent.TButton",
                                    command=self.start_session)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6), ipady=4)
        ttk.Button(actions, text="\U0001F4CA  Analysis Dashboard",
                   command=self.open_analysis).grid(
            row=0, column=1, sticky="ew", padx=(6, 0), ipady=4)

        # --- log -----------------------------------------------------------
        self.log_box = scrolledtext.ScrolledText(
            main, width=10, height=7, state="disabled", relief="flat",
            bg=FIELD, fg=SUB, insertbackground=FG, bd=0,
            font=("Consolas", 9), highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=BORDER)
        self.log_box.grid(row=5, column=0, sticky="ew", pady=(8, 0))

        if not os.path.isfile(BIKECV):
            self.log(f"WARNING: cannot find {BIKECV}")
        else:
            self.log("Ready. Set your options, then Start Fit Session.")
        self.log("In the camera window: SPACE = record, R = reset, Q = quit.")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- styling ----------------------------------------------------------
    def _init_style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACC)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ACC_TXT)
        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Sub.TLabel", background=BG, foreground=SUB)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("CardSub.TLabel", background=CARD, foreground=SUB)
        s.configure("Hint.TLabel", background=CARD, foreground="#6c727a",
                    font=("Segoe UI", 8))
        s.configure("Card.TLabelframe", background=CARD, bordercolor=BORDER,
                    relief="solid", borderwidth=1)
        s.configure("Card.TLabelframe.Label", background=CARD, foreground=ACC,
                    font=("Segoe UI", 9, "bold"))
        s.configure("TButton", background="#23272c", foreground=FG,
                    bordercolor=BORDER, relief="flat", padding=6)
        s.map("TButton", background=[("active", "#2d3238")],
              foreground=[("disabled", DISABLED)])
        s.configure("Accent.TButton", background=ACC, foreground=ACC_TXT,
                    relief="flat", padding=6, font=("Segoe UI", 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#84b4ff")])
        s.configure("Card.TCheckbutton", background=CARD, foreground=FG)
        s.map("Card.TCheckbutton", background=[("active", CARD)],
              foreground=[("disabled", DISABLED)])
        s.configure("Card.TRadiobutton", background=CARD, foreground=FG)
        s.map("Card.TRadiobutton", background=[("active", CARD)])
        s.configure("TEntry", fieldbackground=FIELD, foreground=FG,
                    bordercolor=BORDER, insertcolor=FG, padding=5)
        s.configure("TCombobox", fieldbackground=FIELD, foreground=FG,
                    bordercolor=BORDER, arrowcolor=FG, padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)],
              foreground=[("disabled", DISABLED)])

    def _section(self, parent, title, row):
        f = ttk.Labelframe(parent, text="  " + title + "  ",
                           style="Card.TLabelframe", padding=12)
        f.grid(row=row, column=0, sticky="ew", pady=6)
        f.columnconfigure(1, weight=1)
        return f

    # ---- helpers ----------------------------------------------------------
    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get() or os.getcwd())
        if d:
            self.outdir.set(d)

    def _refresh_mics(self):
        self._mics = list_input_devices()
        values = ["System default"] + [f"{i}: {n}" for i, n in self._mics]
        self.mic_combo["values"] = values
        if self.mic.get() not in values:
            self.mic.set("System default")

    def _toggle_voice(self):
        self.mic_combo.configure(state="readonly" if self.voice.get()
                                 else "disabled")

    def _selected_mic_index(self):
        sel = self.mic.get()
        if sel and sel[0].isdigit() and ":" in sel:
            return sel.split(":", 1)[0].strip()
        return None

    # ---- actions ----------------------------------------------------------
    def start_session(self):
        if self.proc is not None and self.proc.poll() is None:
            self.log("A session is already running. Quit it first (press Q).")
            return
        if not os.path.isfile(BIKECV):
            self.log(f"ERROR: {BIKECV} not found.")
            return
        cmd = [sys.executable, BIKECV,
               "--side", self.side.get(),
               "--source", self.source.get() or "auto",
               "--outdir", self.outdir.get() or DEFAULT_OUTDIR,
               "--name", self.name.get(),
               "--note", self.note.get()]
        if not self.save.get():
            cmd.append("--no-save")
        if self.voice.get():
            cmd.append("--voice")
            idx = self._selected_mic_index()
            if idx is not None:
                cmd += ["--mic", idx]
        self._save_config()
        try:
            self.proc = subprocess.Popen(cmd, creationflags=NEW_CONSOLE)
            self.log(f"Started fit session (PID {self.proc.pid}). "
                     f"Camera window opening...")
        except Exception as e:
            self.log(f"ERROR starting session: {e}")

    def open_analysis(self):
        self._save_config()
        outdir = self.outdir.get() or DEFAULT_OUTDIR
        self.log("Building analysis dashboard...")
        threading.Thread(target=self._run_analysis, args=(outdir,),
                         daemon=True).start()

    def _run_analysis(self, outdir):
        cmd = [sys.executable, BIKECV, "--analyze", "--outdir", outdir]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stdout or "").strip()
            if r.stderr.strip():
                out += "\n" + r.stderr.strip()
        except Exception as e:
            out = f"ERROR: {e}"
        self.root.after(0, lambda: self.log(out or "Done."))

    # ---- config -----------------------------------------------------------
    def _load_config(self):
        try:
            with open(CONFIG, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_config(self):
        try:
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump({"side": self.side.get(), "source": self.source.get(),
                           "outdir": self.outdir.get(), "save": self.save.get(),
                           "voice": self.voice.get()}, f, indent=2)
        except Exception:
            pass

    def _on_close(self):
        self._save_config()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
