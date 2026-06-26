"""
BikePositionGUI.py - Windows control panel for the bike fit tool
================================================================

A Tkinter front-end that wraps BikePositionCV.py so you don't need the command
line. Pick your options, start a live fit session (the camera window opens in
its own console), and open the analysis dashboard (builds the HTML report and
opens it in your browser).

    python BikePositionGUI.py

Nothing extra to install - Tkinter ships with Python on Windows.
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
DEFAULT_OUTDIR = os.path.abspath(os.path.join(os.getcwd(), "captures"))
NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def list_input_devices():
    """[(index, name), ...] of microphones, or [] if sounddevice is missing."""
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


class App:
    def __init__(self, root):
        self.root = root
        self.proc = None
        cfg = self._load_config()
        root.title("Bike Position Tool")
        root.resizable(False, False)

        # --- state ---------------------------------------------------------
        self.side = tk.StringVar(value=cfg.get("side", "right"))
        self.source = tk.StringVar(value=cfg.get("source", "auto"))
        self.outdir = tk.StringVar(value=cfg.get("outdir", DEFAULT_OUTDIR))
        self.save = tk.BooleanVar(value=cfg.get("save", True))
        self.name = tk.StringVar(value="")
        self.note = tk.StringVar(value="")
        self.voice = tk.BooleanVar(value=cfg.get("voice", False))
        self.mic = tk.StringVar(value="System default")

        main = ttk.Frame(root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main, text="Bike Position Tool",
                  font=("Segoe UI", 15, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(main, text="Side-on road-bike fit — measure, record, analyse",
                  foreground="#666").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # --- camera --------------------------------------------------------
        cam = ttk.LabelFrame(main, text="Camera", padding=10)
        cam.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(cam, text="Drive side facing camera:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(cam, text="Right", variable=self.side,
                        value="right").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(cam, text="Left", variable=self.side,
                        value="left").grid(row=0, column=2, sticky="w")
        ttk.Label(cam, text="Source:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(cam, textvariable=self.source, width=24).grid(
            row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(cam, text="'auto' = find IMX415, or a camera index / video path",
                  foreground="#888", font=("Segoe UI", 8)).grid(
            row=2, column=0, columnspan=3, sticky="w")

        # --- saving --------------------------------------------------------
        out = ttk.LabelFrame(main, text="Saving", padding=10)
        out.grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Checkbutton(out, text="Save video, snapshot & report",
                        variable=self.save).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(out, text="Captures folder:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(out, textvariable=self.outdir, width=34).grid(
            row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Button(out, text="Browse…", command=self._browse).grid(
            row=1, column=2, sticky="w", padx=(6, 0), pady=(6, 0))

        # --- run labelling -------------------------------------------------
        run = ttk.LabelFrame(main, text="This run (optional)", padding=10)
        run.grid(row=4, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(run, text="Name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(run, textvariable=self.name, width=34).grid(
            row=0, column=1, columnspan=2, sticky="w")
        ttk.Label(run, text="Notes:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(run, textvariable=self.note, width=34).grid(
            row=1, column=1, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(run, text="e.g. '5mm saddle lowered'  (saved with the run)",
                  foreground="#888", font=("Segoe UI", 8)).grid(
            row=2, column=0, columnspan=3, sticky="w")

        # --- voice ---------------------------------------------------------
        voice = ttk.LabelFrame(main, text="Hands-free voice control", padding=10)
        voice.grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Checkbutton(voice, text="Enable voice (say 'start' / 'stop')",
                        variable=self.voice, command=self._toggle_voice).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(voice, text="Microphone:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.mic_combo = ttk.Combobox(voice, textvariable=self.mic, width=30,
                                      state="readonly")
        self.mic_combo.grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Button(voice, text="Refresh", command=self._refresh_mics).grid(
            row=1, column=2, sticky="w", padx=(6, 0), pady=(6, 0))
        self._refresh_mics()
        self._toggle_voice()

        # --- actions -------------------------------------------------------
        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        self.start_btn = ttk.Button(actions, text="▶  Start Fit Session",
                                    command=self.start_session)
        self.start_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="\U0001F4CA  Open Analysis Dashboard",
                   command=self.open_analysis).grid(row=0, column=1, padx=(8, 0))

        # --- log -----------------------------------------------------------
        self.log_box = scrolledtext.ScrolledText(main, width=58, height=8,
                                                 state="disabled",
                                                 font=("Consolas", 9))
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        if not os.path.isfile(BIKECV):
            self.log(f"WARNING: cannot find {BIKECV}")
        else:
            self.log("Ready. Set your options, then Start Fit Session.")
        self.log("Tip: in the camera window, SPACE=record, R=reset, Q=quit.")

        root.protocol("WM_DELETE_WINDOW", self._on_close)

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
        state = "readonly" if self.voice.get() else "disabled"
        self.mic_combo.configure(state=state)

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
                     f"Camera window opening…")
        except Exception as e:
            self.log(f"ERROR starting session: {e}")

    def open_analysis(self):
        self._save_config()
        outdir = self.outdir.get() or DEFAULT_OUTDIR
        self.log("Building analysis dashboard…")
        threading.Thread(target=self._run_analysis, args=(outdir,),
                         daemon=True).start()

    def _run_analysis(self, outdir):
        cmd = [sys.executable, BIKECV, "--analyze", "--outdir", outdir]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stdout or "").strip() + ("\n" + r.stderr.strip()
                                              if r.stderr.strip() else "")
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
