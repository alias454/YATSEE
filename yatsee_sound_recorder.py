#!/usr/bin/env python3
"""
YATSEE Sound Recorder

A cross-platform GUI utility for real-time audio capture of civic meetings.
Designed for the YATSEE pipeline, it produces high-fidelity FLAC files
while ensuring no audio data is lost during the recording process.

SYSTEM REQUIREMENTS:
- Windows 10/11, macOS, or Linux.
- Python 3.10+ with libraries: customtkinter, sounddevice, soundfile, numpy.

OS-LEVEL MIC BOOST (Hiss Reduction):
A constant background "hiss" or white noise is typically caused by analog gain
multiplication at the OS level. For clean recordings, ensure "Mic Boost" or
"Input Boost" is set to 0.0 dB in your system settings (e.g., 'alsamixer' on
Linux or 'Sound Control Panel' on Windows). This script defaults to 24-bit
(PCM_24) recording to provide superior digital headroom, allowing for
clean amplification later in the pipeline without introducing digital hiss.

WINDOWS-SPECIFIC HARDWARE NOTES (e.g., Shure MV7/USB Mics):
1. Privacy Settings: Ensure 'Allow desktop apps to access your microphone'
   is enabled in Windows Settings > Privacy > Microphone.
2. Host API (WASAPI): For best stability on Windows, select the input device
   entry ending in '(WASAPI)' from the dropdown menu.
3. Exclusive Mode: If recording fails to start, disable 'Allow applications
   to take exclusive control of this device' in the Advanced tab of your
   Microphone Properties in the Windows Sound Control Panel.
4. Auto-Leveling: If using Shure MOTIV software, enabling 'Auto-Level' is
   recommended to handle varying speaker distances in council chambers.

PRODUCTION DEPLOYMENT:
To bundle this script as a standalone Windows executable (.exe):
    pip install pyinstaller
    pyinstaller --noconsole --onefile yatsee_sound_recorder.py

FEATURES:
- Threaded Queue: Captures audio at OS priority to prevent gaps.
- Real-time Disk Write: Streams directly to disk; data is safe even if power fails.
- Custom Browser: Modern, dark-mode folder picker that bypasses legacy OS dialogs.
- FLAC Encoding: Lossless quality at ~50% the file size of WAV.

TODO Potential features to add:
- Silence detection: Pause writing to disk when the audio is below a threshold to save space.
    Use RMS or peak amplitude on audio chunks; skip writing when below a set level.
- Hotword start: Begin recording automatically when a trigger phrase like "let’s go wizard" is detected.
    Vosk or Porcupine for offline keyword spotting; feed your mic stream and trigger start_recording() when detected.
- Hotword stop: Stop recording with a voice command like "YATSEE meeting bot, stop recording".
    Same as above; trigger stop_recording() when phrase is recognized.
- Rolling buffer: Keep a short pre-trigger audio buffer so recordings capture a few seconds before the hotword.
    Use a collections.deque with max length storing audio frames; dump buffer into file once hotword triggers.
- Expandable commands: Future potential to add voice-controlled functions like save, discard, or rename clips.
    Treat recognized text from Vosk as commands; map phrases to functions in your app.
"""
import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import os
import sys
import subprocess
import platform
from datetime import datetime
from tkinter import filedialog

# --- GUI Setup ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ----------------- SECURITY HELPERS -----------------

def safe_open_folder(path):
    """Open a folder with minimal shell exposure - Cross Platform."""
    if not os.path.isdir(path):
        return

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


# ----------------- CUSTOM FOLDER PICKER -----------------

class CustomFolderPicker(ctk.CTkToplevel):
    def __init__(self, parent, start_path, callback):
        super().__init__(parent)
        self.callback = callback

        if not start_path or not os.path.exists(start_path):
            start_path = os.path.expanduser("~")

        self.current_path = os.path.abspath(start_path)

        self.title("Select Save Location")
        self.geometry("600x400")
        self.attributes("-topmost", True)
        self.transient(parent)

        # FIX: Wait for window to be viewable before grabbing focus (Linux fix)
        self.after(100, lambda: self.grab_set_safe())

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.lbl_path = ctk.CTkLabel(self.header_frame, text=self.truncate(self.current_path),
                                     font=("Roboto", 14, "bold"))
        self.lbl_path.pack(side="left", padx=5)

        self.nav_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.nav_frame.grid(row=1, column=0, padx=10, pady=(0, 5), sticky="ew")

        self.btn_up = ctk.CTkButton(self.nav_frame, text="⬆ Up", width=60, command=self.go_up)
        self.btn_up.pack(side="left", padx=5)

        drives = self.get_drives()
        self.drives_menu = ctk.CTkOptionMenu(self.nav_frame, values=drives, command=self.change_drive, width=100)
        self.drives_menu.pack(side="left", padx=5)

        if platform.system() == "Windows":
            self.drives_menu.set(self.current_path[:3].upper())
        else:
            self.drives_menu.set("/")

        self.btn_select = ctk.CTkButton(self.nav_frame, text="Select This Folder", hover_color="#26ad72",
                                        command=self.confirm_selection)
        self.btn_select.pack(side="right", padx=5)

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Folders")
        self.scroll_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.refresh_list()

    def grab_set_safe(self):
        try:
            self.grab_set()
        except:
            pass

    def truncate(self, path):
        return "..." + path[-52:] if len(path) > 55 else path

    def get_drives(self):
        system = platform.system()
        drives = [os.path.expanduser("~")]

        if system == "Windows":
            try:
                import string
                import ctypes
                bitmask = ctypes.windll.kernel32.GetLogicalDrives()
                for letter in string.ascii_uppercase:
                    if bitmask & 1:
                        drives.append(f"{letter}:\\")
                    bitmask >>= 1
            except:
                drives = ["C:\\"]
        else:
            drives.append("/")
            for m in ["/media", "/mnt", f"/media/{os.getlogin() if hasattr(os, 'getlogin') else 'user'}"]:
                if os.path.exists(m):
                    drives.append(m)
        return sorted(list(set(drives)))

    def change_drive(self, drive):
        self.current_path = os.path.abspath(drive)
        self.refresh_list()

    def go_up(self):
        parent = os.path.dirname(self.current_path)
        if parent != self.current_path:
            self.current_path = parent
            self.refresh_list()

    def enter_folder(self, folder_name):
        new_path = os.path.join(self.current_path, folder_name)
        if os.path.isdir(new_path) and os.access(new_path, os.R_OK):
            self.current_path = new_path
            self.refresh_list()

    def refresh_list(self):
        self.lbl_path.configure(text=self.truncate(self.current_path))
        for w in self.scroll_frame.winfo_children():
            w.destroy()

        try:
            items = sorted(os.listdir(self.current_path))
            dirs = [d for d in items if os.path.isdir(os.path.join(self.current_path, d)) and not d.startswith('.')]

            if not dirs:
                ctk.CTkLabel(self.scroll_frame, text="(No accessible subfolders)", text_color="gray").pack(pady=20)

            for d in dirs:
                ctk.CTkButton(
                    self.scroll_frame,
                    text=f"📁 {d}",
                    anchor="w",
                    fg_color="transparent",
                    hover_color=("gray70", "gray30"),
                    command=lambda f=d: self.enter_folder(f)
                ).pack(fill="x", padx=2, pady=1)
        except Exception:
            ctk.CTkLabel(self.scroll_frame, text="Access Denied", text_color="red").pack()

    def confirm_selection(self):
        self.callback(self.current_path)
        self.destroy()


# ---------------------- MAIN APP ----------------------

class RecorderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YATSEE Meeting Recorder")
        self.geometry("700x320")
        self.resizable(False, False)

        self.is_recording = False
        self.start_time = None
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue()
        self.recording_thread = None
        self.device_map = {}
        self.selected_device_id = None

        default_dir = os.path.join(os.path.expanduser("~"), "recordings")
        os.makedirs(default_dir, exist_ok=True)
        self.output_dir = os.path.abspath(default_dir)

        self.build_ui()
        self.after(100, self.refresh_devices)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        ctk.CTkLabel(left, text="Configuration", font=("Roboto", 16, "bold")).pack(anchor="w", pady=(0, 20))

        ctk.CTkLabel(left, text="Audio Input Source:", text_color="gray").pack(anchor="w")
        self.device_menu = ctk.CTkOptionMenu(left, dynamic_resizing=False, values=["Scanning..."],
                                             command=self.change_device, width=280)
        self.device_menu.pack(anchor="w", pady=(0, 20))

        ctk.CTkLabel(left, text="Save Directory:", text_color="gray").pack(anchor="w")
        path_box = ctk.CTkFrame(left, fg_color="transparent")
        path_box.pack(anchor="w", fill="x")

        self.entry_path = ctk.CTkEntry(path_box, width=200)
        self.entry_path.insert(0, self.output_dir)
        self.entry_path.configure(state="disabled")
        self.entry_path.pack(side="left", padx=(0, 10))

        # FIX: Assign to self so it can be disabled during recording
        self.btn_browse = ctk.CTkButton(path_box, text="Browse", width=60, command=self.open_custom_picker)
        self.btn_browse.pack(side="left")

        # FIX: Assign to self so it can be disabled during recording
        self.btn_open_folder = ctk.CTkButton(left, text="📂 Open Folder", fg_color="transparent", border_width=1,
                                             width=280,
                                             command=self.open_output_folder)
        self.btn_open_folder.pack(side="bottom", pady=(20, 0))

        # Right side
        right = ctk.CTkFrame(self, fg_color=("gray85", "gray17"))
        right.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.timer_label = ctk.CTkLabel(right, text="00:00:00", font=("Roboto Medium", 54), text_color="gray")
        self.timer_label.pack(pady=(20, 10))

        self.btn_record = ctk.CTkButton(right, text="REC", font=("Roboto", 24, "bold"),
                                        fg_color="#2CC985", hover_color="#26ad72", height=120, width=120,
                                        corner_radius=60,
                                        command=self.toggle_recording)
        self.btn_record.pack(pady=10)

        self.status_label = ctk.CTkLabel(right, text="System Ready", font=("Roboto", 14))
        self.status_label.pack()

    def open_custom_picker(self):
        CustomFolderPicker(self, self.output_dir, self.set_new_path)

    def set_new_path(self, path):
        self.output_dir = os.path.abspath(path)
        self.entry_path.configure(state="normal")
        self.entry_path.delete(0, "end")
        self.entry_path.insert(0, self.output_dir)
        self.entry_path.configure(state="disabled")

    def refresh_devices(self):
        try:
            devices = sd.query_devices()
        except Exception:
            self.status_label.configure(text="Audio Error", text_color="red")
            return

        self.device_map.clear()
        menu_values = []
        default_id = sd.default.device[0] if sd.default.device else None
        default_name = None

        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                name_clean = dev["name"].replace("\n", " ").strip()
                display = f"{name_clean} ({i})"
                self.device_map[display] = i
                menu_values.append(display)
                if i == default_id:
                    default_name = display

        if menu_values:
            self.device_menu.configure(values=menu_values)
            chosen = default_name if default_name else menu_values[0]
            self.device_menu.set(chosen)
            self.selected_device_id = self.device_map[chosen]

    def change_device(self, choice):
        self.selected_device_id = self.device_map.get(choice)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.selected_device_id is None:
            return

        self.is_recording = True
        self.stop_event.clear()
        self.start_time = time.time()

        self.btn_record.configure(text="STOP", fg_color="#C92C2C", hover_color="#ad2626")
        self.timer_label.configure(text_color="#C92C2C")
        self.status_label.configure(text="● Recording Active", text_color="#C92C2C")

        # Disable settings during recording
        self.device_menu.configure(state="disabled")
        self.btn_browse.configure(state="disabled")
        self.btn_open_folder.configure(state="disabled")

        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()
        self.update_timer()

    def stop_recording(self):
        self.stop_event.set()
        self.status_label.configure(text="Saving...", text_color="orange")

    def update_timer(self):
        if self.is_recording:
            elapsed = int(time.time() - self.start_time)
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            self.timer_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
            self.after(1000, self.update_timer)

    def record_audio(self):
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = os.path.join(self.output_dir, f"Meeting_{ts}.flac")

        def cb(indata, frames, time, status):
            self.audio_q.put(indata.copy())

        try:
            info = sd.query_devices(self.selected_device_id, 'input')
            sr = int(info['default_samplerate'])

            # FIX: Write audio and flush inside the context manager to finalize FLAC header
            with sf.SoundFile(filepath, mode='x', samplerate=sr, channels=1, format='FLAC', subtype='PCM_24') as f:
                with sd.InputStream(samplerate=sr, device=self.selected_device_id, channels=1, callback=cb):
                    while not self.stop_event.is_set():
                        try:
                            data = self.audio_q.get(timeout=0.1)
                            f.write(data)
                        except queue.Empty:
                            continue

                    # Final drain
                    while not self.audio_q.empty():
                        f.write(self.audio_q.get())

            self.after(0, self.finalize_ui)
        except Exception as e:
            print(f"Error: {e}")
            self.after(0, self.finalize_ui)

    def finalize_ui(self):
        self.is_recording = False
        self.btn_record.configure(text="REC", fg_color="#2CC985", hover_color="#26ad72")
        self.timer_label.configure(text_color="gray")
        self.status_label.configure(text="File Saved Successfully", text_color="#2CC985")
        self.device_menu.configure(state="normal")
        self.btn_browse.configure(state="normal")
        self.btn_open_folder.configure(state="normal")

    def open_output_folder(self):
        try:
            safe_open_folder(self.output_dir)
        except:
            pass

    def on_close(self):
        if self.is_recording:
            self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    app = RecorderApp()
    app.mainloop()