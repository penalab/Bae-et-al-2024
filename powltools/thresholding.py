"""GUI Tool to find and set thresholds for spike detection.

Usage: python -m powltools.thresholding_gui

The package `powltools` must be importable from where you run this. Usually,
that means you are in the parent directory of the package.

"""

from __future__ import annotations as _annotations
from functools import cached_property, partial
import json
import os
import re
import numpy as np
import customtkinter
import tkinter
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk  # type: ignore
import matplotlib.pyplot as plt
from .filters.chained import get_spikecontinuous_filter
from .io.analog import AnalogSignal
from .io.file import POwlFile
from .io.parameters import get_params
from .spikes.detect import (
    detect_spikes,
    indexes2times,
    spiketrains2times,
    split_indexes,
)
from ._includes.blit_manager import BlitManager

plt.rcParams["axes.axisbelow"] = False
plt.rcParams["mathtext.default"] = "rm"


class ThresholdingApp(customtkinter.CTk):
    _TITLE = "pOwl Thresholding"
    DEFAULT_THRESHOLD = -32.0

    def __init__(self):
        super().__init__()
        # Inital empty datastructure:
        self.filenames: tuple[str, ...] = tuple()
        self.analog_signals: dict[POwlFile, AnalogSignal] = {}
        self.current_channel = None
        self.last_threshold = None
        # Setting up the GUI
        self.title(self._TITLE)
        self.rowconfigure(0, pad=10, weight=1)
        self.rowconfigure(1, pad=10)
        self.columnconfigure(0, pad=10, weight=1)
        self._init_display()
        self._init_thresholding_actions()
        self._init_file_actions()
        self._init_keyboard_bindings()
        # Bind quit function:
        self.protocol("WM_DELETE_WINDOW", self.callback_quit)
        # Maximize the window using state property
        self.after_idle(lambda: (self.state("zoomed"), self.threshold_updated()))

    def _init_display(self):
        frame = tkinter.Frame(master=self)
        frame.grid(row=0, column=0, sticky="NSEW")
        self.fig, self.ax_trace = plt.subplots(
            1, 1, gridspec_kw=dict(left=0, bottom=0, right=1, top=1)
        )
        self.ax_trace.tick_params(
            "both", direction="in", pad=-30, color="#99f", labelcolor="#99f"
        )
        self.ax_trace.set_facecolor("black")
        self.ax_trace.xaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")
        self.ax_trace.set_ylim(bottom=-160, top=+160)
        self.ax_trace.set_xlim(left=-5, right=+105)
        self.messagebox = self.ax_trace.text(
            0.5,
            0.5,
            "pOwl Thesholding",
            ha="center",
            va="bottom",
            transform=self.ax_trace.transAxes,
            color="black",
            fontdict=dict(fontsize=30),
            bbox=dict(boxstyle="round", facecolor="#fff"),
            zorder=1000,
        )
        self.threshold_line = self.ax_trace.axhline(
            0.0,
            color="r",
            lw=2,
            zorder=100,
        )
        self.file_indicator = self.ax_trace.axvspan(
            xmin=-10,
            xmax=0,
            facecolor="green",
            linestyle="none",
            alpha=0.3,
            zorder=500,
        )
        self.file_indicator.set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.bm = BlitManager(
            self.canvas, [self.threshold_line, self.file_indicator, self.messagebox]
        )
        # Toolbar with fewer selected options:
        orig_toolitems = list(NavigationToolbar2Tk.toolitems)
        NavigationToolbar2Tk.toolitems = [
            t for t in orig_toolitems if t[0] not in ("Subplots",)
        ]
        toolbar = NavigationToolbar2Tk(self.canvas, frame)
        toolbar.update()

    def _init_thresholding_actions(self):
        frame = tkinter.Frame(master=self)
        frame.grid(row=1, column=0, sticky="NSEW", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        self.threshold_var = customtkinter.DoubleVar(value=0.0)
        self.threshold_slider = customtkinter.CTkSlider(
            master=frame,
            variable=self.threshold_var,
            command=lambda v: self.react_slider_moved(),
            from_=-100,
            to=0,
            number_of_steps=2_000,
        )
        self.threshold_slider.grid(row=0, column=0, sticky="EW")
        self.threshold_text_var = customtkinter.StringVar(value=f"+0.0")
        self.threshold_entry = customtkinter.CTkEntry(
            master=frame,
            placeholder_text="Threshold",
            textvariable=self.threshold_text_var,
        )
        self.threshold_entry.bind("<Any-KeyRelease>", self.threshold_entry_changed)
        self.threshold_entry.grid(row=0, column=1, padx=(10, 0))

        self.threshold_negative = customtkinter.BooleanVar(value=True)
        self.switch_threshold_negative = customtkinter.CTkSwitch(
            master=frame,
            text="Neg",
            variable=self.threshold_negative,
            command=self.threshold_sign,
            onvalue=True,
            offvalue=False,
        )
        self.switch_threshold_negative.grid(row=0, column=2, padx=(10, 0))

        self.quick_set_3sd = customtkinter.CTkButton(
            master=frame,
            text="3x SD",
            command=partial(self.callback_standard_deviation, multiplier=3),
            width=60,
        )
        self.quick_set_3sd.grid(row=0, column=3, padx=(10, 0))

        self.quick_set_35sd = customtkinter.CTkButton(
            master=frame,
            text="3.5x SD",
            command=partial(self.callback_standard_deviation, multiplier=3.5),
            width=60,
        )
        self.quick_set_35sd.grid(row=0, column=4, padx=(10, 0))

        self.quick_set_4sd = customtkinter.CTkButton(
            master=frame,
            text="4x SD",
            command=partial(self.callback_standard_deviation, multiplier=4),
            width=60,
        )
        self.quick_set_4sd.grid(row=0, column=5, padx=(10, 0))

    def _init_file_actions(self):
        frame = tkinter.Frame(master=self)
        frame.grid(row=2, column=0, padx=10, pady=10, sticky="NSEW")

        col = -1

        button_open = customtkinter.CTkButton(
            master=frame,
            text="Open",
            command=self.callback_open,
            width=60,
        )
        button_open.grid(row=0, column=(col := col + 1), padx=(0, 10))

        self.selected_channel = customtkinter.StringVar(value="None")
        self.channel_menu = customtkinter.CTkOptionMenu(
            master=frame,
            command=self.callback_channel_selected,
            variable=self.selected_channel,
            width=60,
        )
        self.channel_menu.grid(row=0, column=(col := col + 1), padx=(0, 10))
        self._update_channel_menu()

        self.skip_spikecontinuous_filter = customtkinter.BooleanVar(value=False)
        self.switch_filter = customtkinter.CTkSwitch(
            master=frame,
            text="Skip Filter",
            variable=self.skip_spikecontinuous_filter,
            command=lambda: self.__dict__.pop("spikecontinuous_filter", None),
            onvalue=True,
            offvalue=False,
            # width=80,
        )
        self.switch_filter.grid(row=0, column=(col := col + 1), padx=(0, 10))

        self.auto_load = customtkinter.BooleanVar(value=False)
        self.switch_auto_load = customtkinter.CTkSwitch(
            master=frame,
            text="Auto Load",
            variable=self.auto_load,
            onvalue=True,
            offvalue=False,
        )
        self.switch_auto_load.grid(row=0, column=(col := col + 1), padx=(0, 10))

        button_load = customtkinter.CTkButton(
            master=frame,
            text="Load",
            command=self._load_data,
            width=60,
        )
        button_load.grid(row=0, column=(col := col + 1), padx=(0, 10))

        self.show_all = customtkinter.BooleanVar(value=True)
        self.switch_show_all = customtkinter.CTkSwitch(
            master=frame,
            text="Plot all",
            variable=self.show_all,
            command=self.callback_show_all,
            onvalue=True,
            offvalue=False,
            width=80,
        )
        self.switch_show_all.grid(row=0, column=(col := col + 1), padx=(0, 10))

        self.selected_filename = customtkinter.StringVar(value="")
        self.select_file_menu = customtkinter.CTkOptionMenu(
            master=frame,
            command=self.callback_file_selected,
            variable=self.selected_filename,
            width=300,
        )
        self.select_file_menu.grid(row=0, column=(col := col + 1), padx=(0, 10))
        self._update_select_file_menu()

        self.indicate_selected_file = customtkinter.BooleanVar(value=False)
        switch_indicator = customtkinter.CTkSwitch(
            master=frame,
            text="Highlight",
            variable=self.indicate_selected_file,
            command=self._indicate_selected_file,
            onvalue=True,
            offvalue=False,
            width=80,
        )
        switch_indicator.grid(row=0, column=(col := col + 1), padx=(0, 10))

        button_raster = customtkinter.CTkButton(
            master=frame,
            text="Raster Preview",
            command=self.callback_preview_raster,
            width=60,
        )
        button_raster.grid(row=0, column=(col := col + 1), padx=(0, 10))

        # Spacer to push remaining buttons to the right:
        frame.columnconfigure(index=(col := col + 1), weight=1)

        button_reset = customtkinter.CTkButton(
            master=frame,
            text="Reset",
            command=self.reset_threshold,
            width=60,
        )
        button_reset.grid(row=0, column=(col := col + 1), padx=(10, 0))

        button_save = customtkinter.CTkButton(
            master=frame,
            text="Save",
            command=self.save_current_threshold,
            width=60,
        )
        button_save.grid(row=0, column=(col := col + 1), padx=(10, 0))

    def _init_keyboard_bindings(self):
        key_bindings = {
            "<KeyPress-Up>": dict(change=+1.0),
            "<KeyPress-Down>": dict(change=-1.0),
            "<Shift-KeyPress-Up>": dict(change=+5.0, round_=0),
            "<Shift-KeyPress-Down>": dict(change=-5.0, round_=0),
            "<Control-KeyPress-Up>": dict(change=+0.1),
            "<Control-KeyPress-Down>": dict(change=-0.1),
            "<Shift-Control-KeyPress-Up>": dict(change=+0.1, round_=1),
            "<Shift-Control-KeyPress-Down>": dict(change=-0.1, round_=1),
        }
        for sequence, options in key_bindings.items():
            self.bind(sequence, partial(self.callback_keyboard, **options))

    def generate_new_annotations(self):
        m = re.match(
            (
                "^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})"
                "_"
                "(?P<owl>[^_]+)"
                "_"
                "(?P<condition>.*)$"
            ),
            self.current_dirname,
        )
        session = {
            "condition": "",
            "date": "",
            "owl": "",
            "hemisphere": "",
        }
        if m is not None:
            session["date"] = f"{m.group('year')}-{m.group('month')}-{m.group('day')}"
            session["owl"] = m.group("owl")
            session["condition"] = m.group("condition")
        annotations = {
            "files": {
                os.path.basename(filename): {"thresholds": {}}
                for filename in self.filenames
            },
            "drop_channels": {},
            "regions": {},
            "session": session,
        }
        return annotations

    def get_threshold(self, channel_number: str) -> float | None:
        annotations = self.get_annotations()
        if not "files" in annotations:
            return None
        all_thresholds = [
            file_dict.get("thresholds", {}).get(channel_number, None)
            for file_dict in annotations["files"].values()
        ]
        if len(set(all_thresholds)) != 1:
            print(
                f"Multiple thresholds found for {channel_number = }: {all_thresholds}"
            )
        old_threshold = all_thresholds[0]
        if old_threshold is None:
            return None
        return float(old_threshold.removesuffix("e-6"))

    def get_annotations(self) -> dict:
        annotations_path = os.path.join(self.current_dir, "annotations.json")
        if os.path.isfile(annotations_path):
            with open(annotations_path, "r") as annotations_file:
                annotations = json.load(annotations_file)
        else:
            annotations = self.generate_new_annotations()
        return annotations

    def save_threshold(self, channel_number: str, threshold_microvolt: float) -> None:
        threshold_str = f"{threshold_microvolt:.1f}e-6"
        annotations = self.get_annotations()
        for filename, file_dict in annotations["files"].items():
            if not "thresholds" in file_dict:
                file_dict["thresholds"] = {}
            file_dict["thresholds"][channel_number] = threshold_str
        annotations_path = os.path.join(self.current_dir, "annotations.json")
        with open(annotations_path, "w") as annotations_file:
            json.dump(annotations, annotations_file, indent=4, sort_keys=True)
        self.last_threshold = threshold_microvolt

    def save_current_threshold(self):
        channel_number = self.selected_channel.get()
        current_threshold = self.threshold_var.get()
        self.save_threshold(channel_number, current_threshold)

    def threshold_updated(self):
        self.threshold_line.set_ydata(np.asarray([self.threshold_var.get()] * 2))
        self.bm.update()

    def react_slider_moved(self):
        new_val = self.threshold_var.get()
        self.threshold_text_var.set(f"{new_val:+.1f}")
        self.threshold_updated()

    def threshold_entry_changed(self, event: tkinter.Event):
        try:
            new_value = float(self.threshold_text_var.get())
        except ValueError:
            self.threshold_entry.configure(True, border_color="red")
            self.after(
                250,
                lambda: self.threshold_entry.configure(True, border_color="#979DA2"),
            )
        else:
            if new_value != self.threshold_var.get():
                self.threshold_var.set(new_value)
                self.threshold_negative.set(new_value < 0.0)
                self.threshold_updated()

    def threshold_sign(self):
        is_negative = self.threshold_negative.get()
        new_sign = -1 if is_negative else +1
        if is_negative:
            self.threshold_slider.configure(
                require_redraw=True,
                from_=-150,
                to=0,
            )
        else:
            self.threshold_slider.configure(
                require_redraw=True,
                from_=0,
                to=+150,
            )
        new_threshold = new_sign * abs(self.threshold_var.get())
        self.threshold_var.set(new_threshold)
        self.threshold_text_var.set(f"{new_threshold:+.1f}")
        self.threshold_updated()

    def callback_standard_deviation(self, multiplier: float) -> None:
        new_val = self.current_standard_deviation * -multiplier
        self.threshold_slider.set(new_val)
        self.threshold_text_var.set(f"{new_val:.1f}")
        self.react_slider_moved()

    def callback_keyboard(self, _: tkinter.Event, change: float, round_: float = -1.0):
        old_val = self.threshold_var.get()
        new_val = old_val + change
        if round_ >= 0:
            new_val = round(new_val, 1 if round_ is True else int(round_))
        self.threshold_slider.set(new_val)
        self.threshold_text_var.set(f"{new_val:.1f}")
        self.react_slider_moved()

    def callback_channel_selected(self, value):
        if self.selected_channel.get() == self.current_channel:
            # Nothing changed, return
            return
        if self.threshold_var.get() != self.last_threshold:
            response = messagebox.askyesnocancel(
                "Change channel", "Do you want to save the threshold first?"
            )
            if response is None:
                # Clicked "Cancel", do not switch channel
                self.selected_channel.set(self.current_channel or "")
                return
            if response is True:
                if isinstance(self.current_channel, str):
                    self.save_threshold(self.current_channel, self.threshold_var.get())
                else:
                    self.show_box_message("No channel selected to save")
                    self.after(1500, partial(self.show_box_message, ""))
            elif response is False:
                pass
        # Finally, switch channel and cleanup
        self.current_channel = self.selected_channel.get()
        self.reset_threshold()
        self.analog_signals.clear()
        self._redraw_trace()
        if self.auto_load.get():
            self._load_data()

    def reset_threshold(self):
        self.last_threshold = self.get_threshold(self.selected_channel.get())
        self.threshold_var.set(
            self.last_threshold
            if self.last_threshold is not None
            else self.DEFAULT_THRESHOLD
        )
        self.react_slider_moved()

    def callback_file_selected(self, value):
        if not self.show_all.get():
            self._redraw_trace()
        self._indicate_selected_file()

    def _indicate_selected_file(self):
        if not self.powl_files or not self.analog_signals:
            self.file_indicator.set_visible(False)
            return
        start_index = 0
        for pf in self.powl_files:
            if pf is self.selected_powlfile:
                break
            start_index += self.analog_signals[pf].continuous_signal.size
        stop_index = start_index + self.analog_signals[pf].continuous_signal.size  # type: ignore
        start_time, stop_time = indexes2times(
            np.array([start_index, stop_index]), fs=self.samplingrate
        )
        indicator_xy = self.file_indicator.get_xy()
        indicator_xy[:, 0] = [start_time, start_time, stop_time, stop_time, start_time]
        self.file_indicator.set_xy(indicator_xy)
        self.file_indicator.set_visible(
            self.show_all.get() and self.indicate_selected_file.get()
        )
        self.bm.update()

    def callback_show_all(self):
        self._redraw_trace()
        self._indicate_selected_file()

    @cached_property
    def samplingrate(self) -> float:
        if not self.powl_files:
            return 25000 / 1.024
        with self.powl_files[0] as file:
            global_params = get_params(file)
            if global_params.get("powl_version", "0.0.0").startswith("0."):  # type: ignore
                return global_params["adc_samplingrate"]  # type: ignore
            else:
                return global_params["samplingrates"]["traces"]  # type: ignore

    @property
    def spikecontinuous_filter(self):
        if self.skip_spikecontinuous_filter.get():
            return lambda x: x
        return get_spikecontinuous_filter(fs=self.samplingrate)

    def show_box_message(self, message=""):
        if message:
            self.messagebox.set_text(message)
            self.messagebox.set_visible(True)
        else:
            self.messagebox.set_text("")
            self.messagebox.set_visible(False)
        self.bm.update()

    def _load_data(self):
        self.analog_signals.clear()
        # self.selected_filename.set("")
        self._redraw_trace()
        if not hasattr(self, "powl_files") or not self.powl_files:
            self.show_box_message("No files opened.")
            return
        self.show_box_message("Loading...")
        channel_number = int(self.selected_channel.get())
        # print(f"Load channel {channel_number} from {len(self.powl_files)} files:")
        for i, pf in enumerate(self.powl_files):
            self.show_box_message(
                f"Channel {channel_number}\n"
                f"Loading... ({i+1}/{len(self.powl_files)} files)"
            )
            analog_unfiltered = AnalogSignal(
                pf, channel_number=channel_number, signal_key="traces"
            )
            analog_unfiltered.load()
            self.analog_signals[pf] = analog_unfiltered.copy_filtered(
                filter_func=self.spikecontinuous_filter,
                signal_key="spikecontinuous",
            )
        self.current_standard_deviation = (
            1e6
            * np.std(
                np.hstack(
                    [anas.continuous_signal for anas in self.analog_signals.values()]
                )
            ).item()
        )
        self.quick_set_3sd.configure(text=f"-{3 * self.current_standard_deviation:.1f}")
        self.quick_set_4sd.configure(text=f"-{4 * self.current_standard_deviation:.1f}")
        self._redraw_trace()

    def _redraw_trace(self):
        if hasattr(self, "h_data"):
            self.h_data.remove()
            del self.h_data
        self.canvas.draw()
        if not self.analog_signals or not self.selected_powlfile:
            self.show_box_message("No data to draw")
            return
        self.show_box_message("Drawing...")
        if self.show_all.get():
            self.last_shown = "ALL"
            one_trace = np.hstack(
                [anas.continuous_signal for anas in self.analog_signals.values()]
            )
        else:
            self.last_shown = self.selected_powlfile
            one_trace = self.analog_signals[self.selected_powlfile].continuous_signal
        times = np.arange(0, one_trace.size) / self.samplingrate
        self.h_data, *_ = self.ax_trace.plot(times, 1e6 * one_trace, color="w")
        ylim = min(500, max(160, 1.05 * 1e6 * np.max(np.abs(one_trace))))
        self.ax_trace.set_ylim(bottom=-ylim, top=ylim)
        self.ax_trace.set_xlim(-0.05 * times[-1], 1.05 * times[-1])
        self.canvas.draw()
        self.show_box_message()

    def callback_preview_raster(self):
        if not (pf := self.selected_powlfile):
            return
        anas = self.analog_signals[pf]
        current_threshold = self.threshold_var.get()
        spiketrains = spiketrains2times(
            split_indexes(
                detect_spikes(anas.continuous_signal, 1e-6 * current_threshold),
                anas.trials_start_stop,
            ),
            self.samplingrate,
        )
        fig, ax = plt.subplots(1, 1)
        ax.set_title(
            f"{os.path.basename(pf.filepath)} | channel {self.selected_channel.get()}"
            f"\n$threshold = {current_threshold:.1f} \\times 10^{{-6}}$",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Trial Index (presentation order)")
        eventplot_data = [spiketrains[trial_index] for trial_index in pf.trial_indexes]
        ax.eventplot(eventplot_data)
        fig.show()

    @property
    def current_dirname(self):
        if self.filenames:
            return os.path.basename(self.current_dir)
        else:
            return ""

    @property
    def current_dir(self):
        if self.filenames:
            return os.path.dirname(self.filenames[0])
        else:
            return ""

    def callback_open(self):
        filetypes = (
            ("pOwl Data Files", "*.h5"),
            ("pOwl annotations File", "annotations.json"),
            ("All files", "*.*"),
        )
        selectedfiles = filedialog.askopenfilenames(
            parent=self,
            title="Select pOwl Data Files",
            filetypes=filetypes,
        )
        if len(selectedfiles) == 1 and selectedfiles[0].endswith(".json"):
            annotations_path = selectedfiles[0]
            dirname = os.path.dirname(annotations_path)
            with open(annotations_path, "r") as f:
                annotations = json.load(f)
            selectedfiles = tuple(
                os.path.join(dirname, filename)
                for filename in annotations.get("files", {}).keys()
            )
        self.__dict__.pop("powl_files", None)
        self.__dict__.pop("samplingrate", None)
        if selectedfiles:
            self.filenames = tuple(selectedfiles)
            self.title(
                f"{self._TITLE} | {self.current_dirname} ({len(self.filenames)} files)"
            )
        else:
            self.title(self._TITLE)
        try:
            self._update_channel_menu()
        except KeyError:
            self.show_box_message("It appears some files could not be loaded.")
        else:
            self._update_select_file_menu()
            self.reset_threshold()
            self._redraw_trace()

    def _update_channel_menu(self):
        if self.filenames:
            channel_numbers = [f"{n}" for n in self.powl_files[0].channel_numbers()]
            self.channel_menu.configure(values=channel_numbers)
            self.selected_channel.set(channel_numbers[0])
        else:
            self.channel_menu.configure(values=[])
            self.selected_channel.set("")

    @property
    def selected_powlfile(self):
        current_filename = self.selected_filename.get()
        if not current_filename:
            return None
        for pf in self.powl_files:
            if os.path.basename(pf.filepath) == current_filename:
                return pf

    def _update_select_file_menu(self):
        if self.filenames:
            filenames = [os.path.basename(pf.filepath) for pf in self.powl_files]
            self.select_file_menu.configure(values=filenames)
            self.selected_filename.set(filenames[0])
        else:
            self.select_file_menu.configure(values=[])
            self.selected_filename.set("")

    @cached_property
    def powl_files(self):
        return [POwlFile(filename, mode="r") for filename in self.filenames]

    def callback_quit(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.quit()  # stops mainloop


def main():
    root = ThresholdingApp()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
