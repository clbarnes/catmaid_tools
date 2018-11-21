from argparse import Namespace

import datetime as dt
import os

import subprocess as sp
from tkinter import (
    Frame, Label, Button, Entry, StringVar, IntVar, Tk, filedialog,
    Scale, HORIZONTAL, W
)
import logging

from hotqueue import DeHotQueue

from skeleton_synapses.constants import PROJECT_ROOT, DEFAULT_ROI_RADIUS_PX, DETECTION_INPUT_QUEUE_NAME, \
    ASSOCIATION_INPUT_QUEUE_NAME, DETECTION_OUTPUT_QUEUE_NAME, ASSOCIATION_OUTPUT_QUEUE_NAME, LAZYFLOW_RAM_VAR

logger = logging.getLogger(__name__)

DEBUG = True

INTERPRETER = ["python"] + (["-d"] if DEBUG else []) + ["-m"]


class WorkerManager:
    def __init__(self, process_args, qname):
        self.process_args = process_args
        self.queue = DeHotQueue(qname)
        self.processes = []
        self.logger = logging.getLogger(f'{__name__}.{type(self).__name__}')
        logger.info(f"Worker created with args {' '.join(process_args)} and queue {qname}")

    def prune(self):
        unfinished = []
        finished = dict()
        for process in self.processes:
            response = process.poll()
            if response is None:
                unfinished.append(process)
            else:
                finished[process.pid] = process.returncode

        self.processes = unfinished
        return finished

    def _start_process(self, n=1):
        for _ in range(n):
            self.processes.append(sp.Popen(self.process_args, env=os.environ))

    def _stop_process(self, n=1):
        if n is None:
            n = self.process_count

        for _ in range(n):
            self.queue.put_front(None)

    @property
    def process_count(self):
        self.prune()
        return len(self.processes)

    @process_count.setter
    def process_count(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Value must be an integer, got {type(value)}")

        self.prune()
        diff = value - self.process_count
        if diff < 0:
            self._stop_process(abs(diff))
        elif diff > 0:
            self._start_process(diff)

    def enqueue(self, *items):
        for item in items:
            self.logger.debug(f"enqueuing {item}")
        self.queue.put(*items)

    def clear_queue(self):
        self.queue.clear()

    def killall(self):
        for process in self.processes:
            process.kill()
        self.join()

    def join(self):
        for process in self.processes:
            process.wait()
        self.processes = []

    def __enter__(self):
        return self

    def close(self):
        self.process_count = 0
        self.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self.killall()


class DetectionWorkerManager(WorkerManager):
    def __init__(
        self, credentials_path, input_dir, roi_radius_px=DEFAULT_ROI_RADIUS_PX,
        force=False, debug_images=False, output_dir=None
    ):
        options = ["--roi_radius_px", str(roi_radius_px), "--force", str(int(force))]
        if debug_images:
            options.append('--debug_images')

        process_args = [
            *INTERPRETER,
            "skeleton_synapses",
            # os.path.join(PROJECT_ROOT, "skeleton_synapses"),
            *options, "detect", credentials_path, input_dir
        ]
        if output_dir:
            process_args.append(output_dir)

        super().__init__(process_args, DETECTION_INPUT_QUEUE_NAME)


class AssociationWorkerManager(WorkerManager):
    def __init__(
            self, credentials_path, input_dir, stack_id, roi_radius_px=DEFAULT_ROI_RADIUS_PX,
            force=False, debug_images=False, output_dir=None
    ):
        options = ["--roi_radius_px", str(roi_radius_px), "--force", str(int(force))]
        if debug_images:
            options.append('--debug_images')

        process_args = [
            *INTERPRETER,
            "skeleton_synapses",
            # os.path.join(PROJECT_ROOT, "skeleton_synapses"),
            *options, "associate", credentials_path, input_dir, str(stack_id)
        ]
        if output_dir:
            process_args.append(output_dir)

        super().__init__(process_args, ASSOCIATION_INPUT_QUEUE_NAME)


class SkelSynFrame(Frame):
    max_threads = os.cpu_count()

    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self.parent = parent
        self.parent.title("Skeleton Synapses")

        self.management_process = None

        self.detector_manager = None
        self.associator_manager = None
        self.started_at = None
        self.max_jobs = {"detection": -1, "association": -1}
        self.times = {"elapsed": dict(), "detection": dict(), "association": dict()}

        self.last_row = -1

        self._add_cred(kwargs.get("credentials_path"))
        self._add_io_dir(kwargs.get("input_dir"))
        self._add_radius(kwargs.get("roi_radius_px") or DEFAULT_ROI_RADIUS_PX)
        self._add_stack_id(kwargs.get("stack_id"))
        self._add_skel_ids(kwargs.get("skeleton_ids"))
        self._add_ram_per_proc(int(os.environ.get(LAZYFLOW_RAM_VAR, 0)))

        self._add_start()

        self._add_detector_count()
        self._add_associator_count()

        self._add_queue_counts()

        # todo: force (bool)?
        # todo: debug images (bool)?
        # todo: skip detection (bool)?
        # todo: skip association (bool)?
        # todo: clear queues (default True)

    def _add_cred(self, default=None):
        self.last_row += 1
        self.cred_var = StringVar(value=default)

        self.cred_label = Label(self.parent, text="Credentials: ")
        self.cred_label.grid(sticky=W, row=self.last_row, column=0)

        self.cred_entry = Entry(self.parent, textvariable=self.cred_var)
        self.cred_entry.grid(row=self.last_row, column=1)

        self.cred_browse = Button(self.parent, text='...', command=self.browse_cred)
        self.cred_browse.grid(row=self.last_row, column=2)

    def browse_cred(self):
        self.cred_var.set(filedialog.askopenfilename(
            parent=self.parent,
            defaultextension=".json",
            filetypes=[('JSON', '*.json')],
            initialdir=os.path.join(PROJECT_ROOT, 'config', 'credentials'),
            title="CATMAID credentials file"
        ))

    def _add_io_dir(self, default=None):
        self.last_row += 1
        self.io_dir_var = StringVar(value=default)

        self.io_dir_label = Label(self.parent, text="Input/output directory: ")
        self.io_dir_label.grid(sticky=W, row=self.last_row, column=0)

        self.io_dir_entry = Entry(self.parent, textvariable=self.io_dir_var)
        self.io_dir_entry.grid(row=self.last_row, column=1)

        self.io_dir_browse = Button(self.parent, text='...', command=self.browse_io_dir)
        self.io_dir_browse.grid(row=self.last_row, column=2)

    def browse_io_dir(self):
        self.io_dir_var.set(filedialog.askdirectory(
            parent=self.parent,
            initialdir=PROJECT_ROOT,
            title="Directory containing ILP files and output HDF5"
        ))

    def _add_radius(self, default=None):
        self.last_row += 1
        self.radius_var = IntVar(value=default)

        self.radius_label = Label(self.parent, text="ROI radius (px): ")
        self.radius_label.grid(sticky=W, row=self.last_row, column=0)

        self.radius_entry = Entry(self.parent, textvariable=self.radius_var)
        self.radius_entry.grid(row=self.last_row, column=1)

    def _add_stack_id(self, default=None):
        self.last_row += 1
        self.stack_id_var = IntVar()
        if default is not None:
            self.stack_id_var.set(default)

        self.stack_id_label = Label(self.parent, text="Stack ID: ")
        self.stack_id_label.grid(sticky=W, row=self.last_row, column=0)

        self.stack_id_entry = Entry(self.parent, textvariable=self.stack_id_var)
        self.stack_id_entry.grid(row=self.last_row, column=1)

    def _add_ram_per_proc(self, default=None):
        self.last_row += 1
        self.ram_var = IntVar()
        if default:
            self.ram_var.set(default)

        self.ram_label = Label(self.parent, text="RAM per worker (MB): ")
        self.ram_label.grid(sticky=W, row=self.last_row, column=0)

        self.ram_entry = Entry(self.parent, textvariable=self.ram_var)
        self.ram_entry.grid(row=self.last_row, column=1)

    def _add_skel_ids(self, default=None):
        self.last_row += 1
        if default:
            default = ','.join(str(item) for item in default)
        self.skel_ids_var = StringVar(value=default)

        self.skel_ids_label = Label(self.parent, text="Skeleton IDs: ")
        self.skel_ids_label.grid(sticky=W, row=self.last_row, column=0)

        self.skel_ids_entry = Entry(
            self.parent, textvariable=self.skel_ids_var, validatecommand=self.validate_skel_ids
        )
        self.skel_ids_entry.grid(row=self.last_row, column=1, columnspan=2)

    def validate_skel_ids(self):
        try:
            self.parse_skel_ids()
            return True
        except ValueError:
            self.bell()
            return False

    def parse_skel_ids(self):
        skids = [
            int(item.strip())
            for item in self.skel_ids_var.get().split(',')
            if item.strip()
        ]
        logger.info("Parsed skeleton ID input to %s", skids)
        return skids

    def _add_start(self):
        self.last_row += 1
        self.start_btn = Button(self.parent, text="START", command=self.start)
        self.start_btn.grid(row=self.last_row, column=0, columnspan=3)

    def start(self):
        if self.management_process is not None:
            raise RuntimeError("Cannot double-start the process")
        skids = self.parse_skel_ids()
        if not self.ram_var.get():
            logger.critical("RAM limit not set; set and try again")
            return
        os.environ[LAZYFLOW_RAM_VAR] = str(self.ram_var.get())
        self.management_process = sp.Popen([
            *INTERPRETER, "skeleton_synapses",
            "-r", str(self.radius_var.get()),
            "manage",
            self.cred_var.get(),
            self.io_dir_var.get(),
            str(self.stack_id_var.get()),
            *[str(s) for s in skids]
        ], env=os.environ)
        self.started_at = dt.datetime.now()
        self._disable_widgets(
            self.cred_entry, self.cred_browse,
            self.io_dir_entry, self.io_dir_browse,
            self.radius_entry, self.stack_id_entry,
            self.skel_ids_entry, self.start_btn
        )
        self.update_queue_counts()

    def _disable_widgets(self, *widgets):
        for widget in widgets:
            widget.config(state="disabled")

    def _add_detector_count(self):
        self.last_row += 1

        self.detector_label = Label(self.parent, text="Detector processes: ")
        self.detector_label.grid(sticky=W, row=self.last_row, column=0)

        self.detector_scale = Scale(
            self.parent, from_=0, to=self.max_threads, orient=HORIZONTAL,
            command=self.change_detector_count
        )
        self.detector_scale.set(0)
        self.detector_scale.grid(row=self.last_row, column=1, columnspan=2)

    def change_detector_count(self, new_val):
        new_val = int(new_val)
        ## not necessary right now because detection and
        # max_detectors = self.max_threads - self.associator_scale.get()
        # if max_detectors < new_val:
        #     self.detector_scale.set(max_detectors)
        #     return

        if self.detector_manager is None:
            self.detector_manager = DetectionWorkerManager(
                self.cred_var.get(), self.io_dir_var.get(), self.radius_var.get()
            )

        self.detector_manager.process_count = new_val

    def _add_associator_count(self):
        self.last_row += 1

        self.associator_label = Label(self.parent, text="Associator processes: ")
        self.associator_label.grid(sticky=W, row=self.last_row, column=0)

        self.associator_scale = Scale(
            self.parent, from_=0, to=self.max_threads, orient=HORIZONTAL,
            command=self.change_associator_count
        )
        self.associator_scale.set(0)
        self.associator_scale.grid(row=self.last_row, column=1, columnspan=2)

    def change_associator_count(self, new_val):
        new_val = int(new_val)
        # max_associators = self.max_threads - self.detector_scale.get()
        # if max_associators < new_val:
        #     self.associator_scale.set(max_associators)
        #     return

        if self.associator_manager is None:
            self.associator_manager = AssociationWorkerManager(
                self.cred_var.get(), self.io_dir_var.get(),
                self.stack_id_var.get(), self.radius_var.get()
            )

        self.associator_manager.process_count = new_val

    def _add_queue_counts(self):
        self.queue_counts = dict()
        # elapsed
        self.last_row += 1
        label = Label(self.parent, text="Time elapsed: ")
        label.grid(sticky=W, row=self.last_row, column=0)
        value = StringVar(value="unknown")
        display = Label(self.parent, textvariable=value)
        display.grid(row=self.last_row, column=1)
        self.times["elapsed"] = {
            "label": label,
            "value": value,
            "display": display
        }

        for name in ["detection", "association"]:
            self.queue_counts[name] = dict()
            for qtype in ["input", "output"]:
                self.last_row += 1
                self.queue_counts[qtype] = dict()
                label = Label(self.parent, text=f"{name.title()} {qtype}s: ")
                label.grid(sticky=W, row=self.last_row, column=0)
                value = IntVar(value=0)
                display = Label(self.parent, textvariable=value)
                display.grid(row=self.last_row, column=1)
                self.queue_counts[name][qtype] = {
                    "label": label,
                    "value": value,
                    "display": display
                }

            # time remaining
            self.last_row += 1
            label = Label(self.parent, text=f"{name.title()} time remaining: ")
            label.grid(sticky=W, row=self.last_row, column=0)
            value = StringVar(value="unknown")
            display = Label(self.parent, textvariable=value)
            display.grid(row=self.last_row, column=1)
            self.times[name] = {
                "label": label,
                "value": value,
                "display": display
            }

    def update_queue_counts(self):
        elapsed = dt.datetime.now() - self.started_at
        self.times["elapsed"]["value"].set(str(round_timedelta(elapsed)))

        # inputs
        for name, qname in [
            ("detection", DETECTION_INPUT_QUEUE_NAME),
            ("association", ASSOCIATION_INPUT_QUEUE_NAME),
        ]:
            length = len(DeHotQueue(qname))
            if length > 0:
                self.max_jobs[name] = max(length, self.max_jobs[name])
            self.queue_counts[name]["input"]["value"].set(length)
            ppn_done = (self.max_jobs[name] - length) / self.max_jobs[name]
            if ppn_done <= 0:
                continue
            remaining = round_timedelta(elapsed / ppn_done * (1 - ppn_done))
            self.times[name]["value"].set(str(remaining))

        # outputs
        for name, qname in [
            ("detection", DETECTION_OUTPUT_QUEUE_NAME),
            ("association", ASSOCIATION_OUTPUT_QUEUE_NAME),
        ]:
            length = len(DeHotQueue(qname))
            self.queue_counts[name]["output"]["value"].set(length)

        self.after(1000, self.update_queue_counts)


def round_timedelta(td: dt.timedelta, n_seconds: int=1):
    return dt.timedelta(seconds=round(td.total_seconds() / n_seconds))


def main(parsed_args: Namespace):
    root = Tk()
    window = SkelSynFrame(root, **vars(parsed_args))
    window.mainloop()
