import os

import subprocess as sp
from tkinter import (
    Frame, Label, Button, Entry, StringVar, IntVar, Tk, filedialog,
    Scale, HORIZONTAL
)
import logging

from hotqueue import DeHotQueue

from skeleton_synapses.constants import PROJECT_ROOT, DEFAULT_ROI_RADIUS_PX, DETECTION_INPUT_QUEUE_NAME, \
    ASSOCIATION_INPUT_QUEUE_NAME

logger = logging.getLogger(__name__)

QNAME = 'tk_queue'
WORKER_PATH = 'worker_script.py'

MAX = 5


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
            self.processes.append(sp.Popen(self.process_args))

    def _stop_process(self, n=1):
        if n is None:
            n = self.process_count

        for _ in range(n):
            self.queue.put_front(None)

    @property
    def process_count(self):
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
            "python", "-m", os.path.join(PROJECT_ROOT, "skeleton_synapses"),
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
            "python", "-m", os.path.join(PROJECT_ROOT, "skeleton_synapses"),
            *options, "detect", credentials_path, input_dir, str(stack_id)
        ]
        if output_dir:
            process_args.append(output_dir)

        super().__init__(process_args, ASSOCIATION_INPUT_QUEUE_NAME)


class SkelSynFrame(Frame):
    max_threads = os.cpu_count()

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.title("Skeleton Synapses")

        self.management_process = None

        self.detector_manager = None
        self.associator_manager = None

        self.last_row = -1

        self._add_cred()
        self._add_io_dir()
        self._add_radius()
        self._add_stack_id()
        self._add_skel_ids()

        self._add_start()

        self._add_detector_count()
        self._add_associator_count()

        # todo: force (bool)?
        # todo: debug images (bool)?
        # todo: skip detection (bool)?
        # todo: skip association (bool)?
        # todo: clear queues (default True)

    def _add_cred(self):
        self.last_row += 1
        self.cred_var = StringVar()

        self.cred_label = Label(self.parent, text="Credentials: ")
        self.cred_label.grid(row=self.last_row, column=0)

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

    def _add_io_dir(self):
        self.last_row += 1
        self.io_dir_var = StringVar()

        self.io_dir_label = Label(self.parent, text="Input/output directory: ")
        self.io_dir_label.grid(row=self.last_row, column=0)

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

    def _add_radius(self):
        self.last_row += 1
        self.radius_var = IntVar(value=DEFAULT_ROI_RADIUS_PX)

        self.radius_label = Label(self.parent, text="ROI radius (px): ")
        self.radius_label.grid(row=self.last_row, column=0)

        self.radius_entry = Entry(self.parent, textvariable=self.radius_var)
        self.radius_entry.grid(row=self.last_row, column=1)

    def _add_stack_id(self):
        self.last_row += 1
        self.stack_id_var = IntVar(value=1)

        self.stack_id_label = Label(self.parent, text="Stack ID: ")
        self.stack_id_label.grid(row=self.last_row, column=0)

        self.stack_id_entry = Entry(self.parent, textvariable=self.stack_id_var)
        self.stack_id_entry.grid(row=self.last_row, column=1)

    def _add_skel_ids(self):
        self.last_row += 1
        self.skel_ids_var = StringVar()

        self.skel_ids_label = Label(self.parent, text="Skeleton IDs: ")
        self.skel_ids_label.grid(row=self.last_row, column=0)

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
        return [int(item.strip()) for item in self.skel_ids_var.get() if item.strip()]

    def _add_start(self):
        self.last_row += 1
        self.start_btn = Button(self.parent, text="START", command=self.start)

    def start(self):
        if self.management_process is not None:
            raise RuntimeError("Cannot double-start the process")
        skids = self.parse_skel_ids()
        self.management_process = sp.Popen([
            "python", "-m", "skeleton_synapses",
            "-r", str(self.radius_var.get()),
            "manage", str(self.stack_id_var.get()), *skids
        ])
        self._disable_widgets(
            self.cred_entry, self.cred_browse,
            self.io_dir_entry, self.io_dir_browse,
            self.radius_entry, self.stack_id_entry,
            self.skel_ids_entry, self.start_btn
        )

    def _disable_widgets(self, *widgets):
        for widget in widgets:
            widget.config(state="disabled")

    def _add_detector_count(self):
        self.last_row += 1

        self.detector_label = Label(self.parent, text="Detector processes: ")
        self.detector_label.grid(row=self.last_row, column=0)

        self.detector_scale = Scale(
            self.parent, from_=0, to=self.max_threads, orient=HORIZONTAL,
            label=self.detector_label, command=self.change_detector_count
        )
        self.detector_scale.set(0)
        self.detector_scale.grid(row=self.last_row, column=1, columnspan=2)

    def change_detector_count(self):
        max_detectors = self.max_threads - self.associator_scale.get()
        if max_detectors < self.detector_scale.get():
            self.detector_scale.set(max_detectors)
            return

        if self.detector_manager is None:
            self.detector_manager = DetectionWorkerManager(
                self.cred_var.get(), self.io_dir_var.get(), self.radius_var.get()
            )

        self.detector_manager.process_count = self.detector_scale.get()

    def _add_associator_count(self):
        self.last_row += 1

        self.associator_label = Label(self.parent, text="Associator processes: ")
        self.associator_label.grid(row=self.last_row, column=0)

        self.associator_scale = Scale(
            self.parent, from_=0, to=self.max_threads, orient=HORIZONTAL,
            label=self.associator_label, command=self.change_associator_count
        )
        self.associator_scale.set(0)
        self.associator_scale.grid(row=self.last_row, column=1, columnspan=2)

    def change_associator_count(self):
        max_associators = self.max_threads - self.detector_scale.get()
        if max_associators < self.associator_scale.get():
            self.associator_scale.set(max_associators)
            return

        if self.associator_manager is None:
            self.associator_manager = AssociationWorkerManager(
                self.cred_var.get(), self.io_dir_var.get(),
                self.stack_id_var.get(), self.radius_var.get()
            )

        self.associator_manager.process_count = self.associator_scale.get()


def main(parsed_args):
    root = Tk()
    window = SkelSynFrame(root)
    window.mainloop()
