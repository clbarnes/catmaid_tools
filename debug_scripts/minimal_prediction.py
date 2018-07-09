#!/usr/bin/env python
import multiprocessing as mp
# if __name__ == "__main__":
#     mp.set_start_method("forkserver")
import os
import sys
import logging

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import psutil

RAM_LIMIT_MB = 1500

logger = logging.getLogger(__name__)

# hack to make skeleton_synapses importable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(RAM_LIMIT_MB)

# should contain:
# L1-CNS-description-NO-OFFSET.json
# full-vol-autocontext.ilp
INPUT_DIR = os.path.expanduser('~/work/synapse_detection/projects-2018')

# ROI
OFFSET_XYZ = np.array([13489, 20513, 2215])
SHAPE_XYZ = np.array([512, 512, 1])
COUNT_Z = 15

REFRESH_INTERVAL = 5


def b_to_MB(b):
    return int(b / 1e6)


def setup_and_segment():
    # imports need to be in this order
    from skeleton_synapses.ilastik_utils import projects
    from skeleton_synapses.ilastik_utils import analyse
    # from skeleton_synapses.helpers.logging_ss import setup_logging
    from lazyflow.request import Request
    from lazyflow.utility import Memory

    logger.debug("Setting up classifier")
    opPixelClassification = projects.setup_classifier(description_json_path, autocontext_ilp_path)
    Request.reset_thread_pool(8)
    logger.info("Segment function allowed %sMB of RAM", b_to_MB(Memory.getAvailableRam()))
    # mgr = CacheMemoryManager()
    # time.sleep(3)
    # assert mgr.is_alive()
    # mgr.setRefreshInterval(REFRESH_INTERVAL)

    def fn(x):
        current_mgr = b_to_MB(x)
        current_memory = b_to_MB(Memory.getMemoryUsage())
        available = b_to_MB(Memory.getAvailableRam())
        logger.info("Using %sMB (believes %sMB) out of %sMB", current_memory, current_mgr, available)

    # mgr.totalCacheMemory.subscribe(fn)

    logger.debug("Finished setting up classifier")

    proc = psutil.Process()
    RAM = [proc.memory_info().rss / 1e6]
    for idx in tqdm(range(COUNT_Z)):
        logger.debug("Calculating ROI %s", idx)
        roi_xyz = np.array([
            OFFSET_XYZ + [0, 0, idx*SHAPE_XYZ[-1]],
            OFFSET_XYZ + [0, 0, idx*SHAPE_XYZ[-1]] + SHAPE_XYZ
        ])
        logger.debug("Addressing ROI %s", roi_xyz)

        raw_xy, pred_xyc = analyse.fetch_and_predict(roi_xyz, opPixelClassification)
        logger.debug("Finished addressing ROI %s", roi_xyz)
        this_ram = int(proc.memory_info().rss / 1e6)
        RAM.append(this_ram)
        available = RAM_LIMIT_MB  # int(Memory.getAvailableRam() / 1e6)
        tqdm.write("Latest memory use: {}MB of {}MB".format(this_ram, available))

    return (raw_xy, pred_xyc), RAM


def setup_and_segment_in_pool(n=None):
    with mp.Pool(n or mp.cpu_count()) as pool:
        return pool.apply(setup_and_segment)


class PredProc(mp.Process):
    def run(self):
        setup_and_segment()


def setup_and_segment_in_process():
    proc = PredProc()
    proc.start()
    proc.join()


def plot_ram(ram_MB, limit_MB=None, fig_ax=None, show=False):
    fig, ax = fig_ax or plt.subplots(1, 1)
    ax.plot(ram_MB, label="RAM usage")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RAM usage (MB)")

    if limit_MB:
        ax.axhline(RAM_LIMIT_MB, linestyle="--", color="orange", label="RAM limit")

    ax.legend()
    if show:
        plt.show()


def plot_ims(raw_xy, pred_xy, fig_axes=None, show=False):
    fig, (ax1, ax2) = fig_axes or plt.subplots(1, 2)
    ax1.imshow(raw_xy, cmap="gray")
    ax2.imshow(pred_xy)

    if show:
        plt.show()


description_json_path = os.path.join(INPUT_DIR, "L1-CNS-description-NO-OFFSET.json")
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")

assert os.path.isdir(INPUT_DIR), "no directory found at {}".format(INPUT_DIR)
assert os.path.isfile(description_json_path), "no L1-CNS-description-NO-OFFSET.json found in {}".format(INPUT_DIR)
assert os.path.isfile(autocontext_ilp_path), "no full-vol-autocontext.ilp found in {}".format(INPUT_DIR)

print(os.environ["LAZYFLOW_TOTAL_RAM_MB"])


if __name__ == '__main__':
    # setup_logging('output', level=logging.DEBUG)
    # logger.info("Main thread allowed %sMB of RAM", b_to_MB(Memory.getAvailableRam()))

    # opPixelClassification = projects.setup_classifier(description_json_path, autocontext_ilp_path)

    # ims, results = setup_and_segment()
    ims, results = setup_and_segment_in_pool(1)
    plot_ram(results, RAM_LIMIT_MB)
    plot_ims(*ims)
    plt.show()

    # setup_and_segment_in_process()
