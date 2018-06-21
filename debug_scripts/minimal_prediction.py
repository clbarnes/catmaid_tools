#!/usr/bin/env python
import os
import sys
import logging

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import psutil

RAM_LIMIT_MB = 1500

# hack to make skeleton_synapses importable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
os.environ["LAZYFLOW_TOTAL_RAM_MB"] = str(RAM_LIMIT_MB)

# imports need to be in this order
from skeleton_synapses.ilastik_utils import projects
from skeleton_synapses.ilastik_utils import analyse
# from lazyflow.operators.cacheMemoryManager import CacheMemoryManager
# from lazyflow.utility import Memory

# logging.basicConfig(level=logging.WARNING)
# logger = logging.getLogger()

# should contain:
# L1-CNS-description-NO-OFFSET.json
# full-vol-autocontext.ilp
INPUT_DIR = os.path.expanduser('~/work/synapse_detection/projects-2018')

# ROI
OFFSET_XYZ = np.array([13489, 20513, 2215])
SHAPE_XYZ = np.array([512, 512, 10])
COUNT_Z = 4

description_json_path = os.path.join(INPUT_DIR, "L1-CNS-description-NO-OFFSET.json")
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")

assert os.path.isdir(INPUT_DIR), "no directory found at {}".format(INPUT_DIR)
assert os.path.isfile(description_json_path), "no L1-CNS-description-NO-OFFSET.json found in {}".format(INPUT_DIR)
assert os.path.isfile(autocontext_ilp_path), "no full-vol-autocontext.ilp found in {}".format(INPUT_DIR)

print(os.environ["LAZYFLOW_TOTAL_RAM_MB"])
opPixelClassification = projects.setup_classifier(description_json_path, autocontext_ilp_path)
# assert Memory.getAvailableRam(), "RAM limit not picked up"
# cache_manager = CacheMemoryManager()
#
# cache_manager.setRefreshInterval(10)

# print("{}MB available".format(Memory.getAvailableRam() // 1e6))
def print_memory(x):
    print("Lazyflow thinks it's using {}MB of {}MB".format(x//1e6, RAM_LIMIT_MB))

# cache_manager.totalCacheMemory.subscribe(print_memory)

proc = psutil.Process()
RAM = [proc.memory_info().rss / 1e6]
print("Latest memory use: {}MB".format(int(RAM[-1])))
out = []
for idx in tqdm(range(COUNT_Z)):
    # if idx == START_MANAGING_AT:
    #     CacheMemoryManager()
    #     tqdm.write("CacheMemoryManager started")

    roi_xyz = np.array([OFFSET_XYZ + [0, 0, idx*SHAPE_XYZ[-1]], OFFSET_XYZ + [0, 0, idx*SHAPE_XYZ[-1]] + SHAPE_XYZ])
    raw_xy, pred_xyc = analyse.fetch_and_predict(roi_xyz, opPixelClassification)
    this_ram = int(proc.memory_info().rss / 1e6)
    RAM.append(this_ram)
    available = RAM_LIMIT_MB  # int(Memory.getAvailableRam() / 1e6)
    tqdm.write("Latest memory use: {}MB of {}MB".format(this_ram, available))

# fig, ax_arr = plt.subplots(1, 3)
# raw_ax, pred_ax, ram_ax = ax_arr.flatten()
#
# raw_ax.imshow(out[-1][0], cmap="gray")
# pred_ax.imshow(out[-1][1])

fig, ram_ax = plt.subplots(1, 1)
ram_ax.plot(RAM, label="RAM usage")
ram_ax.set_xlabel("iteration")
ram_ax.set_ylabel("RAM usage (MB)")

if RAM_LIMIT_MB:
    ram_ax.axhline(RAM_LIMIT_MB, linestyle="--", label="RAM limit")
# if START_MANAGING_AT:
#     ram_ax.axvline(START_MANAGING_AT, linestyle="--", label="memory manager started")

ram_ax.legend()

plt.show()
