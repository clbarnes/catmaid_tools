import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# hack to make skeleton_synapses importable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from skeleton_synapses.ilastik_utils import projects
from skeleton_synapses.ilastik_utils import analyse
from skeleton_synapses.helpers.logging_ss import setup_logging

INPUT_DIR = os.path.expanduser('~/work/synapse_detection/projects-2018')

OFFSET_XYZ = np.array([13489, 20513, 2215])
SHAPE_XYZ = np.array([512, 512, 1])
COUNT_Z = 15

description_json_path = os.path.join(INPUT_DIR, "L1-CNS-description-NO-OFFSET.json")
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")
multicut_ilp_path = os.path.join(INPUT_DIR, "multicut", "L1-CNS-multicut.ilp")

assert os.path.isdir(INPUT_DIR), "no directory found at {}".format(INPUT_DIR)
assert os.path.isfile(description_json_path), "no L1-CNS-description-NO-OFFSET.json found in {}".format(INPUT_DIR)
assert os.path.isfile(autocontext_ilp_path), "no full-vol-autocontext.ilp found in {}".format(INPUT_DIR)


def plot_raw_pred_seg(raw_xy, predictions_xyc, segmentation_xy, fig_ax=None, show=False):
    fig, ax_arr = fig_ax or plt.subplots(1, 3)
    raw_ax, pred_ax, seg_ax = ax_arr.flatten()

    raw_ax.imshow(raw_xy, cmap="gray")
    pred_ax.imshow(predictions_xyc)
    seg_ax.imshow(segmentation_xy, cmap="Set1")
    fig.tight_layout()

    if show:
        plt.show()


if __name__ == '__main__':
    setup_logging('output/logs')
    opPixelClassification, multicut_shell = projects.setup_classifier_and_multicut(
        description_json_path, autocontext_ilp_path, multicut_ilp_path
    )

    raw_xy, predictions_xyc = analyse.fetch_and_predict([OFFSET_XYZ, OFFSET_XYZ + SHAPE_XYZ], opPixelClassification)
    segmentation_xy = analyse.segmentation_for_img(raw_xy, predictions_xyc, multicut_shell.workflow)

    plot_raw_pred_seg(raw_xy, predictions_xyc, segmentation_xy, show=True)
