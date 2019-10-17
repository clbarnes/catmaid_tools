import numpy as np
import pytest

from skeleton_synapses.dto import SkeletonAssociationOutput
from skeleton_synapses.helpers.segmentation import get_border, get_node_associations
from skeleton_synapses.helpers.img_to_area import im_to_area_smooth, im_to_area_fast

import logging

logger = logging.getLogger(__name__)


EXPECTED_LENGTH = 300
TOLERANCE = 10

NEURON_SEG_ID = 1
SYNAPSE_SEG_ID = 2
NODE_ID = 3
SKELETON_ID = 4


@pytest.fixture
def neuron_segment() -> np.ndarray:
    """background 0, foreground 1"""
    img = np.zeros((400, 400), dtype=np.uint64)
    img[100:300, 100:300] = NEURON_SEG_ID
    return img


@pytest.fixture
def synapse_segment() -> np.ndarray:
    """background 0, foreground 2"""
    img = np.zeros((400, 400), dtype=np.uint64)
    img[50:150, 50:350] = SYNAPSE_SEG_ID
    return img


@pytest.fixture
def node_locations():
    return {
        NODE_ID: {
            "coords": {
                "x": 200,
                "y": 200
            },
            "treenode_id": NODE_ID,
            "skeleton_id": 4,
        }
    }


@pytest.fixture
def overlapping_segments():
    return {NEURON_SEG_ID: {SYNAPSE_SEG_ID}}


@pytest.mark.parametrize('fn', [im_to_area_fast, im_to_area_smooth])
def test_im_to_area(neuron_segment, synapse_segment, fn):
    neuron_border = get_border(neuron_segment == NEURON_SEG_ID)
    contact_pixels = neuron_border * (synapse_segment == SYNAPSE_SEG_ID)

    length = fn(contact_pixels)
    logger.debug("length is %s, expected around %s", length, EXPECTED_LENGTH)
    assert np.abs(length - EXPECTED_LENGTH) < TOLERANCE


def test_get_node_associations(neuron_segment, synapse_segment, node_locations, overlapping_segments):
    outputs = get_node_associations(synapse_segment, neuron_segment, node_locations, overlapping_segments)
    neuron_border = get_border(neuron_segment == NEURON_SEG_ID)
    contact_pixels = neuron_border * (synapse_segment == SYNAPSE_SEG_ID)
    length = im_to_area_smooth(contact_pixels)
    assert outputs == [SkeletonAssociationOutput(NODE_ID, SYNAPSE_SEG_ID, length)]
