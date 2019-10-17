from collections import defaultdict
import logging

from skimage.morphology import skeletonize, binary_erosion
import numpy as np
import vigra

from skeleton_synapses.dto import SkeletonAssociationOutput
from skeleton_synapses.helpers.img_to_area import im_to_area_smooth, im_to_area_fast


logger = logging.getLogger(__name__)


def show_im(img):
    from matplotlib import pyplot as plt
    _, ax = plt.subplots()
    ax.imshow(img)
    plt.show()


def get_synapse_segment_overlaps(synapse_cc_xy, segmentation_xy, synapse_slice_ids):
    """
    Find the neuron segment: synapse slice ID intersections

    Parameters
    ----------
    synapse_cc_xy : vigra.VigraArray
        Synapse slice image
    segmentation_xy : vigra.VigraArray
        Neuron segmentation
    synapse_slice_ids : list

    Returns
    -------
    dict
        {neuron segment : set of synapse slice IDs}
    """
    # todo: test
    overlapping_segments = dict()
    for synapse_slice_id in synapse_slice_ids:
        # todo: need to cast some types?
        segments = np.unique(segmentation_xy[synapse_cc_xy == synapse_slice_id])
        for overlapping_segment in segments:
            if overlapping_segment not in overlapping_segments:
                overlapping_segments[overlapping_segment] = set()
            overlapping_segments[overlapping_segment].add(synapse_slice_id)

    return overlapping_segments


def get_border(binary_img):
    return skeletonize(np.logical_xor(binary_img, binary_erosion(binary_img)))


def get_node_associations(synapse_cc_xy, segmentation_xy, node_locations, overlapping_segments):
    """

    Parameters
    ----------
    synapse_cc_xy : vigra.VigraArray
    segmentation_xy : vigra.VigraArray
    node_locations : dict
        dict whose values are a dicts containing a 'coords' dict (relative within this image) and a 'treenode_id' value
    overlapping_segments : dict
        Neuron segment to synapse slice ID

    Returns
    -------
    list of SkeletonAssociationOutput
    """
    # todo: test
    node_locations_arr = node_locations_to_array(synapse_cc_xy.shape, node_locations)
    where_nodes_exist = node_locations_arr >= 0

    outputs = []

    segment_to_nodes_data = defaultdict(list)
    for node, segment in zip(
        node_locations_arr[where_nodes_exist],
        segmentation_xy[where_nodes_exist],
    ):
        segment_to_nodes_data[segment].append(node_locations[node])

    for segment_id, nodes_data in segment_to_nodes_data.items():
        # in this particular segment, skeleton ID to smallest node ID
        skels_in_seg = {
            d["skeleton_id"]: d["treenode_id"] for d in sorted(nodes_data, key=lambda x: x["treenode_id"], reverse=True)
        }

        if len(skels_in_seg) > 1:
            logger.warning(
                "Treenodes of different skeletons found in the same neuron segment:\n\t%s",
                '\n\t'.join(
                    "treenode {}, skeleton {}".format(skid, tnid)
                    for skid, tnid in skels_in_seg.items()
                )
            )

        for node_id in skels_in_seg.values():
            segment = segmentation_xy == segment_id
            segment_border = get_border(segment)

            for synapse_slice_id in overlapping_segments.get(segment_id, []):
                contact_img = segment_border * (synapse_cc_xy == synapse_slice_id)
                contact_length = im_to_area_smooth(contact_img)
                outputs.append(SkeletonAssociationOutput(node_id, synapse_slice_id, contact_length))

    return outputs


def node_locations_to_array(arr_shape, node_locations):
    """
    Given a vigra image in xy and a dict containing xy coordinates, return a vigra image of the same shape, where nodes
    are represented by their integer ID, and every other pixel is -1.

    Parameters
    ----------
    arr_shape : tuple

    node_locations : dict
        dict whose values are a dicts containing a 'coords' dict (relative within this image) and a 'treenode_id' value

    Returns
    -------
    vigra.VigraArray
    """
    arr_xy = vigra.VigraArray(arr_shape, dtype=np.int64, value=-1, axistags=vigra.AxisTags('xy'))

    for node_location in node_locations.values():
        coords = node_location['coords']
        arr_xy[coords['x'], coords['y']] = int(node_location['treenode_id'])

    return arr_xy
