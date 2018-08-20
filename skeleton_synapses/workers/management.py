import logging

import numpy as np
from catpy import CatmaidClient
from hotqueue import HotQueue

from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import (
    DEFAULT_ROI_RADIUS_PX, DETECTION_OUTPUT_QUEUE_NAME, DETECTION_INPUT_QUEUE_NAME,
    ASSOCIATION_INPUT_QUEUE_NAME, ASSOCIATION_OUTPUT_QUEUE_NAME
)
from skeleton_synapses.helpers.files import ensure_list, Paths, get_algo_notes, TILE_SIZE, hash_algorithm
from skeleton_synapses.helpers.logging_ss import Timestamper
from skeleton_synapses.parallel.queues import (
    commit_tilewise_results_from_queue, commit_node_association_results_from_queue,
    populate_tile_input_queue, populate_synapse_queue
)

logger = logging.getLogger(__name__)


def main(parsed_args):
    paths = Paths(parsed_args.credentials_path, parsed_args.input_dir, parsed_args.output_dir)
    stack_id = parsed_args.stack_id
    roi_radius_px = parsed_args.roi_radius_px
    force = bool(parsed_args.force)
    skeleton_ids = parsed_args.skeleton_ids
    skip_detection = parsed_args.skip_detection
    skip_association = parsed_args.skip_association

    return _main(
        paths, stack_id, skeleton_ids, roi_radius_px, force,
        skip_detection=skip_detection, skip_association=skip_association
    )


def _main(paths, stack_id, skeleton_ids, roi_radius_px=DEFAULT_ROI_RADIUS_PX, force=False, **kwargs):
    logger.info("STARTING TILEWISE")

    catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(paths.credentials_json), stack_id)
    stack_info = catmaid.get_stack_info(stack_id)

    skeleton_ids = ensure_list(skeleton_ids)

    paths.initialise(catmaid, stack_info, skeleton_ids, force)

    algo_notes = get_algo_notes(paths.projects_dir)

    if force:
        logger.info('Using random hash')
        algo_hash = hash(np.random.random())
    else:
        algo_hash = hash_algorithm(paths.autocontext_ilp, paths.multicut_ilp)

    workflow_id = catmaid.get_workflow_id(
        stack_info['sid'], algo_hash, TILE_SIZE, detection_notes=algo_notes['synapse_detection']
    )

    logger.info('Populating tile queue')

    timestamper = Timestamper()

    if not kwargs.get("skip_detection"):
        timestamper.log('started detecting synapses')
        enqueue_and_submit_detections(catmaid, workflow_id, paths, stack_info, skeleton_ids, roi_radius_px)

    if not kwargs.get("skip_association"):
        timestamper.log('finished detecting synapses; started associating skeletons')
        enqueue_and_submit_associations(catmaid, workflow_id, stack_info, skeleton_ids, roi_radius_px, algo_hash, algo_notes)

    logger.info("DONE with skeletons.")


def enqueue_and_submit_detections(catmaid, workflow_id, paths, stack_info, skeleton_ids, roi_radius_px):
    node_infos = catmaid.get_node_infos(skeleton_ids, stack_info['sid'])

    tile_queue, tile_count = populate_tile_input_queue(
        catmaid, roi_radius_px, workflow_id, node_infos, HotQueue(DETECTION_INPUT_QUEUE_NAME)
    )

    if tile_count:
        output_queue = HotQueue(DETECTION_OUTPUT_QUEUE_NAME)
        commit_tilewise_results_from_queue(
            output_queue, paths.output_image_store, tile_count, TILE_SIZE, workflow_id, catmaid
        )

    else:
        logger.debug('No tiles found (probably already processed)')
        try:
            tile_queue.close()
        except AttributeError:
            pass


def enqueue_and_submit_associations(catmaid, workflow_id, stack_info, skeleton_ids, roi_radius_px, algo_hash, algo_notes):
    project_workflow_id = catmaid.get_project_workflow_id(
        workflow_id, algo_hash, association_notes=algo_notes['skeleton_association']
    )

    synapse_queue, synapse_count = populate_synapse_queue(
        catmaid, roi_radius_px, project_workflow_id, stack_info, skeleton_ids, HotQueue(ASSOCIATION_INPUT_QUEUE_NAME)
    )

    if synapse_count:
        logger.info('Segmenting {} synapse windows'.format(synapse_count))
        output_queue = HotQueue(ASSOCIATION_OUTPUT_QUEUE_NAME)

        commit_node_association_results_from_queue(output_queue, synapse_count, project_workflow_id, catmaid)

    else:
        logger.debug('No synapses required re-segmenting')
        try:
            synapse_queue.close()
        except AttributeError:
            pass
