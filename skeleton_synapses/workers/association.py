import logging

from hotqueue import HotQueue

from skeleton_synapses.helpers.files import Paths
from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.constants import ASSOCIATION_INPUT_QUEUE_NAME, ASSOCIATION_OUTPUT_QUEUE_NAME
from ilastik_utils.analyse import associate_skeletons
from ilastik_utils.projects import setup_classifier_and_multicut


logger = logging.getLogger(__name__)


def main(parsed_args):
    paths = Paths(parsed_args.credentials_path, parsed_args.input_dir, parsed_args.output_dir)
    stack_id = parsed_args.stack_id
    debug_images = parsed_args.debug_images

    return _main(paths, stack_id, debug_images)


def _main(paths, stack_id, debug_images):
    from lazyflow.utility.timer import Timer
    opPixelClassification, multicut_shell = setup_classifier_and_multicut(paths.description_json, paths.autocontext_ilp, paths.multicut_ilp)
    catmaid = CatmaidSynapseSuggestionAPI.from_json(paths.credentials_json, stack_id)

    in_queue = HotQueue(ASSOCIATION_INPUT_QUEUE_NAME)
    out_queue = HotQueue(ASSOCIATION_OUTPUT_QUEUE_NAME)

    time_logger = logging.getLogger(logger.name + '.timing')

    logger.info("Waiting for work items")
    for idx, skeleton_association_input in enumerate(in_queue.consume()):
        logger.info("Processing work item #%s: %s", idx, skeleton_association_input)
        with Timer() as t:
            skeleton_association_outputs = associate_skeletons(
                paths.output_image_store, opPixelClassification, multicut_shell, catmaid, skeleton_association_input
            )

        time_logger.debug("Tile timer: {}".format(t.seconds()))

        out_queue.put(skeleton_association_outputs)
        logger.info("Waiting for work item #%s", idx+1)
