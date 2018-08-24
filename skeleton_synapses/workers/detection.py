import logging

from hotqueue import HotQueue

from skeleton_synapses.helpers.files import Paths, TILE_SIZE
from skeleton_synapses.ilastik_utils.projects import setup_classifier
from skeleton_synapses.ilastik_utils.analyse import detect_synapses
from skeleton_synapses.constants import DETECTION_INPUT_QUEUE_NAME, DETECTION_OUTPUT_QUEUE_NAME
from skeleton_synapses.workers.common import setup_from_args

logger = logging.getLogger(__name__)


def main(parsed_args):
    setup_from_args(parsed_args)
    paths = Paths(parsed_args.credentials_path, parsed_args.input_dir, parsed_args.output_dir)
    debug_images = bool(parsed_args.debug_images)

    return _main(paths, debug_images)


def _main(paths, debug_images):
    from lazyflow.utility.timer import Timer

    opPixelClassification = setup_classifier(paths.description_json, paths.autocontext_ilp)
    in_queue = HotQueue(DETECTION_INPUT_QUEUE_NAME)
    out_queue = HotQueue(DETECTION_OUTPUT_QUEUE_NAME)

    time_logger = logging.getLogger(logger.name + '.timing')

    logger.info("Waiting for work items")
    for idx, tile_idx in enumerate(in_queue.consume()):  # breaks on None
        logger.info("Processing work item #%s: %s", idx, tile_idx)
        with Timer() as t:
            output = detect_synapses(TILE_SIZE, opPixelClassification, tile_idx)

        time_logger.debug("Tile timer: {}".format(t.seconds()))
        out_queue.put(output)

        logger.info("Waiting for work item #%s", idx+1)
