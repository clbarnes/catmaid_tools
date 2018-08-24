import os

import logging
from hotqueue import HotQueue

from skeleton_synapses.helpers.logging_ss import setup_logging
from skeleton_synapses.constants import LOG_LEVEL, QUEUE_NAMES, DEBUG


def clear_queues():
    for qname in QUEUE_NAMES:
        HotQueue(qname).clear()


def setup_from_args(parsed_args):
    parsed_args.output_dir = parsed_args.output_dir or parsed_args.input_dir

    instance_name = "{}_{}".format(parsed_args.subparser, '' if parsed_args.subparser == 'manage' else os.getpid())
    setup_logging(parsed_args, LOG_LEVEL, instance_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.NOTSET)
    logger.info("Received arguments {}".format(parsed_args))
    if parsed_args.clear_queues:
        logger.info("Clearing queues")
        clear_queues()

    os.environ['SS_DEBUG_IMAGES'] = str(int(DEBUG) or os.environ.get('SS_DEBUG_IMAGES', 0) or parsed_args.debug_images)
