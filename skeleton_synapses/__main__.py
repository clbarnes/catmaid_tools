#!/usr/bin/env python
import sys

import os

import argparse
import logging
import signal

import psutil
from pid import PidFile

from skeleton_synapses.constants import DEBUG, LOG_LEVEL, DEFAULT_ROI_RADIUS_PX
from skeleton_synapses.helpers.logging_ss import setup_logging
from skeleton_synapses.workers.common import clear_queues

logger = logging.getLogger(__name__)


def kill_child_processes(signum=None, frame=None):
    current_proc = psutil.Process()
    killed = []
    for child_proc in current_proc.children(recursive=True):
        logger.debug('Killing process: {} with status {}'.format(child_proc.name(), child_proc.status()))
        child_proc.kill()
        killed.append(child_proc.pid)
    clear_queues()
    logger.debug('Killed {} processes'.format(len(killed)))
    return killed


def create_parser():
    # optional arguments
    root_parser = argparse.ArgumentParser()
    root_parser.add_argument('-r', '--roi_radius_px', default=DEFAULT_ROI_RADIUS_PX,
                        help='The radius (in pixels) around each skeleton node to search for synapses')
    root_parser.add_argument('-f', '--force', type=int, default=0,
                        help="Whether to delete all prior results for a given skeleton: pass 1 for true or 0")
    root_parser.add_argument('-d', '--debug_images', action='store_true')
    root_parser.add_argument('--skip_detection', action="store_true", help="Whether to skip synapse detection")
    root_parser.add_argument('--skip_association', action="store_true",
                        help="Whether to skip skeleton-synapse association")
    root_parser.add_argument('--clear_queues', action="store_true")
    root_parser.set_defaults(fn=manage)

    subparsers = root_parser.add_subparsers(dest="subparser")

    # management arguments
    manage_parser = subparsers.add_parser("manage")
    add_paths_arguments(manage_parser)
    manage_parser.add_argument('stack_id',
                             help='ID or name of image stack in CATMAID')
    manage_parser.add_argument('skeleton_ids', nargs='+', type=int,
                             help="Skeleton IDs in CATMAID")

    # detection arguments
    detection_parser = subparsers.add_parser("detect")
    add_paths_arguments(detection_parser)
    detection_parser.set_defaults(fn=detect)

    # association arguments
    association_parser = subparsers.add_parser("associate")
    add_paths_arguments(association_parser)
    association_parser.add_argument('stack_id',
                             help='ID or name of image stack in CATMAID')
    association_parser.set_defaults(fn=associate)

    # gui arguments
    gui_parser = subparsers.add_parser("gui")
    gui_parser.set_defaults(fn=gui)

    return root_parser


def add_paths_arguments(parser):
    parser.add_argument('credentials_path',
                             help='Path to a JSON file containing CATMAID credentials (see credentials/example.json)')
    parser.add_argument('input_dir', help="A directory containing project files.")
    parser.add_argument('-o', '--output_dir',
                             help='A directory containing output files')


def detect(parsed_args):
    from skeleton_synapses.workers import detection
    detection.main(parsed_args)


def associate(parsed_args):
    from skeleton_synapses.workers import association
    association.main(parsed_args)


def manage(parsed_args):
    from skeleton_synapses.workers import management
    management.main(parsed_args)


def gui(parsed_args):
    from skeleton_synapses.workers import gui
    gui.main(parsed_args)


parser = create_parser()

if DEBUG:
    input_dir = "../projects-2017/L1-CNS"
    output_dir = input_dir
    cred_path = "credentials_dev.json"
    stack_id = 1
    skel_ids = [18531735]  # small test skeleton only on CLB's local instance

    parsed_args = parser.parse_args([
        cred_path, str(stack_id), output_dir, *skel_ids,
    ])
else:
    parsed_args = parser.parse_args()

instance_name = "{}_{}".format(
    parsed_args.subparser, '' if parsed_args.subparser in ['manage', 'gui'] else os.getpid()
)

with PidFile(pidname=instance_name, piddir=os.path.expanduser('~/.ss_pids')) as p:
    signal.signal(signal.SIGTERM, kill_child_processes)

    exit_code = 1

    try:
        parsed_args.fn(parsed_args)
        exit_code = 0
    except Exception as e:
        logger.exception('Errored, killing all child processes and exiting')
        kill_child_processes()
        raise
    finally:
        sys.exit(exit_code)
