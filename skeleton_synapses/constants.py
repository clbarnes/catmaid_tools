import logging
import os
from warnings import warn

DEBUG = bool(int(os.getenv('SS_DEBUG', 0)))
ALGO_HASH = "2018-09-05T13:20:00"  # set to fix algorithm hash
LOG_LEVEL = logging.DEBUG

DEFAULT_THREADS = 3
DEFAULT_RAM_MB_PER_PROCESS = 1200

DEFAULT_ROI_RADIUS_PX = 150

DEFAULT_SYNAPSE_DISTANCE_NM = 600

TQDM_KWARGS = {
    'ncols': 80,
}

RESULTS_TIMEOUT_SECONDS = 5*60  # result fetchers time out after 5 minutes

# ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
PACKAGE_ROOT = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

THREADS = int(os.getenv('SYNAPSE_DETECTION_THREADS', DEFAULT_THREADS))
RAM_MB_PER_PROCESS = int(os.getenv('SYNAPSE_DETECTION_RAM_MB_PER_PROCESS', DEFAULT_RAM_MB_PER_PROCESS))

LAZYFLOW_RAM_VAR = 'LAZYFLOW_TOTAL_RAM_MB'

MONITOR_HOST = 'localhost'
MONITOR_PORT = int(os.getenv('MONITOR_PORT', 8088))
MONITOR_INTERVAL = 10

REQUEST_THREADS = 4

WORKER_TIMEOUT = 5  # seconds

# hotqueue names
DETECTION_INPUT_QUEUE_NAME = 'detection_input'
DETECTION_OUTPUT_QUEUE_NAME = 'detection_output'
ASSOCIATION_INPUT_QUEUE_NAME = 'association_input'
ASSOCIATION_OUTPUT_QUEUE_NAME = 'association_output'
QUEUE_NAMES = [
    DETECTION_INPUT_QUEUE_NAME,
    DETECTION_OUTPUT_QUEUE_NAME,
    ASSOCIATION_INPUT_QUEUE_NAME,
    ASSOCIATION_OUTPUT_QUEUE_NAME
]

ILP_RETRAIN = bool(int(os.getenv('SYNAPSE_DETECTION_RETRAIN', 0)))
ILP_READONLY = bool(int(os.getenv('SYNAPSE_DETECTION_READONLY_ILP', 1)))
if ILP_RETRAIN and ILP_READONLY:
    warn('ILP must be writable if it is to retrain. Disabling read-only mode')
    ILP_READONLY = False
