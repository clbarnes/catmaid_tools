import os
import time

from datetime import datetime
from queue import Empty

from unittest import mock
from hotqueue import HotQueue

import pytest

from tests.context import skeleton_synapses

from skeleton_synapses.dto import SkeletonAssociationOutput
from skeleton_synapses.parallel.queues import (
    iterate_queue, commit_node_association_results_from_queue, QueueOverpopulatedException
)


TIMEOUT = 1  # must be an integer because redis is dumb


@pytest.fixture
def catmaid():
    catmaid = mock.Mock()
    catmaid.add_synapse_treenode_associations = mock.Mock()
    return catmaid


@pytest.fixture
def hot_queue(request):
    timestamp = datetime.now().isoformat()
    qname = "{}_{}".format(request.node.name, timestamp)
    q = HotQueue(qname)
    q.clear()
    yield q
    q.clear()


def populate_queue(items, queue, poll_interval=0.01):
    """

    Parameters
    ----------
    items : sequence or int
    queue : HotQueue
    poll_interval : float

    Returns
    -------

    """
    try:
        len(items)
    except TypeError:
        items = [1 for _ in range(items)]
    for item in items:
        queue.put(item)
    while len(queue) < len(items):
        time.sleep(poll_interval)


def test_iterate_queue(hot_queue):
    item_count = final_size = 5
    populate_queue(item_count, hot_queue)

    results = list(iterate_queue(hot_queue, final_size, timeout=TIMEOUT))
    assert len(results) == final_size
    assert sum(results) == final_size


def test_iterate_queue_underpopulated(hot_queue):
    item_count = 3
    final_size = 5
    populate_queue(item_count, hot_queue)

    with pytest.raises(Empty):
        for idx, result in enumerate(iterate_queue(hot_queue, final_size, timeout=TIMEOUT)):
            assert result
            assert idx < item_count


def test_iterate_queue_overpopulated(hot_queue):
    item_count = 7
    final_size = 5
    populate_queue(item_count, hot_queue)

    with pytest.raises(QueueOverpopulatedException):
        for idx, result in enumerate(iterate_queue(hot_queue, final_size, timeout=TIMEOUT)):
            assert result
            assert idx < final_size


def test_commit_node_association_results_from_queue(catmaid, hot_queue):
    item_count = 10
    items = [
        SkeletonAssociationOutput('tnid{}'.format(i), 'ssid{}'.format(i), 'contact{}'.format(i)) for i in range(item_count)
    ]
    item_chunkings = [slice(None, 3), slice(3, 5), slice(5, None)]

    expected_args = [('ssid{}'.format(i), 'tnid{}'.format(i), 'contact{}'.format(i)) for i in range(item_count)]
    expected_args_chunks = [expected_args[chunk] for chunk in item_chunkings]

    item_chunks = [items[chunk] for chunk in item_chunkings]
    populate_queue(item_chunks, hot_queue)
    commit_node_association_results_from_queue(hot_queue, len(item_chunks), None, catmaid)

    for arg_chunk in expected_args_chunks:
        catmaid.add_synapse_treenode_associations.assert_any_call(arg_chunk, None)


if __name__ == '__main__':
    pytest.main(['-v', os.path.realpath(__file__)])
