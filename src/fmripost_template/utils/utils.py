# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utility functions for fMRIPost-template."""

import logging

LGR = logging.getLogger(__name__)


def _get_wf_name(bold_fname, prefix):
    """Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "template")
    'template_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = '_'.join(fname.split('_')[1:-1])
    return f'{prefix}_{fname_nosub.replace("-", "_")}_wf'


def update_dict(orig_dict, new_dict):
    """Update dictionary with values from another dictionary.

    Parameters
    ----------
    orig_dict : dict
        Original dictionary.
    new_dict : dict
        Dictionary with new values.

    Returns
    -------
    updated_dict : dict
        Updated dictionary.
    """
    updated_dict = orig_dict.copy()
    for key, value in new_dict.items():
        if (orig_dict.get(key) is not None) and (value is not None):
            print(f'Updating {key} from {orig_dict[key]} to {value}')
            updated_dict[key].update(value)
        elif value is not None:
            updated_dict[key] = value

    return updated_dict


def find_shortest_path(space_pairs, start, end):
    """Find the shortest path between two spaces in a list of space pairs.

    Parameters
    ----------
    space_pairs : list of tuples
        List of tuples where each tuple contains two spaces of the form (from, to).
    start : str
        The starting space.
    end : str
        The ending space.

    Returns
    -------
    list
        List of indices that represent the shortest path between the two spaces.

    Raises
    ------
    ValueError
        If no path exists between the two spaces.

    Examples
    --------
    >>> space_pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    >>> find_shortest_path(space_pairs, "a", "d")
    [0, 2]
    """
    from collections import deque

    # Create a graph from the space pairs and keep track of indices
    graph = {}
    index_map = {}
    for index, (src, dst) in enumerate(space_pairs):
        if src not in graph:
            graph[src] = []
        graph[src].append(dst)
        if src not in index_map:
            index_map[src] = []
        index_map[src].append(index)

    # Perform BFS to find the shortest path
    queue = deque([(start, [start], [])])
    visited = set()

    while queue:
        current, path, indices = queue.popleft()
        if current == end:
            return indices
        if current not in visited:
            visited.add(current)
            for neighbor, idx in zip(
                graph.get(current, []), index_map.get(current, []), strict=False
            ):
                queue.append((neighbor, path + [neighbor], indices + [idx]))

    raise ValueError(f'No path exists between {start} and {end}')


def get_transforms(source, target, local_transforms=None):
    """Get the transforms required to go from source to target space."""
    import templateflow.api as tflow
    from bids.layout import Entity, parse_file_entities

    query = [
        Entity('template', 'tpl-([a-zA-Z0-9]+)'),
        Entity('from', 'from-([a-zA-Z0-9]+)'),
    ]

    all_transforms = local_transforms or []

    templates = tflow.get_templates()
    for template in templates:
        template_transforms = tflow.get(template, suffix='xfm', extension='h5')
        if not isinstance(template_transforms, list):
            template_transforms = [template_transforms]
        all_transforms += template_transforms

    links = []
    for transform in all_transforms:
        entities = parse_file_entities(transform, entities=query)
        link = (entities['from'], entities['template'])
        links.append(link)

    path = None
    try:
        path = find_shortest_path(links, source, target)
        print('Shortest path:', path)
    except ValueError as e:
        print(e)

    selected_transforms = [all_transforms[i] for i in path]

    return selected_transforms
