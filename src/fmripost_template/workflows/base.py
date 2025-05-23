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
"""
fMRIPost-template workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmripost_template_wf
.. autofunction:: init_single_subject_wf
.. autofunction:: init_single_run_wf

"""

import os
import sys
from collections import defaultdict
from copy import deepcopy

import yaml
from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_template import config
from fmripost_template.utils.utils import _get_wf_name, update_dict


def init_fmripost_template_wf():
    """Build *fMRIPost-template*'s pipeline.

    This workflow organizes the execution of fMRIPost-template,
    with a sub-workflow for each subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_template.workflows.tests import mock_config
            from fmripost_template.workflows.base import init_fmripost_template_wf

            with mock_config():
                wf = init_fmripost_template_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)

    fmripost_template_wf = Workflow(name=f'fmripost_template_{ver.major}_{ver.minor}_wf')
    fmripost_template_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmripost_template_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmripost_template.toml')

    return fmripost_template_wf


def init_single_subject_wf(subject_id: str):
    """Organize the postprocessing pipeline for a single subject.

    It collects and reports information about the subject,
    and prepares sub-workflows to postprocess each BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_template.workflows.tests import mock_config
            from fmripost_template.workflows.base import init_single_subject_wf

            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.
    """
    from bids.utils import listify
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo

    from fmripost_template.interfaces.bids import DerivativesDataSink
    from fmripost_template.interfaces.reportlets import AboutSummary, SubjectSummary
    from fmripost_template.utils.bids import collect_derivatives

    spaces = config.workflow.spaces

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from postprocessing
performed using *fMRIPost-template* {config.environment.version},
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = """

For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPost-template*'s documentation]\
(https://fmripost_template.readthedocs.io/en/latest/workflows.html).


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPost-template
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the
[CC0](https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

"""
    entities = config.execution.bids_filters or {}
    entities['subject'] = subject_id

    if config.execution.derivatives:
        # Raw dataset + derivatives dataset
        config.loggers.workflow.info('Raw+derivatives workflow mode enabled')
        # Just build a list of BOLD files right now
        subject_data = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=entities,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        subject_data['bold'] = listify(subject_data['bold_raw'])
    else:
        # Derivatives dataset only
        config.loggers.workflow.info('Derivatives-only workflow mode enabled')
        # Just build a list of BOLD files right now
        subject_data = collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=config.execution.layout,
            entities=entities,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        # Patch standard-space BOLD files into 'bold' key
        subject_data['bold'] = listify(subject_data['bold_mni152nlin6asym'])

    # Make sure we always go through these two checks
    if not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f'No BOLD images found for participant {subject_id} and '
            f'task {task_id if task_id else "<all>"}. '
            'All workflows require BOLD images. '
            f'Please check your BIDS filters: {config.execution.bids_filters}.'
        )

    config.loggers.workflow.info(
        f'Collected subject data:\n{yaml.dump(subject_data, default_flow_style=False, indent=4)}',
    )

    bids_info = pe.Node(
        BIDSInfo(
            bids_dir=config.execution.bids_dir,
            bids_validate=False,
            in_file=subject_data['bold'][0],
        ),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(
            bold=subject_data['bold'],
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name='summary',
        run_without_submitting=True,
    )
    workflow.connect([(bids_info, summary, [('subject', 'subject_id')])])

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )
    workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')])])

    ds_report_about = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='about',
            datatype='figures',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )
    workflow.connect([(about, ds_report_about, [('out_report', 'in_file')])])

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data postprocessing

: For each of the {len(subject_data['bold'])} BOLD runs found per subject
(across all tasks and sessions), the following postprocessing was performed.
"""
    workflow.__desc__ += func_pre_desc

    for bold_file in subject_data['bold']:
        single_run_wf = init_single_run_wf(bold_file)
        workflow.add_nodes([single_run_wf])

    return clean_datasinks(workflow)


def init_single_run_wf(bold_file):
    """Set up a single-run workflow for fMRIPost-template."""
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from fmripost_template.utils.bids import collect_derivatives, extract_entities

    spaces = config.workflow.spaces

    workflow = Workflow(name=_get_wf_name(bold_file, 'single_run'))
    workflow.__desc__ = ''

    entities = extract_entities(bold_file)

    # Attempt to extract the associated fmap ID
    fmapid = None
    all_fmapids = config.execution.layout.get_fmapids(
        subject=entities['subject'],
        session=entities.get('session', None),
    )
    if all_fmapids:
        fmap_file = config.execution.layout.get_nearest(
            bold_file,
            to=all_fmapids,
            suffix='xfm',
            extension='.txt',
            strict=False,
            **{'from': 'boldref'},
        )
        fmapid = config.execution.layout.get_file(fmap_file).entities['to']

    functional_cache = defaultdict(list, {})
    if config.execution.derivatives:
        # Collect native-space derivatives and transforms
        functional_cache = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=entities,
            fieldmap_id=fmapid,
            allow_multiple=False,
            spaces=None,
        )
        for deriv_dir in config.execution.derivatives.values():
            functional_cache = update_dict(
                functional_cache,
                collect_derivatives(
                    raw_dataset=None,
                    derivatives_dataset=deriv_dir,
                    entities=entities,
                    fieldmap_id=fmapid,
                    allow_multiple=False,
                    spaces=spaces,
                ),
            )

        if not functional_cache['bold_confounds']:
            if config.workflow.dummy_scans is None:
                raise ValueError(
                    'No confounds detected. '
                    'Automatical dummy scan detection cannot be performed. '
                    'Please set the `--dummy-scans` flag explicitly.'
                )

            # TODO: Calculate motion parameters from motion correction transform
            raise ValueError('No confounds detected.')

    else:
        # Collect MNI152NLin6Asym:res-2 derivatives
        # Only derivatives dataset was passed in, so we expected standard-space derivatives
        functional_cache.update(
            collect_derivatives(
                raw_dataset=None,
                derivatives_dataset=config.execution.layout,
                entities=entities,
                fieldmap_id=fmapid,
                allow_multiple=False,
                spaces=spaces,
            ),
        )

    config.loggers.workflow.info(
        (
            f'Collected run data for {os.path.basename(bold_file)}:\n'
            f'{yaml.dump(functional_cache, default_flow_style=False, indent=4)}'
        ),
    )

    if config.workflow.dummy_scans is not None:
        skip_vols = config.workflow.dummy_scans
    else:
        if not functional_cache['bold_confounds']:
            raise ValueError(
                'No confounds detected. '
                'Automatical dummy scan detection cannot be performed. '
                'Please set the `--dummy-scans` flag explicitly.'
            )
        skip_vols = get_nss(functional_cache['bold_confounds'])

    print(skip_vols)  # just to circumvent flake8 warning

    # Fill in datasinks seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).inputs.base_directory = config.execution.output_dir
            workflow.get_node(node).inputs.source_file = bold_file

    return workflow


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def clean_datasinks(workflow: pe.Workflow) -> pe.Workflow:
    """Overwrite ``out_path_base`` of DataSinks."""
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = ''
    return workflow


def get_nss(confounds_file):
    """Get number of non-steady state volumes.

    Parameters
    ----------
    confounds_file : :obj:`str`
        Path to the confounds file.

    Returns
    -------
    :obj:`int`
        Number of non-steady state volumes.

    Notes
    -----
    This function assumes that all non-steady state volumes are contiguous,
    and that the non-steady state outlier columns are named `non_steady_state_outlier*`.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_table(confounds_file)

    nss_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]

    dummy_scans = 0
    if nss_cols:
        initial_volumes_df = df[nss_cols]
        dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
        dummy_scans = np.where(dummy_scans)[0]

        # reasonably assumes all NSS volumes are contiguous
        dummy_scans = int(dummy_scans[-1] + 1)

    return dummy_scans
