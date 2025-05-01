# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
import os
import typing as ty

import bids
import nibabel as nb
from fmriprep.interfaces.reports import FunctionalSummary
from fmriprep.utils.bids import extract_entities
from fmriprep.workflows.bold.reference import init_validation_and_dummies_wf
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.func.util import init_skullstrip_bold_wf

from fmripost_template import config


def get_sbrefs(
    bold_files: list[str],
    entity_overrides: dict[str, ty.Any],
    layout: bids.BIDSLayout,
) -> list[str]:
    """Find single-band reference(s) associated with BOLD file(s)

    Parameters
    ----------
    bold_files
        List of absolute paths to BOLD files
    entity_overrides
        Query parameters to override defaults
    layout
        :class:`~bids.layout.BIDSLayout` to query

    Returns
    -------
    sbref_files
        List of absolute paths to sbref files associated with input BOLD files,
        sorted by EchoTime
    """
    entities = extract_entities(bold_files)
    entities.pop('echo', None)
    entities.update(suffix='sbref', extension=['.nii', '.nii.gz'])
    entities.update(entity_overrides)

    return sorted(
        layout.get(return_type='file', **entities),
        key=lambda fname: layout.get_metadata(fname).get('EchoTime'),
    )


def init_bold_fit_wf(
    *,
    bold_series: list[str],
    precomputed: dict = None,
    fieldmap_id: str | None = None,
    jacobian: bool = False,
    omp_nthreads: int = 1,
    name: str = 'bold_fit_wf',
) -> pe.Workflow:
    """
    This workflow controls the minimal estimation steps for functional preprocessing.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.tests import mock_config
            from fmriprep import config
            from fmriprep.workflows.bold.fit import init_bold_fit_wf
            with mock_config():
                bold_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_bold.nii.gz"
                wf = init_bold_fit_wf(bold_series=[str(bold_file)])

    Parameters
    ----------
    bold_series
        List of paths to NIfTI files, sorted by echo time.
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.
    fieldmap_id
        ID of the fieldmap to use to correct this BOLD series. If :obj:`None`,
        no correction will be applied.

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    anat2std_xfm
        List of transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
    fmap_id
        Unique identifiers to select fieldmap files
    fmap
        List of estimated fieldmaps (collated with fmap_id)
    fmap_ref
        List of fieldmap reference files (collated with fmap_id)
    fmap_coeff
        List of lists of spline coefficient files (collated with fmap_id)
    fmap_mask
        List of fieldmap masks (collated with fmap_id)
    sdc_method
        List of fieldmap correction method names (collated with fmap_id)

    Outputs
    -------
    hmc_boldref
        BOLD reference image used for head motion correction.
        Minimally processed to ensure consistent contrast with BOLD series.
    coreg_boldref
        BOLD reference image used for coregistration. Contrast-enhanced
        and fieldmap-corrected for greater anatomical fidelity, and aligned
        with ``hmc_boldref``.
    bold_mask
        Mask of ``coreg_boldref``.
    motion_xfm
        Affine transforms from each BOLD volume to ``hmc_boldref``, written
        as concatenated ITK affine transforms.
    boldref2anat_xfm
        Affine transform mapping from BOLD reference space to the anatomical
        space.
    boldref2fmap_xfm
        Affine transform mapping from BOLD reference space to the fieldmap
        space, if applicable.
    dummy_scans
        The number of dummy scans declared or detected at the beginning of the series.

    See Also
    --------
    * :py:func:`~fmriprep.workflows.bold.fit.init_bold_fit_wf`
    """
    from fmriprep.utils.misc import estimate_bold_mem_usage
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    if precomputed is None:
        precomputed = {}
    layout = config.execution.layout
    bids_filters = config.execution.get().get('bids_filters', {})

    # Fitting operates on the shortest echo
    # This could become more complicated in the future
    bold_file = bold_series[0]

    # Collect sbref files, sorted by EchoTime
    sbref_files = get_sbrefs(
        bold_series,
        entity_overrides=bids_filters.get('sbref', {}),
        layout=layout,
    )

    basename = os.path.basename(bold_file)
    sbref_msg = f'No single-band-reference found for {basename}.'
    if sbref_files and 'sbref' in config.workflow.ignore:
        sbref_msg = f'Single-band reference file(s) found for {basename} and ignored.'
        sbref_files = []
    elif sbref_files:
        sbref_msg = 'Using single-band reference file(s) {}.'.format(
            ','.join([os.path.basename(sbf) for sbf in sbref_files])
        )
    config.loggers.workflow.info(sbref_msg)

    # Get metadata from BOLD file(s)
    entities = extract_entities(bold_series)
    metadata = layout.get_metadata(bold_file)
    orientation = ''.join(nb.aff2axcodes(nb.load(bold_file).affine))

    bold_tlen, mem_gb = estimate_bold_mem_usage(bold_file)

    bold_preprocessed = precomputed.get('bold_preprocessed')
    hmc_boldref = precomputed.get('hmc_boldref')
    coreg_boldref = precomputed.get('coreg_boldref')
    # Can contain
    #  1) boldref2fmap
    #  2) boldref2anat
    #  3) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    boldref2fmap_xform = transforms.get('boldref2fmap')
    boldref2anat_xform = transforms.get('boldref2anat')

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold_file',
                # Fieldmap registration
                'fmap',
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
                'fmap_id',
                'sdc_method',
                # Anatomical coregistration
                't1w_preproc',
                't1w_mask',
                't1w_dseg',
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.bold_file = bold_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dummy_scans',
                'hmc_boldref',
                'coreg_boldref',
                'bold_mask',
                'motion_xfm',
                'boldref2anat_xfm',
                'boldref2fmap_xfm',
            ],
        ),
        name='outputnode',
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    target = 'raw'  # or 'preprocessed', if raw + transforms aren't available

    hmcref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'bold_file', 'dummy_scans']),
        name='hmcref_buffer',
    )
    fmapref_buffer = pe.Node(niu.Function(function=_select_ref), name='fmapref_buffer')
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=['hmc_xforms']), name='hmc_buffer')
    fmapreg_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref2fmap_xfm']), name='fmapreg_buffer'
    )
    regref_buffer = pe.Node(
        niu.IdentityInterface(fields=['boldref', 'boldmask']), name='regref_buffer'
    )

    if hmc_boldref:
        hmcref_buffer.inputs.boldref = hmc_boldref
        config.loggers.workflow.debug('Reusing motion correction reference: %s', hmc_boldref)
    if hmc_xforms:
        hmc_buffer.inputs.hmc_xforms = hmc_xforms
        config.loggers.workflow.debug('Reusing motion correction transforms: %s', hmc_xforms)
    if boldref2fmap_xform:
        fmapreg_buffer.inputs.boldref2fmap_xfm = boldref2fmap_xform
        config.loggers.workflow.debug('Reusing BOLD-to-fieldmap transform: %s', boldref2fmap_xform)
    if coreg_boldref:
        regref_buffer.inputs.boldref = coreg_boldref
        config.loggers.workflow.debug('Reusing coregistration reference: %s', coreg_boldref)
    fmapref_buffer.inputs.sbref_files = sbref_files

    summary = pe.Node(
        FunctionalSummary(
            distortion_correction='None',  # Can override with connection
            registration=(
                'Precomputed'
                if boldref2anat_xform
                else 'FreeSurfer'
                if config.workflow.run_reconall
                else 'FSL'
            ),
            registration_dof=config.workflow.bold2anat_dof,
            registration_init=config.workflow.bold2anat_init,
            pe_direction=metadata.get('PhaseEncodingDirection'),
            echo_idx=entities.get('echo', []),
            tr=metadata['RepetitionTime'],
            orientation=orientation,
        ),
        name='summary',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    summary.inputs.dummy_scans = config.workflow.dummy_scans
    if config.workflow.level == 'full':
        # Hack. More pain than it's worth to connect this up at a higher level.
        # We can consider separating out fit and transform summaries,
        # or connect a bunch a bunch of summary parameters to outputnodes
        # to make available to the base workflow.
        summary.inputs.slice_timing = (
            bool(metadata.get('SliceTiming')) and 'slicetiming' not in config.workflow.ignore
        )

    workflow.connect([
        (hmcref_buffer, outputnode, [
            ('boldref', 'hmc_boldref'),
            ('dummy_scans', 'dummy_scans'),
        ]),
        (regref_buffer, outputnode, [
            ('boldref', 'coreg_boldref'),
            ('boldmask', 'bold_mask'),
        ]),
        (fmapreg_buffer, outputnode, [('boldref2fmap_xfm', 'boldref2fmap_xfm')]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
        ]),
    ])  # fmt:skip

    # Stage 1: Generate motion correction boldref
    hmc_boldref_source_buffer = pe.Node(
        niu.IdentityInterface(fields=['in_file']),
        name='hmc_boldref_source_buffer',
    )
    if not hmc_boldref and target == 'raw':
        config.loggers.workflow.info('No HMC boldref found - requiring preprocessed data')
        target = 'preprocessed'
    else:
        config.loggers.workflow.info('Found HMC boldref - skipping Stage 1')

        validation_and_dummies_wf = init_validation_and_dummies_wf(bold_file=bold_file)

        workflow.connect([
            (validation_and_dummies_wf, hmcref_buffer, [
                ('outputnode.bold_file', 'bold_file'),
                ('outputnode.skip_vols', 'dummy_scans'),
            ]),
            (hmcref_buffer, hmc_boldref_source_buffer, [('boldref', 'in_file')]),
        ])  # fmt:skip

    # Stage 2: Estimate head motion
    if not hmc_xforms and target == 'raw':
        config.loggers.workflow.info('No HMC transforms found - requiring preprocessed data')
        target = 'preprocessed'
    else:
        config.loggers.workflow.info('Found motion correction transforms - skipping Stage 2')

    # Stage 3: Create coregistration reference
    # Fieldmap correction only happens during fit if this stage is needed
    if not coreg_boldref and target == 'raw':
        config.loggers.workflow.info('No coregistration boldref found - requiring preprocessed data')
        target = 'preprocessed'
    else:
        config.loggers.workflow.info('Found coregistration reference - skipping Stage 3')

        # TODO: Allow precomputed bold masks to be passed
        # Also needs consideration for how it interacts above
        skullstrip_precomp_ref_wf = init_skullstrip_bold_wf(name='skullstrip_precomp_ref_wf')
        skullstrip_precomp_ref_wf.inputs.inputnode.in_file = coreg_boldref
        workflow.connect([
            (skullstrip_precomp_ref_wf, regref_buffer, [('outputnode.mask_file', 'boldmask')])
        ])  # fmt:skip

    if not boldref2anat_xform and target == 'raw':
        config.loggers.workflow.info('No coregistration boldref found - requiring preprocessed data')
        target = 'preprocessed'
    else:
        config.loggers.workflow.info('Found coregistration boldref - skipping Stage 4')
        outputnode.inputs.boldref2anat_xfm = boldref2anat_xform

    if target == 'preprocessed' and not bold_preprocessed:
        raise RuntimeError('Preprocessed data not found')

    return workflow


def _select_ref(sbref_files, boldref_files):
    """Select first sbref or boldref file, preferring sbref if available"""
    from niworkflows.utils.connections import listify

    refs = sbref_files or boldref_files
    return listify(refs)[0]
