Develop a plan to make the fMRIPost workflow feature-complete.

First, we need a subworkflow that generates postprocessing derivatives.
This subworkflow should take in preprocessed BOLD images in some arbitrary space/resolution (MNI152NLin6Asym:res-02 for this example workflow) and output two derivatives:
a mean image (3D) and a squared image (4D).
These simulate actual postprocessing steps that real fMRIPost workflows perform.

In the examples below, we have a workflow that requires data in MNI152NLin6Asym:res-02 for the postprocessing subworkflow.
BOLD data in that space/resolution is either (1) available in the fMRIPrep outputs,
(2) can be generated from boldref-space (also known as func space or native space) fMRIPrep outputs and transforms,
or (3) can be generated from raw BOLD data and transforms in the fMRIPrep outputs.

Additionally, the examples below include an output space that the fMRIPost workflow should be able to generate postprocessing derivatives in (MNIPediatricAsym:res-02).

Here are a set of different cases that should be supported by the fMRIPost BIDS App.

- fMRIPrep with --level full and the required output space for the fMRIPost workflow, but no boldref-space outputs.
    -   For example, with a workflow that requires data in MNI152NLin6Asym space,
        fMRIPost can directly use the fMRIPrep outputs in MNI152NLin6Asym:res-02 space.
        ```
        fmriprep /raw /fmriprep participant \
            --output-spaces MNI152NLin6Asym:res-02 MNIPediatricAsym:res-02 \
            --level full \
            --participant-label 01

        # fMRIPost can be called directly on the fMRIPrep outputs
        fmripost-template /fmriprep /fmripost participant \
            --output-spaces MNI152NLin6Asym:res-02 MNIPediatricAsym:res-02 \
            --participant-label 01
        ```
- fMRIPrep with --level full and boldref-space outputs, along with transforms to the required output space for the fMRIPost workflow.
    -   For example, with a workflow that requires data in MNI152NLin6Asym:res-02 space.
        Note that the output space for the fMRIPrep workflow is MNI152NLin6Asym:res-05, but the fMRIPost workflow requires data in MNI152NLin6Asym:res-02,
        so we need to warp the func-space outputs to MNI152NLin6Asym:res-02 instead of using the MNI152NLin6Asym:res-05 outputs directly.
        ```
        fmriprep /raw /fmriprep participant \
            --output-spaces func MNI152NLin6Asym:res-05 MNIPediatricAsym:res-02 \
            --level full \
            --participant-label 01

        # fMRIPost can be called directly on the fMRIPrep outputs
        fmripost-template /fmriprep /fmripost participant \
            --output-spaces MNI152NLin6Asym:res-02 MNIPediatricAsym:res-02 \
            --participant-label 01
        ```
- fMRIPrep with --level full and boldref-space outputs, along with transforms to spaces that can be combined with public transforms (from TemplateFlow) to required spaces.
    -   For example, with a workflow that requires data in MNI152NLin6Asym space, fMRIPost can use the MNI152NLin2009cAsym-to-MNI152NLin6Asym transform from TemplateFlow.
        It should chain together the func-to-T1w, T1w-to-MNI152NLin2009cAsym, and MNI152NLin2009cAsym-to-MNI152NLin6Asym transforms to generate the required data in MNI152NLin6Asym:res-02 space
        from the func-space outputs.
        ```
        fmriprep /raw /fmriprep participant \
            --output-spaces func MNI152NLin2009cAsym MNIPediatricAsym \
            --level full \
            --participant-label 01

        # fMRIPost can be called directly on the fMRIPrep outputs
        fmripost-template /fmriprep /fmripost participant \
            --output-spaces MNI152NLin6Asym:res-02 MNIPediatricAsym:res-02 \
            --participant-label 01
        ```
- fMRIPrep with --level minimal and raw BOLD data.
    -   For example, with a workflow that requires data in MNI152NLin6Asym:res-02 space.
        Since minimal-level fMRIPrep outputs do not include resampled BOLD data,
        fMRIPost will need to apply minimal preprocessing steps internally to generate the required derivatives,
        as well as warp the BOLD data to the required space/resolution.
        ```
        fmriprep /raw /fmriprep participant \
            --output-spaces func MNI152NLin6Asym MNIPediatricAsym \
            --level minimal \
            --participant-label 01

        # fMRIPost requires both the raw dataset and the fMRIPrep outputs
        fmripost-template /raw /fmripost participant \
            --output-spaces MNI152NLin6Asym:res-02 MNIPediatricAsym:res-02 \
            --participant-label 01 \
            --datasets fmriprep=/fmriprep
        ```

Let's link this to actual fMRIPost workflows.

1.  fMRIPost-AROMA (https://github.com/nipreps/fmripost-aroma): The key steps (SUSAN + MELODIC + ICA-AROMA) must be applied to preprocessed BOLD data in MNI152NLin6Asym:res-02 space.
    These steps produce confound time series that can be used to denoise preprocessed BOLD data in any space/resolution,
    so fMRIPost-AROMA needs MNI152NLin6Asym:res-02 space/resolution for the postprocessing subworkflow,
    but should be able to denoise native-space BOLD data and resample that to any requested spaces/resolutions for which the necessary transforms are available,
    or even denoise preprocessed BOLD data already available in those spaces/resolutions in the fMRIPrep outputs.
2.  fMRIPost-tedana (https://github.com/nipreps/fmripost-tedana): The key step (tedana) must be applied to preprocessed echo-wise BOLD data in native space.
    tedana produces optimally combined and denoised BOLD data in native space,
    so fMRIPost-tedana needs native-space BOLD data for the postprocessing subworkflow,
    and should be able to resample that to any requested spaces/resolutions for which the necessary transforms are available.
    Echo-wise BOLD data are only created for multi-echo fMRI data when `--me-output-echos` is used in fMRIPrep.
    Tedana also produces confound time series that can be used to denoise preprocessed BOLD data in any space/resolution,
    so fMRIPost-tedana's denoising can be applied to preprocessed BOLD data already available in fMRIPrep outputs
    (rather than having to denoise native-space BOLD data and resample that to the requested spaces/resolutions).
3.  fMRIPost-phase (https://github.com/nipreps/fmripost-phase): This workflow uses phase data from the raw dataset, then applies transforms from the fMRIPrep outputs to generate the required derivatives.
    The actual scope of this workflow is not fully defined yet, so we need to figure out what spaces/resolutions are required for different steps.
    For example, phase unwrapping happens in native space, but phase regression may need to happen in MNI152NLin6Asym:res-02 space.
    fMRIPrep does not preprocess phase data, so fMRIPost-phase needs access to the raw dataset.
    Luckily, the phase data in the raw dataset is in the same space as the raw BOLD (magnitude) data,
    so the transforms from the fMRIPrep outputs can be applied to the phase data in the raw dataset.
4.  fMRIPost-rapidtide (https://github.com/nipreps/fmripost-rapidtide): The key step (rapidtide) may be applied to preprocessed BOLD data in any space/resolution,
    but for simplicity's sake we will apply it to native-space BOLD data.
    This may require warping other files (e.g., tissue maps) to native BOLD space before running rapidtide.
    Rapidtide will denoise the native-space BOLD data and output additional derivatives (e.g., lag maps, correlation maps, etc.),
    which fMRIPost-rapidtide will need to warp to any requested spaces/resolutions for which the necessary transforms are available.
5.  XCP-D (https://github.com/PennLINC/xcp_d): While not currently an fMRIPost workflow,
    once the fMRIPost structure is complete, XCP-D can be restructured into an fMRIPost workflow.
    XCP-D currently only postprocesses BOLD data in one of a few volumetric spaces
    (MNI152NLin6Asym, MNI152NLin2009cAsym, MNIInfant) or one CIFTI space (fsLR:den91k).
    We ultimately want it to be able to postprocess BOLD data in any space/resolution and accept an `--output-spaces` argument like fMRIPost does.

Here is a *tentative* plan for the template:

1. Determine the requested spaces/resolutions from the command line arguments.
2. If a specific space/resolution is required for the postprocessing subworkflow,
   add it to the list of spaces/resolutions to be generated.
   fMRIPrep has something similar to this where the transform to MNI152NLin2009cAsym is automatically added to the list of spaces/resolutions to be generated,
   so fMRIPost can model itself on that behavior.
3. Collect the necessary data from the fMRIPrep outputs, raw dataset (if provided and needed), and any other BIDS datasets provided with the `--datasets` argument.
4. If a specific space is requested that is not available, and there are no transforms available to generate that space/resolution from the available data,
   raise an error.
5. If using raw data, apply minimal preprocessing steps
   (slice timing correction, motion correction, susceptibility distortion correction)
   to the raw data and warp to the required spaces/resolutions to generate the required data for the postprocessing subworkflow.
6. If using native-space (but preprocessed) BOLD data, warp the data to the required spaces/resolutions to generate the required data for the postprocessing subworkflow.
7. Apply the workflow-specific postprocessing steps to the data in the required spaces/resolutions.
8. Warp the postprocessing derivatives to the requested spaces/resolutions.
   This may include projection to surfaces or combined surface-volume (CIFTI) formats.

The above plan is necessarily incomplete and subject to change as we develop the fMRIPost BIDS App structure.
Other steps that may need to be added include:
- Calculating QC measures before and after postprocessing.
- Generating figures for the visual reports.
- Extracting confound time series.
- Wrangling files and transforms from multiple BIDS datasets provided with the `--datasets` argument.
- Concatenating denoised BOLD data across runs (XCP-D only right now).
- Parcellating denoised BOLD data (XCP-D only right now).
- Correlating parcellated denoised BOLD data (XCP-D only right now).
- Aggregating derivatives across runs, then using the aggregated derivatives to generate more run-wise outputs
  (planned for fMRIPost-rapidtide, in which lag maps will be generated for each run, then averaged across runs,
  then used to denoise the run-wise data).

What the fMRIPost BIDS Apps should *not* do:
- Calculate transforms/registrations. This is the domain of fMRIPrep.
- Apply computationally-intensive segmentation steps like FreeSurfer's recon-all. This is the domain of fMRIPrep.
- Generate field maps. This is the domain of fMRIPrep.
- Generate basic confounds like motion parameters. This is the domain of fMRIPrep.
