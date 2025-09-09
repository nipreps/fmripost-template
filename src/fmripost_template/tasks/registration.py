import nibabel as nb
import numpy as np
from fileformats.generic import File
from fileformats.medimage import Nifti1, NiftiGz
from fmriprep.utils.transforms import load_transforms
from sdcflows.utils.tools import ensure_positive_cosines

from ..utils.resampling import resample_image
from .utils import fname_presuffix


def resample_image_pydra(
    source_path: Nifti1 | NiftiGz,
    target_path: Nifti1 | NiftiGz,
    transforms_path: list[File],
    fieldmap_path: Nifti1 | NiftiGz | None = None,
    pe_info: list[tuple[int, float]] | None = None,
    pe_dir: str | None = None,
    ro_time: float | None = None,
    jacobian: bool = True,
    nthreads: int = 1,
    transforms_inverse: list[bool] = None,
    output_dtype: np.dtype | str | None = 'f4',
    order: int = 3,
    mode: str = 'constant',
    cval: float = 0.0,
    prefilter: bool = True,
) -> Nifti1 | NiftiGz:
    """Pydra node for registering a bold series to a defined target space.

    Parameters
    ----------
    source
        The 3D bold image or 4D bold series to resample.
    target
        An image sampled in the target space.
    transforms
        A nitransforms TransformChain that maps images from the individual
        BOLD volume space into the target space.
    fieldmap
        The fieldmap, in Hz, sampled in the target space
    pe_info
        A list of readout vectors in the form of (axis, signed-readout-time)
        ``(1, -0.04)`` becomes ``[0, -0.04, 0]``, which indicates that a
        +1 Hz deflection in the field shifts 0.04 voxels toward the start
        of the data array in the second dimension.
    nthreads
        Number of threads to use for parallel resampling
    output_dtype
        The dtype of the output array.
    order
        Order of interpolation (default: 3 = cubic)
    mode
        How ``data`` is extended beyond its boundaries. See
        :func:`scipy.ndimage.map_coordinates` for more details.
    cval
        Value to fill past edges of ``data`` if ``mode`` is ``'constant'``.
    prefilter
        Determines if ``data`` is pre-filtered before interpolation.

    Returns
    -------
    resampled_bold
        The BOLD series resampled into the target space
    """
    out_path = fname_presuffix(source_path, suffix='ref', newpath='')

    source = nb.load(source_path)
    target = nb.load(target_path)

    fieldmap = nb.load(fieldmap_path) if fieldmap_path else None

    nvols = source.shape[3] if source.ndim > 3 else 1

    transforms = load_transforms(transforms_path, transforms_inverse)
    pe_info = None

    if pe_dir and ro_time:
        pe_axis = 'ijk'.index(pe_dir[0])
        pe_flip = pe_dir.endswith('-')

        # Nitransforms displacements are positive
        source, axcodes = ensure_positive_cosines(source)
        axis_flip = axcodes[pe_axis] in 'LPI'

        pe_info = [(pe_axis, -ro_time if (axis_flip ^ pe_flip) else ro_time)] * nvols

    output = resample_image(
        source=source,
        target=target,
        transforms=transforms,
        fieldmap=fieldmap,
        pe_info=pe_info,
        jacobian=jacobian,
        nthreads=nthreads,
        output_dtype=output_dtype,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )
    output.to_filename(out_path)
    return out_path
