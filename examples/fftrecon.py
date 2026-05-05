"""Standalone multi-slice 2D FFT reconstruction handler.

This module is a self-contained example of an ``mrdserver`` reconstruction
handler.  It implements the mandatory ``process(connection, config, metadata)``
entry point and can be used as a template for custom recon scripts.

The handler accumulates all readouts until ``ACQ_LAST_IN_MEASUREMENT``,
stacks them into a ``[cha, RO, PE, SLC]`` array, applies a 2-D IFFT and
root-sum-of-squares coil combination, then returns one ``ismrmrd.Image``
per slice.

Usage
-----
Pass the path to this file to ``mrdserver`` via the ``--handler`` option::

    mrdserver --handler examples/fftrecon.py

Or drop it in the ``recon/`` directory of the pulserver data tree::

    cp fftrecon.py /export/home/sdc/pulserver/recon/fftrecon.py
"""

import ctypes
import logging
from collections.abc import Generator, Iterator
from typing import Any

import ismrmrd
import numpy as np
import numpy.fft as fft

# This example uses mrd2dicom to convert ISMRMRD images to DICOM for transmission
# back to the scanner. If mrd2dicom is not available, remove the import and the
# conversion step to send plain ISMRMRD images instead.
try:
    from mrdserver.mrd2dicom import MrdDicomBuilder

    HAS_MRD2DICOM = True
except ImportError:
    HAS_MRD2DICOM = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Run multi-slice 2D FFT reconstruction and send images back.

    Parameters
    ----------
    connection : mrdserver.Connection
        Active MRD connection.  Iterating over it yields
        ``ismrmrd.Acquisition`` objects; call ``connection.send(image)`` to
        return results.  The connection is closed automatically when this
        function returns.
    config : str or dict
        Configuration string / dict forwarded from the CONFIG MRD message.
    metadata : ismrmrd.xsd.ismrmrdHeader
        Parsed ISMRMRD XML header for the current scan.

    Notes
    -----
    If ``mrdserver.mrd2dicom`` is available, images are converted to DICOM
    before transmission (for on-scanner use).  Otherwise, plain ISMRMRD
    images are sent.
    """
    logging.info("fftrecon — config: %s", config)

    dicom_gen = MrdDicomBuilder(metadata) if HAS_MRD2DICOM else None

    for group in _conditional_groups(
        connection,
        accept=lambda acq: not acq.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT),
        finish=lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT),
    ):
        for slice_image in _reconstruct(group, metadata):
            mrd_image = _array_to_image(slice_image, group, metadata)
            if dicom_gen:
                # Convert to DICOM for the scanner
                named_dset = dicom_gen(mrd_image)
                connection.send(named_dset)
            else:
                # Send plain ISMRMRD image
                connection.send(mrd_image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conditional_groups(
    iterable: Iterator[Any],
    accept: Any,
    finish: Any,
) -> Generator[list[ismrmrd.Acquisition], None, None]:
    """Yield lists of acquisitions split on a finish predicate.

    Parameters
    ----------
    iterable : Iterator
        Source of ``ismrmrd.Acquisition`` objects.
    accept : callable
        Return ``True`` for acquisitions that belong in the current group.
    finish : callable
        Return ``True`` for the acquisition that closes the current group.

    Yields
    ------
    list[ismrmrd.Acquisition]
        One list per completed group.
    """
    group: list[ismrmrd.Acquisition] = []
    for item in iterable:
        if item is None:
            break
        if accept(item):
            group.append(item)
        if finish(item):
            yield group
            group = []


def _reconstruct(
    group: list[ismrmrd.Acquisition],
    metadata: Any,
) -> list[np.ndarray]:
    """Reconstruct per-slice images from a list of k-space readouts.

    Parameters
    ----------
    group : list[ismrmrd.Acquisition]
        All readout lines for the measurement.
    metadata : ismrmrd.xsd.ismrmrdHeader
        Parsed ISMRMRD XML header.

    Returns
    -------
    list[numpy.ndarray]
        One ``int16`` array of shape ``(RO, PE)`` per slice,
        ordered by ascending slice index.
    """
    if not group:
        return []

    logging.info("Reconstructing %d readouts", len(group))

    # Stack into [cha, RO, PE*SLC]
    data = np.stack([acq.data for acq in group], axis=-1)

    # Reshape to [cha, RO, PE, SLC] and sort by slice index
    slice_indices = [acq.idx.slice for acq in group]
    n_slices = max(slice_indices) + 1
    data = data.reshape(data.shape[0], data.shape[1], -1, n_slices)
    sort_order = slice_indices[:n_slices]
    data = data[..., np.argsort(sort_order)]

    # 2-D IFFT (apply along RO and PE axes)
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Root-sum-of-squares coil combination  ->  [RO, PE, SLC]
    data = np.sqrt(np.sum(np.abs(data) ** 2, axis=0))

    # Scale to bit depth and convert to int16
    bits_stored = _get_bits_stored(metadata) or 12
    max_val = 2**bits_stored - 1
    data = np.around(data * (max_val / data.max())).astype(np.int16)

    # Return one [RO, PE] array per slice
    return [data[..., s] for s in range(n_slices)]


def _array_to_image(
    data: np.ndarray,
    group: list[ismrmrd.Acquisition],
    metadata: Any,
) -> ismrmrd.Image:
    """Wrap a 2-D pixel array in an ``ismrmrd.Image``.

    Parameters
    ----------
    data : numpy.ndarray
        Pixel data with shape ``(RO, PE)`` and dtype ``int16``.
    group : list[ismrmrd.Acquisition]
        Source readouts (first element supplies orientation metadata).
    metadata : ismrmrd.xsd.ismrmrdHeader
        Parsed ISMRMRD XML header.

    Returns
    -------
    ismrmrd.Image
        ISMRMRD image ready for transmission.
    """
    enc = metadata.encoding[0]
    # from_array with transpose=False expects [..., y, x]; transpose [RO, PE] -> [PE, RO]
    image = ismrmrd.Image.from_array(data.T, acquisition=group[0], transpose=False)
    image.image_index = 1
    image.field_of_view = (
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.x),
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.y),
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.z),
    )

    head = image.getHead()
    meta = ismrmrd.Meta(
        {
            "DataRole": "Image",
            "ImageProcessingHistory": ["FIRE", "PYTHON"],
            "ImageRowDir": [f"{head.read_dir[i]:.18f}" for i in range(3)],
            "ImageColumnDir": [f"{head.phase_dir[i]:.18f}" for i in range(3)],
        }
    )
    image.attribute_string = meta.serialize()
    return image


def _get_bits_stored(metadata: Any) -> int | None:
    """Return the BitsStored user parameter from the ISMRMRD header, or None.

    Parameters
    ----------
    metadata : ismrmrd.xsd.ismrmrdHeader
        Parsed ISMRMRD XML header.

    Returns
    -------
    int or None
        Value of the ``BitsStored`` user parameter, or ``None`` if absent.
    """
    try:
        for param in metadata.userParameters.userParameterLong:
            if param.name == "BitsStored":
                return int(param.value)
    except AttributeError:
        pass
    return None
