from __future__ import annotations

import bluesky.plans as bps


def pencil_scan(slit, cam, start, stop, num_points):
    """
    Perform a pencil scan by moving the slit from start to stop positions
    while acquiring images from the camera.

    Parameters:
    slit : Device
        The slit device to be moved.
    cam : Device
        The camera device to scan.
    start : float
        The starting position of the slit.
    stop : float
        The stopping position of the slit.
    num_points : int
        The number of points to scan between start and stop.
    """
    yield from bps.scan(cam, slit, start, stop, num_points)
