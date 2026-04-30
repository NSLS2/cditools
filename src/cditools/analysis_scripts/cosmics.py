"""
Cosmics for a counting detector will trigger values of 1 or 2,
which are hard to see in default color settings, so we mask out
high and zero values so that cosmics stand out more.

Data is accessed like data[image][row][column],
rather than data[image][x][y]
"""

from __future__ import annotations

from typing import Any

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson
from skimage.measure import label


def setup(
    det_name: str, date_dir: str, file_name: str, proposal_id: str
) -> NDArray[np.floating[Any]]:
    """Sets up the data to use later for analysis.

    Args:
        det_name (str): name of directory for the detector, e.g. "merlines-1"
        date_dir (str): data in format "YYYY/MM/DD"
        file_name (str): file name without the .h5 extension, e.g. "scan_0000"
        proposal_id (str): current PASS proposal ID

    Returns:
        data: dataset to be analyzed, in the form of a 3D array with dimensions (num_images, rows, cols)
    """
    file_path = f"/nsls2/data/cdi/proposals/commissioning/pass-{proposal_id}/assets/{det_name}/{date_dir}/{file_name}.h5"
    with h5py.File(file_path, "r") as f:
        return np.asarray(f["entry"]["data"]["data"])


def movie(data: NDArray[np.floating[Any]], vmin: int = 1, vmax: int = 5) -> None:
    """Creates a movie of the images in the dataset, with a colormap that highlights values between vmin and vmax.

    Args:
        data (NDArray[np.floating[Any]]): data set with entries in the form of (num_images, rows, cols)
        vmin (int, optional): minimum value for cosmics. Defaults to 1.
        vmax (int, optional): maximum value for cosmics. Defaults to 5.
    """
    cmap = mpl.colormaps["gray"].with_extremes(under="aliceblue", over="r")
    norm = mpl.colors.Normalize(vmin, vmax)
    for n in range(data.shape[0]):
        plt.imshow(data[n], cmap=cmap, norm=norm)
        plt.title(f"Image {n}")
        plt.draw()
        plt.pause(0.1)


def label_cosmics(data: NDArray[np.floating[Any]]) -> tuple[list[int], float]:
    """Labels the cosmics in each image and counts the number of cosmics per image,
    then calculates the average number of cosmics across all images.

    Args:
        data (NDArray[np.floating[Any]]): data set with entries in the form of (num_images, rows, cols)

    Returns:
        tuple[list[int], float]: A tuple containing a list of counts of cosmics per image and the average number of cosmics across all images.
    """
    counts = []
    cosmic_sum = 0
    for n in range(data.shape[0]):
        image = data[n].copy()
        # find indices of top 3 pixels
        flat_idx = np.argpartition(image, -3, axis=None)[-3:]
        coords = np.unravel_index(flat_idx, image.shape)
        # set them to 0
        image[coords] = 0
        _, num = label(image, return_num=True)
        counts.append(num)
        cosmic_sum += num

    average = cosmic_sum / data.shape[0]
    return counts, average


def plot_counts(counts: NDArray[np.floating[Any]]):
    """Plot the histogram of the cosmic counts and fit a poisson distribution

    Args:
        counts (NDArray[np.floating[Any]]): array of counts of cosmics per image
    """
    counts = np.array(counts)
    k_vals = np.arange(counts.max() + 1)
    fit_vals = poisson.pmf(k_vals, mu=np.mean(counts))

    plt.hist(counts, bins=np.arange(counts.max() + 2) - 0.5, density=True, alpha=0.6)
    plt.plot(k_vals, fit_vals, "o-", label="Poisson fit")
    plt.xlabel("Counts per image")
    plt.ylabel("Probability")
    plt.title("Histogram of counts")
    plt.legend()
    plt.show()


def check_empty(data: NDArray[np.floating[Any]]) -> list[int]:
    """Checks for images that have values of all zero

    Args:
        data (NDArray[np.floating[Any]]): data set with entries in the form of (num_images, rows, cols)
    Returns:
        list[int]: list of indices of images that have all zero values
    """
    zero_images = []
    for n in range(data.shape[0]):
        if not np.any(data[n]):
            zero_images.append(n)
    return zero_images


def find_cosmics(
    data: NDArray[np.floating[Any]], vmin: int = 1, vmax: int = 5
) -> set[int]:
    """Masks images to find the cosmics in

    Args:
        data (NDArray[np.floating[Any]]): data set with entries in the form of (num_images, rows, cols)
        vmin (int, optional): minimum value for cosmics. Defaults to 1.
        vmax (int, optional): maximum value for cosmics. Defaults to 5.

    Returns:
        set[int]: set of points that are identified as cosmics, based on the values in the data set being between vmin and vmax
    """
    cosmic_mask = (data >= vmin) & (data <= vmax)
    return set(np.flatnonzero(np.any(cosmic_mask, axis=(1, 2))))


def show_image(
    data: NDArray[np.floating[Any]], image_index: int, vmin: int = 1, vmax: int = 5
):
    """Display the desired image by index

    Args:
        data (NDArray[np.floating[Any]]): data set with entries in the form of (num_images, rows, cols)
        image_index (int): index of the image to display
        vmin (int, optional): minimum value for cosmics. Defaults to 1.
        vmax (int, optional): maximum value for cosmics. Defaults to 5.
    """
    cmap = mpl.colormaps["gray"].with_extremes(under="aliceblue", over="r")
    norm = mpl.colors.Normalize(vmin, vmax)
    plt.imshow(data[image_index], cmap=cmap, norm=norm)
    plt.title(f"Image {image_index}")
    plt.colorbar()
    plt.show()
