"""
Cosmics for a counting detector will trigger values of 1 or 2,
which are hard to see in default color settings, so we mask out
high and zero values so that cosmics stand out more.

Data is accessed like data[image][row][column],
rather than data[image][x][y]
"""
from typing import Any

import h5py
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.measure import label
from scipy.stats import poisson

def setup(det_name:str, date_dir:str, file_name:str, proposal_id:str) -> NDArray[np.floating[Any]]:
    """ Setups the data to use later for analysis.

    Args:
        det_name (str): name of directory for the detector, e.g. "merlines-1"
        date_dir (str): data in format "YYYY/MM/`DD"
        file_name (str): file name without the .h5 extension, e.g. "scan_0000"
        proposal_id (str): current PASS proposal ID

    Returns:
        data: dataset to be analyzed, in the form of a 3D array with dimensions (num_images, rows, cols)
    """
    file_path = f'/nsls2/data/cdi/proposals/commissioning/pass-{proposal_id}/assets/{det_name}/{date_dir}/{file_name}.h5'
    with h5py.File(file_path, 'r') as f:
        return np.asarray(f['entry']['data']['data'])

def movie(data: NDArray[np.floating[Any]], vmin: int = 1, vmax: int = 5) -> None:
    # Set colormapping and mask
    cmap = mpl.colormaps['gray'].with_extremes(under='aliceblue', over='r')
    norm = mpl.colors.Normalize(vmin, vmax) 
    for n in range(data.shape[0]):
        plt.imshow(data[n], cmap=cmap, norm=norm)    
        plt.title(f"Image {n}")
        plt.draw()
        plt.pause(0.1)

def label_cosmics(data: NDArray[np.floating[Any]]) -> tuple[list[int], float]:
    counts = []
    cosmic_sum = 0
    for n in range(data.shape[0]):
        image = data[n].copy()
        #find indices of top 3 pixels
        flat_idx = np.argpartition(image, -3, axis=None)[-3:]
        coords = np.unravel_index(flat_idx, image.shape)
        # set them to 0
        image[coords] = 0
        _, num = label(image, return_num = True) 
        counts.append(num)
        cosmic_sum+=num

    average = cosmic_sum/data.shape[0]
    return counts, average

def plot_counts(counts: NDArray[np.floating[Any]]):
    counts = np.array(counts)
    k_vals = np.arange(counts.max() + 1)
    fit_vals = poisson.pmf(k_vals, mu=np.mean(counts))

    plt.hist(counts, bins=np.arange(counts.max()+2)-0.5, density=True, alpha=0.6)
    plt.plot(k_vals, fit_vals, 'o-', label="Poisson fit")
    plt.xlabel("Counts per image")
    plt.ylabel("Probability")
    plt.title("Histogram of counts")
    plt.legend()
    plt.show()

def check_empty(data: NDArray[np.floating[Any]]): 
    for n in range(data.shape[0]):
        if not np.any(data[n]):
            print(f"Image {n} is empty.")

def find_cosmics(data: NDArray[np.floating[Any]], vmin: int = 1, vmax: int = 5):
    cosmic_mask = (data >= vmin) & (data <= vmax)
    cosmic_points = set(np.flatnonzero(np.any(cosmic_mask, axis=(1, 2))))
    return cosmic_points

def show_image(data: NDArray[np.floating[Any]], image_index:int, vmin: int = 1, vmax: int = 5):
    cmap = mpl.colormaps['gray'].with_extremes(under='aliceblue', over='r')
    norm = mpl.colors.Normalize(vmin, vmax)
    plt.imshow(data[image_index], cmap=cmap, norm=norm)
    plt.title(f"Image {image_index}")
    plt.colorbar()
    plt.show()
