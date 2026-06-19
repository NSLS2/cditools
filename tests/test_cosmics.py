from __future__ import annotations

import numpy as np

from cditools.analysis_scripts.cosmics import check_empty, find_cosmics, label_cosmics


def test_check_empty():
    data = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 0]]])
    empty_indices = check_empty(data)
    assert empty_indices == [0, 2]


def test_find_cosmics():
    data = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [5, 0]]])
    cosmic_points = find_cosmics(data, vmin=1, vmax=5)
    assert cosmic_points == {1, 2}


def test_label_cosmics():
    data = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [5, 0]]])
    labeled_data = label_cosmics(data)
    expected_labeled_data = ([0, 0, 0], 0.0)
    assert labeled_data == expected_labeled_data
