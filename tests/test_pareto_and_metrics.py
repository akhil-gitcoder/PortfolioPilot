"""Tests for dominance, sorting, crowding and metrics (hand-checkable cases)."""
import numpy as np
import pytest

from src.metrics import c_metric, hypervolume, igd, spacing
from src.multiobjective.pareto import (crowding_distance, dominates, fast_non_dominated_sort,
                                       nondominated_mask, dominance_matrix)

# the "growth up / moodiness down" example from the Phase 1 explanation, as minimisation
# P1 (10,18) P2 (7,9) P3 (6,6) P4 (7,12) P5 (5,8)  -> objectives (-growth, moodiness)
F5 = np.array([[-10, 18], [-7, 9], [-6, 6], [-7, 12], [-5, 8]], float)


def test_dominates_basic():
    assert dominates([1, 1], [2, 2])
    assert dominates([1, 2], [1, 3])
    assert not dominates([1, 3], [2, 2])
    assert not dominates([1, 1], [1, 1])


def test_nondominated_set_of_example():
    assert list(np.where(nondominated_mask(F5))[0]) == [0, 1, 2]    # P1,P2,P3


def test_sorting_ranks_of_example():
    assert list(fast_non_dominated_sort(F5)) == [0, 0, 0, 1, 1]


def test_crowding_boundaries_infinite():
    d = crowding_distance(np.array([[0, 3], [1, 2], [2, 1], [3, 0]], float))
    assert np.isinf(d[0]) and np.isinf(d[3]) and np.isfinite(d[1])


def test_deb_constrained_dominance_rules():
    F = np.array([[5, 5], [1, 1], [9, 9], [2, 2]], float)
    CV = np.array([0.0, 3.0, 0.0, 1.0])
    D = dominance_matrix(F, CV)
    assert D[0, 1]            # feasible beats infeasible even if worse in objectives
    assert D[3, 1]            # both infeasible: smaller violation wins
    assert not D[1, 3]
    assert D[0, 2] is np.True_ or D[0, 2]   # both feasible: ordinary dominance


def test_hypervolume_2d_known():
    # single point (1,1) with ref (3,3) -> area 4 ; add (2,0.5): union area
    assert hypervolume(np.array([[1.0, 1.0]]), [3, 3]) == pytest.approx(4.0)
    # points (1,2) and (2,1), ref (3,3): areas 2 + 2 - overlap 1 = 3
    assert hypervolume(np.array([[1.0, 2.0], [2.0, 1.0]]), [3, 3]) == pytest.approx(3.0)


def test_hypervolume_3d_known():
    assert hypervolume(np.array([[1, 1, 1.0]]), [3, 3, 3]) == pytest.approx(8.0)
    # two boxes: (1,1,2) -> 2*2*1=4 ; (2,2,1) -> 1*1*2=2 ; overlap (2,2,2)->1*1*1=1 => 5
    assert hypervolume(np.array([[1, 1, 2.0], [2, 2, 1.0]]), [3, 3, 3]) == pytest.approx(5.0)


def test_hypervolume_ignores_points_outside_reference():
    assert hypervolume(np.array([[4.0, 4.0]]), [3, 3]) == 0.0


def test_igd_spacing_cmetric():
    R = np.array([[0, 0], [1, 1]], float)
    assert igd(R, R) == 0.0
    assert igd(R, np.array([[0, 0]], float)) == pytest.approx(np.sqrt(2) / 2)
    even = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    assert spacing(even) == pytest.approx(0.0)
    A = np.array([[1, 1]], float)
    B = np.array([[2, 2], [0, 5]], float)
    assert c_metric(A, B) == pytest.approx(0.5)
