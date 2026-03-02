"""
utils/evaluation.py
-------------------
Shared evaluation functions for the TripAdvisor IR project.

Previously these functions were copy-pasted into each individual notebook
(Notebooks 2, 3, 4, 5), violating DRY.  This module centralises the logic.

Usage in a notebook:
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '..'))
    from utils.evaluation import eval_level_1, extract_subcategories, eval_level_2
"""

import pandas as pd
import numpy as np


def eval_level_1(query_typeR, sorted_test_typeR_list):
    """
    Level 1 Ranking Error.

    Find the index of the first test document whose `typeR` matches the query.
    Error = index (0-indexed rank), so a perfect first-place match = error 0.
    Returns None if the query typeR is NaN or does not exist in the test pool
    (undefined case per assignment spec).

    Parameters
    ----------
    query_typeR : str
        The typeR of the query place (one of 'H', 'R', 'A', 'AP').
    sorted_test_typeR_list : list[str]
        Ordered list of typeR values from the ranked retrieval output.

    Returns
    -------
    int or None
    """
    if pd.isna(query_typeR) or query_typeR not in sorted_test_typeR_list:
        return None
    for i, t in enumerate(sorted_test_typeR_list):
        if t == query_typeR:
            return i
    return None


def extract_subcategories(row):
    """
    Type-aware subcategory extraction for Level 2 evaluation.

    FIX: Previous implementation pooled ALL subcategory columns (activiteSubType,
    restaurantType, restaurantTypeCuisine, priceRange) into one flat set for every
    place, regardless of its type. This caused spurious cross-type matches, e.g.,
    a Hotel with priceRange='mid-range' matching a Restaurant with priceRange='mid-range'.

    Per the assignment spec, subcategory comparison is type-specific:
    - H  (Hotel)      -> priceRange
    - R  (Restaurant) -> restaurantType, restaurantTypeCuisine
    - A / AP (Attraction) -> activiteSubType

    Parameters
    ----------
    row : pd.Series
        A row from a merged DataFrame containing typeR and subcategory columns.

    Returns
    -------
    set[str]
    """
    cats = set()
    type_r = str(row.get('typeR', '')).strip()

    if type_r == 'H':
        val = row.get('priceRange', '')
        if pd.notna(val) and str(val).strip():
            for p in str(val).split(','):
                cats.add(p.strip().lower())

    elif type_r == 'R':
        for col in ['restaurantType', 'restaurantTypeCuisine']:
            val = row.get(col, '')
            if pd.notna(val) and str(val).strip():
                for p in str(val).split(','):
                    cats.add(p.strip().lower())

    elif type_r in ('A', 'AP'):
        val = row.get('activiteSubType', '')
        if pd.notna(val) and str(val).strip():
            for p in str(val).split(','):
                cats.add(p.strip().lower())

    return cats


def eval_level_2(query_subcats, sorted_test_indices, test_subcats_list):
    """
    Level 2 Ranking Error.

    Find the index of the first test document that shares at least one
    subcategory with the query.  Returns None if query has no subcategories
    (undefined case) or no match is found.

    Parameters
    ----------
    query_subcats : set[str]
        Subcategories of the query place (from extract_subcategories).
    sorted_test_indices : array-like[int]
        Ranked indices into test_subcats_list, from highest to lowest score.
    test_subcats_list : list[set[str]]
        Precomputed subcategory sets for every test document.

    Returns
    -------
    int or None
    """
    if not query_subcats:
        return None
    for i, test_idx in enumerate(sorted_test_indices):
        if query_subcats.intersection(test_subcats_list[test_idx]):
            return i
    return None


def per_row_minmax(matrix):
    """
    Per-row Min-Max normalization.

    FIX: Previously a global min/max was applied across the entire score matrix,
    compressing score ranges inconsistently across queries with different specificity.
    Per-row normalization ensures each query's score vector is independently scaled
    to [0, 1] before fusion.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_queries, n_docs)

    Returns
    -------
    np.ndarray, same shape, values in [0, 1]
    """
    mins = matrix.min(axis=1, keepdims=True)
    maxs = matrix.max(axis=1, keepdims=True)
    denom = np.where(maxs - mins == 0, 1, maxs - mins)
    return (matrix - mins) / denom
