# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


#=========================================================
# MAIN CLASS FOR CREATING A WEIGHTED SAMPLER
#=========================================================

# ------------ Weighted Sampler to Handle Class Imbalance ---------------------------

def make_weighted_sampler(dataset):
    """
    Create a weighted sampler to handle class imbalance during training.

    The function calculates how many samples belong to each class and assigns
    higher sampling weights to minority classes. This helps the DataLoader sample
    real/live and fake/spoof examples more evenly during training.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset containing a dataframe `df` with an integer `label` column.

    Returns
    -------
    torch.utils.data.WeightedRandomSampler
        Sampler that can be passed to a PyTorch DataLoader.

    Notes
    -----
    - Does not modify the dataset.
    - Use this sampler instead of `shuffle=True` in the DataLoader.
    - replacement=True allows minority samples to be selected more than once.
    """

    # Get all labels
    labels = dataset.df['label'].values

    # Count samples in each class (real samples/fake samples)
    counts = np.bincount(labels)

    # Assign higher weight to minority class
    weights = 1.0 / counts

    # Create weighted sampler - makes training balanced
    return torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(weights[labels]),
        len(labels),
        replacement=True
    )


# ============================================================================
#  Dataset-aware weighted sampler for combined training
# ============================================================================
 
def make_combined_weighted_sampler(df, group_cols=('dataset', 'label')):
    """
    df : the dataframe backing the training Dataset (must have 'dataset' and
         'label' columns — combined_train_df in our case)
    Returns a WeightedRandomSampler that balances across every unique
    (dataset, label) combination present in df.
    """
    group_counts = df.groupby(list(group_cols)).size()               # count per (dataset, label) group
    group_weights = 1.0 / group_counts                                # inverse frequency -> smaller groups get bigger weight
 
    # map each row to its group's weight
    sample_weights = df.apply(lambda row: group_weights[tuple(row[c] for c in group_cols)], axis=1)
 
    sampler = WeightedRandomSampler(
        weights=sample_weights.values,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler



# ============================================================================
#  Sanity checks for weighted sampler combined
# ============================================================================
def verify_combined_sampler_weights(df, sampler, group_cols=('dataset', 'label'), tol=1e-9):
    """
    Verifies a WeightedRandomSampler's weights for dataset/label balancing.
 
    Runs two checks:
    1. Recomputes weight = 1/group_count for every row from scratch and
       compares it against sampler.weights row-by-row.
    2. Confirms every group's total mass (weight * count) is equal across
       groups — the property that makes the balancing actually work.
 
    Parameters
    ----------
    df : pandas.DataFrame — must contain all columns in group_cols.
    sampler : torch.utils.data.WeightedRandomSampler — the sampler to check.
    group_cols : tuple of str, default ('dataset', 'label') — columns
        defining a "group"; each unique combination is balanced separately.
    tol : float, default 1e-9 — max allowed floating-point difference
        before a weight is flagged as a mismatch.
 
    Returns
    -------
    bool — True if both checks pass. Diagnostic output (including up to 5
    mismatched rows and per-group mass values) is printed either way.
 
    Assumes the sampler exposes a `.weights` attribute, as standard
    PyTorch WeightedRandomSampler does.
    """
    # ---- recompute the CORRECT weights independently, from scratch ----
    group_counts = df.groupby(list(group_cols)).size()
    group_weights_ref = 1.0 / group_counts

    expected_weights = df.apply(
        lambda row: group_weights_ref[tuple(row[c] for c in group_cols)], axis=1
    ).values

    # ---- pull the ACTUAL weights the sampler was built with ----
    actual_weights = np.asarray(sampler.weights)

    print("=" * 70)
    print("CHECK 1 — do the sampler's actual weights match the expected formula?")
    print("=" * 70)
    if len(actual_weights) != len(expected_weights):
        print(f"FAIL — length mismatch: sampler has {len(actual_weights)} weights, "
              f"df has {len(expected_weights)} rows.")
        return False

    diffs = np.abs(actual_weights - expected_weights)
    max_diff = diffs.max()
    n_mismatched = (diffs > tol).sum()

    if n_mismatched == 0:
        print(f"PASS — all {len(actual_weights)} weights match 1/group_count exactly "
              f"(max diff: {max_diff:.2e})")
    else:
        print(f"FAIL — {n_mismatched} of {len(actual_weights)} weights don't match "
              f"(max diff: {max_diff:.2e})")
        bad_idx = np.where(diffs > tol)[0][:5]
        for i in bad_idx:
            row = df.iloc[i]
            print(f"    row {i}: {tuple(row[c] for c in group_cols)} "
                  f"-> expected {expected_weights[i]:.6f}, got {actual_weights[i]:.6f}")

    # ---- CHECK 2: does every group's total mass (weight * count) come out equal? ----
    print("\n" + "=" * 70)
    print("CHECK 2 — does every group end up with EQUAL total sampling mass?")
    print("=" * 70)
    print(f"{'Group':<25}{'Count':>8}{'Weight':>14}{'Total Mass (weight*count)':>28}")
    print("-" * 75)

    masses = []
    for group_key, count in group_counts.items():
        w = group_weights_ref[group_key]
        mass = w * count
        masses.append(mass)
        label_str = str(group_key)
        print(f"{label_str:<25}{count:>8}{w:>14.6f}{mass:>28.6f}")

    masses = np.array(masses)
    if np.allclose(masses, masses[0], atol=1e-6):
        print(f"\nPASS — all groups have equal total mass (~{masses[0]:.4f} each), "
              f"confirming the reweighting cancels out group size correctly.")
    else:
        print(f"\nFAIL — masses are NOT equal: {masses}")
        print("   This means the balancing is NOT working as intended.")

    return (n_mismatched == 0) and np.allclose(masses, masses[0], atol=1e-6)