import pytest
import numpy as np
import os
import pickle

EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "data", "expected")
ACTUAL_DIR = os.path.join(os.path.dirname(__file__), "data", "actual")

def read(fpath):
    if not os.path.exists(fpath):
        return None
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data

def get_test_cases():
    if not os.path.exists(EXPECTED_DIR):
        return []
    cases = []
    for fname in os.listdir(EXPECTED_DIR):
        if fname.endswith(".res"):
            # Sample name: dataset=small_spherewithhole_k=14_eta_min=5_cost-fn=distortion.res
            try:
                dname = fname.split("dataset=")[1].split("_k=")[0]
                n_neighbors = int(fname.split("_k=")[1].split("_eta_min=")[0])
                min_cluster_size = int(fname.split("_eta_min=")[1].split("_cost-fn=")[0])
                cost_function = fname.split("_cost-fn=")[1].replace(".res", "")
                cases.append((n_neighbors, min_cluster_size, cost_function, dname))
            except Exception:
                continue
    return cases

# Ensure pytest finds tests even if there's no data locally generated.
# In CI, the data will be present, so this dynamic parametrize works smoothly.
TEST_CASES = get_test_cases()
if not TEST_CASES:
    TEST_CASES = [(14, 5, "distortion", "dummy_no_baseline")]

@pytest.mark.parametrize("n_neighbors,min_cluster_size,cost_function,dataset_name", TEST_CASES)
def test_end_to_end(n_neighbors, min_cluster_size, cost_function, dataset_name):
    fname = f"dataset={dataset_name}_k={n_neighbors}_eta_min={min_cluster_size}_cost-fn={cost_function}.res"
    expected_path = os.path.join(EXPECTED_DIR, fname)
    actual_path = os.path.join(ACTUAL_DIR, fname)
    
    data_pre = read(expected_path)
    if data_pre is None:
        pytest.skip(f"Expected baseline not found: {expected_path}")
        
    data_post = read(actual_path)
    if data_post is None:
        pytest.fail(f"Actual result not found: {actual_path}")

    y = np.allclose(data_pre["y"], data_post["y"])
    
    if data_pre["color_of_pts_on_tear"] is None or data_post["color_of_pts_on_tear"] is None:
        color = data_pre["color_of_pts_on_tear"] == data_post["color_of_pts_on_tear"]
    else:
        color = np.allclose(data_pre["color_of_pts_on_tear"], data_post["color_of_pts_on_tear"], equal_nan=True)

    Utilde = np.allclose(data_pre["Utilde"].toarray(), data_post["Utilde"].toarray())
    C = np.allclose(data_pre["C"].toarray(), data_post["C"].toarray())
    overlap = np.allclose(data_pre["n_Utilde_Utilde"].toarray(), data_post["n_Utilde_Utilde"].toarray())
    c = np.allclose(data_pre["c"], data_post["c"])

    assert y, f"y output matrix mismatch for {fname}"
    assert color, f"color mismatch for {fname}"
    assert Utilde, f"Utilde matrix mismatch for {fname}"
    assert C, f"C matrix mismatch for {fname}"
    assert overlap, f"overlap logic mismatch for {fname}"
    assert c, f"c attribute mismatch for {fname}"
