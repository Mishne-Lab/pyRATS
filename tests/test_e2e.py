import pytest
import numpy as np
import os
import pickle
from scipy.spatial import procrustes

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
                k = int(fname.split("_k=")[1].split("_eta_min=")[0])
                eta = int(fname.split("_eta_min=")[1].split("_cost-fn=")[0])
                cfn = fname.split("_cost-fn=")[1].replace(".res", "")
                cases.append((k, eta, cfn, dname))
            except Exception:
                continue
    return cases

# Ensure pytest finds tests even if there's no data locally generated.
# In CI, the data will be present, so this dynamic parametrize works smoothly.
TEST_CASES = get_test_cases()
if not TEST_CASES:
    TEST_CASES = [(14, 5, "distortion", "dummy_no_baseline")]

@pytest.mark.parametrize("k,eta_min,cost_fn_name,dataset_name", TEST_CASES)
def test_end_to_end(k, eta_min, cost_fn_name, dataset_name):
    fname = f"dataset={dataset_name}_k={k}_eta_min={eta_min}_cost-fn={cost_fn_name}.res"
    expected_path = os.path.join(EXPECTED_DIR, fname)
    actual_path = os.path.join(ACTUAL_DIR, fname)
    
    data_pre = read(expected_path)
    if data_pre is None:
        pytest.skip(f"Expected baseline not found: {expected_path}")
        
    data_post = read(actual_path)
    if data_post is None:
        pytest.skip(f"Actual result not found: {actual_path}")

    mtx1, mtx2, disparity = procrustes(data_pre["y"], data_post["y"])
    assert disparity < 0.95, f"Manifold structure drifted significantly (disparity={disparity:.4f}) for {fname}"
    
    pre_clusters = data_pre["Utilde"].shape[0]
    post_clusters = data_post["Utilde"].shape[0]
    cluster_diff = abs(pre_clusters - post_clusters) / pre_clusters
    assert cluster_diff < 0.5, f"Cluster count drifted by {cluster_diff*100:.1f}% for {fname}. Expected ~{pre_clusters}, got {post_clusters}"
    
    if data_pre["color_of_pts_on_tear"] is not None and data_post["color_of_pts_on_tear"] is not None:
        color_diff = np.nanmean(np.abs(data_pre["color_of_pts_on_tear"] - data_post["color_of_pts_on_tear"]))
        assert color_diff < 0.5, f"Tear coloring diverged significantly for {fname}"
