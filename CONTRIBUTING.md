# Contributing to pyRATS

Thank you for your interest in contributing to pyRATS! This guide will help you set up your local environment and run the regression test suite.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mishne-Lab/pyRATS.git
   cd pyRATS
   ```

2. **Install in editable mode with test dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -e .[test]
   ```
   *Note: This will install core dependencies (numpy, scipy, scikit-learn) as well as test/example dependencies (pandas, matplotlib, Pillow, imageio, seaborn).*

## Regression Testing Against Baseline

The pyRATS test suite is designed to ensure that code changes do not break existing logic. This is done by comparing the output of the **current code** against a **known-good baseline commit** (`8e27b77b3247e1ac7447e7ea429e0f0c103a3fb0`).

### 1. Simple Comparison (Fast Mode)
To run a quick regression test:

```bash
# A. Generate "Expected" snapshots from the baseline source
git checkout 8e27b77b3247e1ac7447e7ea429e0f0c103a3fb0 -- src/
python tests/scripts/generate_snapshots.py --path-to-results-dir tests/data/expected --fast-mode

# B. Restore your current branch's source code
git checkout HEAD -- src/

# C. Generate "Actual" snapshots from your current code
python tests/scripts/generate_snapshots.py --path-to-results-dir tests/data/actual --fast-mode

# D. Run the comparison
pytest tests/test_e2e.py
```

### 2. Full Regression Test
To run the complete suite of datasets and parameters:

```bash
# A. Generate "Expected" snapshots from baseline
git checkout 8e27b77b3247e1ac7447e7ea429e0f0c103a3fb0 -- src/
python tests/scripts/generate_snapshots.py --path-to-results-dir tests/data/expected

# B. Restore current source and generate "Actual" snapshots
git checkout HEAD -- src/
python tests/scripts/generate_snapshots.py --path-to-results-dir tests/data/actual

# C. Run comparison
pytest tests/test_e2e.py
```

## Continuous Integration
- **Pushes**: Automatically runs the test suite in **Fast Mode**.
- **Pull Requests**: Automatically runs the **Full Regression Test**.
- **Manual Trigger**: You can manually trigger the "CI" workflow from the Actions tab on GitHub and toggle the `fast_mode` input.
