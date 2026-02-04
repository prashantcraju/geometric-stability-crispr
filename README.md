# From syntax to semantics: the geometric crisis in perturbation biology

## Setup & Dependencies

### 1. Python Environment
This project requires specific versions of PyTorch to ensure compatibility with `scGPT`.
```bash
pip install -r requirements.txt
```

> **Note:** If you are using Python 3.12, this project enforces `torch==2.3.1` and `torchtext==0.18.0` to avoid compatibility issues.

### 2. Download scGPT Model
To run the scGPT analysis, you must download the pre-trained foundation model weights locally.

1. Download the **"Whole Human"** model (`scGPT_human`) from the [official scGPT repository](https://github.com/bowang-lab/scGPT) or their provided Google Drive links.
2. Unzip the folder to a location on your machine (e.g., `./models/scGPT_human`).
3. Ensure the folder contains `best_model.pt`, `vocab.json`, and `args.json`.

## Usage

### 1. Standard Geometric Stability (PCA)
To reproduce Figure 2 using the standard PCA workflow:

1. Run `geometric_stability_main_analysis.py`. This will produce several CSV files with results.
2. Open `fig_2.py` and update the file path to point to the generated `shesha_crispr_results_euclidean.csv`.
3. Run `fig_2.py`.

### 2. Semantic Stability (scGPT)
To compare the standard results against the scGPT foundation model:

1. Open `scgpt_analysis.py`.
2. Locate the `model_dir` parameter in the `if __name__ == "__main__":` block at the bottom of the file.
3. Update it to the absolute path where you unzipped the model weights:
   ```python
   model_dir = "/path/to/your/scGPT_human"
    ```
4. Run the script:
   ```bash
   python scgpt_analysis.py
    ```

> **Note:** scGPT requires raw counts (integers). The script handles re-loading raw data if available, but ensure your AnnData object is not pre-normalized if loading from an external file.

## Notes
The code in `geometric_stability_main_analysis.py` was originally used for another paper, *Geometric Stability: The Missing Axis of Representations* (arXiv:2601.09173).
