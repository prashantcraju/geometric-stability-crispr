#  Geometric coherence of single-cell CRISPR perturbations reveals regulatory architecture and predicts cellular stress
<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2604.16642" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-blue" /></a>
    <a style="text-decoration:none !important;" href="https://huggingface.co/papers/2604.16642" alt="Hugging Face Papers"><img src="https://img.shields.io/badge/paper-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" /></a>
</p>



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
To reproduce the figures using the standard PCA workflow:

1. Run `main_analysis_updated.py`. This will produce several CSV files with results.
2. Open the relevant figure script (`figs/fig2,4,5.py`, `figs/fig3.py`, `figs/fig_norman.py`, or `figs/fig_replogle.py`) and update the file path to point to the generated CSV files (e.g., `shesha_crispr_results_euclidean.csv`).
3. Run the figure script.

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

## Citation

```bibtex
@article{raju2026crispr,
  title = {Geometric coherence of single-cell CRISPR perturbations reveals regulatory architecture and predicts cellular stress},
  author = {Raju, Prashant C.},
  journal = {arXiv preprint arXiv:2604.16642},
  year = {2026}
}
```
