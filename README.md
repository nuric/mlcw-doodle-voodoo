# Doodle Voodoo - Quick, Draw! Machine Learning Coursework

Skeleton repository for the **Machine Learning** coursework on sketch classification and generative sketch modelling using Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

> **Start here:** Read **[`spec.pdf`](spec.pdf)** carefully for the full specification, tasks, deliverables, and marking criteria.

---

## Repository Structure

In order of what is useful and relevant to the coursework, the repository contains the following files and directories:

| File / Directory | Description |
| --- | --- |
| `spec.pdf` | Full coursework specification |
| `part1_common.py`, `part2_common.py` | Shared model/training utilities for Parts 1 and 2 |
| `part1_preprocess.py` | Part 1 data loading and preprocessing |
| `part1_knn.py`, `part1_logreg.py`, `part1_mlp.py`, `part1_cnn.py` | Part 1 sketch classifiers |
| `part1_studio.py` | Part 1 interactive sketch classification GUI |
| `part2_preprocess.py` | Part 2 stroke sequence preprocessing |
| `part2_kmeans.py`, `part2_rnn.py` | Part 2 generative models (K-Means clustering & RNN) |
| `part2_genai.py` | Part 2 interactive generative sketch GUI |
| `part1_report.tex`, `part2_report.tex` | LaTeX report templates for Part 1 and Part 2 *(keep in the repository root)* |
| `binary_file_parser.py`, `helpers.py` | Quick, Draw! `.bin` parser and rasterisation/plotting helpers |
| `requirements.txt` | Python package dependencies |
| `data/` | Place Quick, Draw! `.bin` dataset files here *(ignored by git)* |

---

## Getting Started

The **computing labs** already have the required Python packages and the Quick, Draw! dataset installed.

If you are working on your own machine:

1. **Create and activate a virtual environment, then install dependencies:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   > **Note:** `part1_studio.py` and `part2_genai.py` use **PyQt6** for their interactive GUIs. Install `PyQt6` if it is not already available in your environment.

2. **Download the dataset:**
   Download the `.bin` files from the [Quick, Draw! dataset repository](https://github.com/googlecreativelab/quickdraw-dataset#get-the-data) and place them inside `data/`.

---

## Running the Scripts

Run each part **in order** so that intermediate outputs (`part1_outputs/`, `part2_outputs/`) are generated for the downstream scripts and GUIs.

### Part 1 — Sketch Classification

```bash
python3 part1_preprocess.py
python3 part1_knn.py
python3 part1_logreg.py
python3 part1_mlp.py
python3 part1_cnn.py
python3 part1_studio.py
```

### Part 2 — Generative Sketch Modelling

```bash
python3 part2_preprocess.py
python3 part2_kmeans.py
python3 part2_rnn.py
python3 part2_genai.py
```
