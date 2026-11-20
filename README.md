# Intro to Machine Learning — Quick, Draw! Coursework

Please read **`spec.pdf`** carefully for the full coursework specification, tasks, deliverables, and marking criteria.

## Repository Layout

- `spec.pdf` — Full coursework specification.
- `part1_report.tex`, `part2_report.tex` — LaTeX report templates for Part 1 and Part 2 (keep these in the repository root).
- `part1_preprocess.py`, `part1_knn.py`, `part1_logreg.py`, `part1_mlp.py`, `part1_cnn.py`, `part1_studio.py`, `part1_common.py` — Part 1 classification scripts and interactive studio.
- `part2_preprocess.py`, `part2_kmeans.py`, `part2_rnn.py`, `part2_genai.py`, `part2_common.py` — Part 2 generative modelling scripts.
- `binary_file_parser.py`, `helpers.py` — Shared Quick, Draw! `.bin` parser and plotting/rasterisation utilities.
- `requirements.txt` — Python dependencies.
- `data/` — Place the Quick, Draw! `.bin` dataset files here (ignored by git).

## Setup

The computing labs already have the required dependencies installed along with the Quick, Draw! dataset.

Otherwise, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `part1_studio.py` and `part2_genai.py` use PyQt6 for their interactive GUI (install `PyQt6` if not already available in your environment).

Get the data from the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset#get-the-data) and place the `.bin` files in the `data/` folder.

## Running the Scripts

Run each part in order so that intermediate outputs (`part1_outputs/`, `part2_outputs/`) are generated for subsequent scripts.

### Part 1: Sketch Classification

```bash
python3 part1_preprocess.py
python3 part1_knn.py
python3 part1_logreg.py
python3 part1_mlp.py
python3 part1_cnn.py
python3 part1_studio.py
```

### Part 2: Generative Sketch Modelling

```bash
python3 part2_preprocess.py
python3 part2_kmeans.py
python3 part2_rnn.py
python3 part2_genai.py
```
