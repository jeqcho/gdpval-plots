# GDPval Plots

`benchmark_results.yaml` is downloaded from [METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/).
`model_performance.csv` is GPT-5's transcription of the headline plot in [GDPval](https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf).

## Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.10 or newer

## Environment Setup
1. (Optional) Create and activate a virtual environment if you do not plan to use `uv venv`.
2. Run `uv sync` to install the locked dependencies. If you are offline, run `UV_CACHE_DIR=.uv-cache uv sync` so uv stores its cache locally.
   - The dependency set is defined in `pyproject.toml`. You can regenerate a `uv.lock` by running `uv lock` once you have network access.

## Usage
- Generate the merged CSV and plots:
  ```bash
  uv run python scripts/plot_model_performance.py
  ```
- Outputs:
  - `data/model_performance_merged.csv`: merged dataset restricted to the models in `model_performance.csv`.
  - `data/model_performance_projections.csv`: regression-based milestone calendar with columns `metric`, `percent`, `date`, and `line_type` (`overall` vs `frontier`).
  - `data/model_performance_projections.md`: markdown table containing the same milestone schedule as the CSV.
  - `figures/model_performance_actual_percent.png`
  - `figures/model_performance_log_percent.png`
  - `figures/model_performance_odds.png`
  - `figures/model_performance_logit.png`

    Each figure highlights frontier models (green) versus other models (gray), overlays the frontier-only regression line, and includes a legend. Milestone annotations list the frontier projections for the 50%, 75%, 90%, 95%, and 99% thresholds on the chart (overall projections remain available in the CSV), alongside the frontier regression equation (in `delta_days` since first release) and an "Industry Expert" reference line at p = 0.5 translated into the respective scale. All y-axes are scaled to represent probabilities from 0% to 100% via the appropriate transformation and tick labelling.

## Troubleshooting
- If `matplotlib` cannot find a GUI backend, set `MPLBACKEND=Agg` before running the script: `MPLBACKEND=Agg uv run python scripts/plot_model_performance.py`.
- When working without network access, uv operations that require downloading packages (e.g., `uv lock`) will fail; rerun them when connectivity is available.
