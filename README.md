Tabulator
=========

A minimal Flask app scaffold (in the style of the referenced codex project) with a local_dev mode and a simple UI to upload a data table. Accepted formats: `.h5`, `.csv`, pickle DataFrame (`.pkl`/`.pickle`), or MATLAB table (`.mat`).

Quick start (Poetry)
-------------------

Install dependencies first, then run the app using Poetry.

1) Install Poetry (if not installed)
- Follow: https://python-poetry.org/docs/#installation
- Verify: `poetry --version`

2) Create env and install deps
- From the project root: `poetry install`
  - This creates a virtualenv and installs all dependencies (Flask, pandas, scipy, tables, etc.).

3) Run in local dev mode
- `poetry run python run_local.py`
- Or: `export LOCAL_DEV=1 && poetry run flask --app tabulator.app run --debug`

Project structure
-----------------

- `tabulator/`: Flask application package
  - `app.py`: application factory and wiring
  - `config.py`: configuration (BaseConfig and LocalDevConfig)
  - `routes.py`: main blueprint and upload handling
  - `templates/`: HTML templates
  - `static/`: static assets (CSS)
- `uploads/`: where uploaded files are stored (gitignored)
- `run_local.py`: convenience runner for local development
- `requirements.txt`: Python dependencies

Notes
-----

- Uploads are parsed into a pandas DataFrame for: `.csv`, `.h5` (first readable key), `.pkl`/`.pickle`, `.mat` (best-effort heuristics via SciPy).
- A simple in-memory LRU cache stores the most recent 10 datasets; the current dataset’s ID is kept in the session.
- Max upload size is 100 MB by default. Adjust in `tabulator/config.py` if needed.

Troubleshooting
---------------

- Error: `ModuleNotFoundError: No module named 'flask'`
  - Cause: Dependencies weren’t installed into the Poetry env yet.
  - Fix: Run `poetry install` first, then use `poetry run python run_local.py`.

- Build issues on macOS (rare):
  - Ensure Xcode Command Line Tools: `xcode-select --install`.
  - If PyTables complains about HDF5, install it via Homebrew: `brew install hdf5`.
  - If issues persist on Python 3.12, try Python 3.11 in Poetry: `poetry env use 3.11` then `poetry install`.
