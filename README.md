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
- Text columns are supported. Loaders only cast a column to numeric when all non-empty values are numeric-like; otherwise the column remains as text and is available for grouping or display without losing values.
- A simple in-memory LRU cache stores the most recent 10 datasets; the current dataset’s ID is kept in the session.
- Max upload size is 100 MB by default. Adjust in `tabulator/config.py` if needed.

Database connection (DataJoint)
-------------------------------

- The “Load From Database” option reuses stored credentials to open a DataJoint connection with `dj.conn`.
- Create a `.env` file (already gitignored) with the required values:

```
DJ_HOST=vfsmdatajoint01.fsm.northwestern.edu
DJ_USER=AppServer
DJ_PASSWORD=rujKab-sabmaj-sezqu6
DJ_SCHEMA=sln_results  # override if you need a different schema
```

- Once populated, switch to the “From Database” tab to fetch the available tables from the configured schema, choose one, and click “Load Table.” The backend opens a DataJoint connection (no custom port needed) and loads the selected relation into Tabulator.
- Tables are listed and selected using their DataJoint class names (e.g., `DatasetUncaging`) rather than the raw SQL table identifiers.
- If you change schemas frequently, update `DJ_SCHEMA` in `.env` (defaults to `sln_results`).

Sanity-check locally
--------------------

1) Install deps and run the app

- With Poetry: `poetry install && poetry run python run_local.py`
- Or with pip (example): `python -m pip install flask python-dotenv pandas scipy tables h5py && python run_local.py`

2) Create a sample CSV with text and numeric columns

```
name,score,group
-,unit,none
Alice,10,A
Bob,hello,B
Cara,20,A
```

3) Upload and query via curl (keeps session cookies so the API can see your dataset)

```
# In another terminal
curl -c cookies.txt -b cookies.txt -F "datafile=@sample.csv" -L -s -o /dev/null http://127.0.0.1:5001/upload

# List columns and numeric flags
curl -b cookies.txt http://127.0.0.1:5001/api/columns | jq .

# If you have a fully-numeric column (e.g., `score` only containing numbers),
# you can fetch a bar plot aggregation grouped by a text column:
curl -b cookies.txt "http://127.0.0.1:5001/api/plot/bar?value=score&group=group" | jq .
```

Expected behavior
-----------------

- Columns are considered numeric only if all non-empty values are numeric-like. Mixed columns (numbers plus words) remain text.
- Grouping works with text columns (e.g., `group`).
- Plotting requires a numeric `value` column; mixed or text-only value columns will be ignored during aggregation (non-numeric entries are dropped).
- Bar plots support ordering bars by label or by value/mean (ascending/descending) via the UI.

Troubleshooting
---------------

- Error: `ModuleNotFoundError: No module named 'flask'`
  - Cause: Dependencies weren’t installed into the Poetry env yet.
  - Fix: Run `poetry install` first, then use `poetry run python run_local.py`.

- Build issues on macOS (rare):
  - Ensure Xcode Command Line Tools: `xcode-select --install`.
  - If PyTables complains about HDF5, install it via Homebrew: `brew install hdf5`.
  - If issues persist on Python 3.12, try Python 3.11 in Poetry: `poetry env use 3.11` then `poetry install`.
