emotion-detection
==============================
### Installation

1. **Install uv package manager** (if you haven't already)
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Or via pip
   pip install uv
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/serpentile-c137/emotion-detection.git
   cd emotion-detection
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   # Create and activate virtual environment with Python 3.8+
   uv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   # Or if you alredy have pyptoject.toml
   uv sync

   # If you want to manually install
   uv add numpy pandas scikit-learn nltk dvc
   ```


DVC commands

```bash
dvc stage add -n data_ingestion -d src/data/data_ingestion.py -o data/raw/train.csv -o data/raw/test.csv python src/data.data_ingestion.py
```
```bash
dvc stage add -n data_preprocessing -d src/data/data_preprocessing.py -d data/raw/train.csv -d data/raw/test.csv -o data/processed/train.csv -o data/processed/test.csv python src/data/data_preprocessing.py
```
```bash
dvc stage add -n feature_engineering -d src/features/features.py -d data/processed/train.csv -d data/processed/test.csv -o data/interim/train_bow.csv -o data/interim/test_bow.csv python src/features/features.py
```
```bash
dvc stage add -n model_training -d src/modelling/modelling.py -d data/interim/train_bow.csv -o models/random_forest_model.pkl python src/modelling/modelling.py
```
```bash
dvc stage add -n model_evaluation -d src/modelling/model_evaluation.py -d models/random_forest_model.pkl -d data/interim/test_bow.csv -o reports/metrics.json python src/modelling/model_evaluation.py
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
