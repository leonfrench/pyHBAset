# pyHBAset

pyHBAset is a Streamlit app for testing brain-region-specific gene expression
using adult and fetal microarray data from the
[Allen Human Brain Atlas](https://human.brain-map.org/).

The app supports:

- single-gene average expression views with colored Allen SVG anatomy
- gene-list AUROC enrichment by brain region
- optional background gene universes for gene-list tests
- adult and fetal Allen datasets

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate HBA_enrichment_app
```

The same Python dependencies are also listed in `requirements-streamlit.txt`.

## Data

The app expects Allen raw data under `data/raw/` and compact preprocessed app
inputs under `data/processed/`.

If the raw Allen data are not already present, download them with:

```bash
./download_expression_data.sh
```

Build or refresh the compact app inputs with:

```bash
python preprocess_adult_hba.py
python preprocess_fetal_hba.py
```

These scripts write:

- `adult_hba_average_expression.csv.gz`
- `adult_hba_rank_zscore_expression.csv.gz`
- `adult_hba_gene_symbol_aliases.csv.gz`
- `adult_hba_region_lookup.csv.gz`
- `adult_hba_region_donor_counts.csv.gz`
- `adult_hba_manifest.json`
- `fetal_hba_average_expression.csv.gz`
- `fetal_hba_rank_zscore_expression.csv.gz`
- `fetal_hba_gene_symbol_aliases.csv.gz`
- `fetal_hba_region_lookup.csv.gz`
- `fetal_hba_region_donor_counts.csv.gz`
- `fetal_hba_manifest.json`

## Run

```bash
streamlit run streamlit_app.py
```

Choose the dataset in the sidebar. Enter one gene to view average expression
across Allen brain regions. Enter two or more genes to calculate AUROC,
Mann-Whitney p-values, and FDR values for every brain region.

## Project Files

- `streamlit_app.py`: Streamlit user interface and enrichment workflow
- `adult_hba_app_utils.py`: SVG coloring, tooltip, and legend helpers
- `preprocess_adult_hba.py`: adult Allen HBA preprocessing
- `preprocess_fetal_hba.py`: fetal Allen HBA preprocessing
- `download_expression_data.sh`: Allen raw data downloader
