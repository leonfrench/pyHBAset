# pyHBAset

pyHBAset is a Streamlit app for testing brain-region-specific gene expression
using adult and fetal microarray data from the
[Allen Human Brain Atlas](https://human.brain-map.org/).

The app supports:

- single-gene average expression views with colored Allen SVG anatomy
- gene-list AUROC enrichment by brain region
- optional background gene universes for gene-list tests
- adult and fetal Allen datasets

## Background and attribution

This project was originally developed by Leon French in 2013 and hosted by Paul Pavlidis and the Pavlidis lab at [UBC](https://hbaset.msl.ubc.ca/). We thank Paul Pavlidis and the Pavlidis lab for hosting and keeping it online for many years.

Derek Howard later re-implemented and extended the original Java version in Python, and that codebase was used to study genes associated with anorexia nervosa in:
Howard D, et al. *Molecular neuroanatomy of anorexia nervosa* (2020). [PubMed](https://pubmed.ncbi.nlm.nih.gov/32651428/)

The present version builds on that analysis code:
[derekhoward/molecular_AN](https://github.com/derekhoward/molecular_AN)

This interactive web application was then built by Leon on top of that codebase by OpenAI Codex.

## Data sources and references

This project relies on Allen Institute human brain atlas resources, including:

- Hawrylycz MJ, et al. *An anatomically comprehensive atlas of the adult human brain transcriptome* (2012). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243026/)
- Ding S-L, et al. *Comprehensive cellular-resolution atlas of the adult human brain* (2016). [PubMed](https://pubmed.ncbi.nlm.nih.gov/27418273/).


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
