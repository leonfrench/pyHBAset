"""Preprocess fetal Allen Human Brain Atlas data for the Streamlit app.

This script builds compact fetal matrices from the Allen fetal LMD matrix
folders:

1. Average expression by gene and brain region, averaged across fetal donors.
2. Rank-z-scored expression by gene and brain region, matching the notebook
   workflow used for multi-gene AUROC analyses.
3. Gene-symbol aliases from Allen's original symbols to the reannotated symbols.
4. Fetal region lookup metadata for SVG mapping.
5. Fetal region donor/sample coverage.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_FETAL_DIR = Path("data/raw/allen_human_fetal_brain")
ANNOTATION_FILE = Path("data/raw/gene_symbol_annotations/AllenInstitute_custom_Agilent_Array.txt")
OUTPUT_DIR = Path("data/processed")

AVERAGE_EXPRESSION_FILE = "fetal_hba_average_expression.csv.gz"
RANK_ZSCORE_FILE = "fetal_hba_rank_zscore_expression.csv.gz"
ALIASES_FILE = "fetal_hba_gene_symbol_aliases.csv.gz"
REGION_LOOKUP_FILE = "fetal_hba_region_lookup.csv.gz"
REGION_DONOR_COUNTS_FILE = "fetal_hba_region_donor_counts.csv.gz"
MANIFEST_FILE = "fetal_hba_manifest.json"


def strip_left_right(structure_name: str) -> str:
    fragments = []
    for fragment in str(structure_name).split(","):
        if fragment.strip() not in {"left", "right", "Left", "Right"}:
            fragments.append(fragment)
    return ",".join(fragments).strip()


def find_fetal_folders(raw_fetal_dir: Path) -> list[Path]:
    fetal_folders = sorted(raw_fetal_dir.glob("lmd_matrix_*"))
    if not fetal_folders:
        raise FileNotFoundError(f"No fetal LMD matrix folders found under {raw_fetal_dir}")
    return fetal_folders


def donor_id_from_folder(fetal_folder: Path) -> str:
    return fetal_folder.name.split("_")[-1]


def read_expression_file(file_name: Path) -> pd.DataFrame:
    expression_df = pd.read_csv(
        file_name,
        index_col=0,
        header=None,
        dtype=np.float32,
        memory_map=True,
    )
    expression_df.index.rename("probe_id", inplace=True)
    return expression_df


def read_samples_file(samples_file: Path) -> pd.DataFrame:
    samples = pd.read_csv(samples_file)
    samples.set_index(samples.index + 1, inplace=True)
    samples.index.rename("sample_id", inplace=True)
    samples["structure_name"] = samples["structure_name"].map(strip_left_right)
    return samples


def read_probe_reannotations(annotation_file: Path) -> pd.DataFrame:
    reannotations = pd.read_table(
        annotation_file,
        usecols=["#PROBE_ID", "Gene_symbol"],
    )
    reannotations.rename(
        columns={"#PROBE_ID": "probe_name", "Gene_symbol": "gene_symbol"},
        inplace=True,
    )
    reannotations.dropna(inplace=True)

    split_symbols = (
        reannotations["gene_symbol"]
        .astype(str)
        .str.split(";", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("gene_symbol")
    )
    reannotations = reannotations.drop(columns=["gene_symbol"]).join(split_symbols)
    reannotations["gene_symbol"] = reannotations["gene_symbol"].str.strip()
    reannotations = reannotations[reannotations["gene_symbol"] != ""]
    return reannotations.drop_duplicates()


def read_probes_file(
    probes_file: Path,
    annotation_file: Path,
    probe_strategy: str,
) -> pd.DataFrame:
    probes = pd.read_csv(
        probes_file,
        usecols=["probeset_id", "probeset_name", "gene_symbol"],
    )
    probes.rename(
        columns={"probeset_id": "probe_id", "probeset_name": "probe_name"},
        inplace=True,
    )

    if probe_strategy == "reannotator":
        reannotations = read_probe_reannotations(annotation_file)
        probes = probes.drop(columns=["gene_symbol"]).merge(
            reannotations,
            on="probe_name",
            how="inner",
        )
    elif probe_strategy != "default":
        raise ValueError(f"Unsupported probe strategy: {probe_strategy}")

    probes.dropna(subset=["gene_symbol"], inplace=True)
    probes["gene_symbol"] = probes["gene_symbol"].astype(str).str.strip()
    probes = probes[probes["gene_symbol"] != ""]
    probes.set_index("probe_id", inplace=True)
    return probes


def get_expression_by_genes(exp_df: pd.DataFrame, probes_df: pd.DataFrame) -> pd.DataFrame:
    annotated_df = exp_df.merge(
        probes_df[["gene_symbol"]],
        left_index=True,
        right_index=True,
        how="inner",
    )
    return annotated_df.groupby("gene_symbol").mean()


def get_mean_expression_by_brain_area(
    exp_by_genes: pd.DataFrame,
    samples_df: pd.DataFrame,
) -> pd.DataFrame:
    if exp_by_genes.T.shape[0] != samples_df.shape[0]:
        raise ValueError(
            "Expression sample count does not match sample annotation rows: "
            f"{exp_by_genes.T.shape[0]} vs {samples_df.shape[0]}"
        )

    annotated_df = exp_by_genes.T.merge(
        samples_df[["structure_name"]],
        left_index=True,
        right_index=True,
    )
    expression_by_structure = annotated_df.groupby("structure_name").mean()
    expression_by_structure.T.index.rename("gene_symbol", inplace=True)
    return expression_by_structure.T


def rank_zscore_by_gene(exp_by_region: pd.DataFrame) -> pd.DataFrame:
    ranked = exp_by_region.rank(axis=0, ascending=True, method="average")
    row_mean = ranked.mean(axis=1)
    row_std = ranked.std(axis=1, ddof=0).replace(0, np.nan)
    return ranked.sub(row_mean, axis=0).div(row_std, axis=0)


def process_donor(
    fetal_folder: Path,
    annotation_file: Path,
    probe_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_id = donor_id_from_folder(fetal_folder)
    print(f"Processing fetal donor {donor_id}")

    expression = read_expression_file(fetal_folder / "expression_matrix.csv")
    samples = read_samples_file(fetal_folder / "columns_metadata.csv")
    probes = read_probes_file(fetal_folder / "rows_metadata.csv", annotation_file, probe_strategy)

    exp_by_genes = get_expression_by_genes(expression, probes)
    exp_by_region = get_mean_expression_by_brain_area(exp_by_genes, samples)
    ranked_zscore = rank_zscore_by_gene(exp_by_region)

    print(
        f"  donor matrix: {exp_by_region.shape[0]:,} genes x "
        f"{exp_by_region.shape[1]:,} regions"
    )
    return exp_by_region.astype(np.float32), ranked_zscore.astype(np.float32)


def average_donor_matrices(matrices: Iterable[pd.DataFrame]) -> pd.DataFrame:
    region_by_gene = [matrix.T for matrix in matrices]
    combined = pd.concat(region_by_gene, axis=0, sort=False)
    averaged = combined.groupby(combined.index).mean().T
    averaged.index.name = "gene_symbol"
    averaged.sort_index(axis=0, inplace=True)
    averaged.sort_index(axis=1, inplace=True)
    return averaged.astype(np.float32)


def build_alias_table(
    fetal_folders: list[Path],
    annotation_file: Path,
    available_symbols: pd.Index,
) -> pd.DataFrame:
    probe_frames = []
    for fetal_folder in fetal_folders:
        probes = pd.read_csv(
            fetal_folder / "rows_metadata.csv",
            usecols=["probeset_name", "gene_symbol"],
        )
        probes.rename(columns={"probeset_name": "probe_name"}, inplace=True)
        probe_frames.append(probes)

    original = pd.concat(probe_frames, ignore_index=True).dropna()
    original["gene_symbol"] = original["gene_symbol"].astype(str).str.strip()
    original = original[original["gene_symbol"] != ""].drop_duplicates()

    reannotated = read_probe_reannotations(annotation_file)
    symbol_pairs = original.merge(
        reannotated.rename(columns={"gene_symbol": "reannotated_gene_symbol"}),
        on="probe_name",
        how="inner",
    )

    alias_rows = []
    alias_rows.append(
        pd.DataFrame(
            {
                "input_symbol": available_symbols.astype(str),
                "gene_symbol": available_symbols.astype(str),
                "source": "reannotated_exact",
            }
        )
    )
    alias_rows.append(
        symbol_pairs.rename(
            columns={
                "gene_symbol": "input_symbol",
                "reannotated_gene_symbol": "gene_symbol",
            }
        )[["input_symbol", "gene_symbol"]].assign(source="allen_original_to_reannotated")
    )

    aliases = pd.concat(alias_rows, ignore_index=True).dropna()
    aliases["input_symbol"] = aliases["input_symbol"].astype(str).str.strip()
    aliases["gene_symbol"] = aliases["gene_symbol"].astype(str).str.strip()
    aliases = aliases[aliases["gene_symbol"].isin(set(available_symbols))]
    aliases["input_symbol_normalized"] = aliases["input_symbol"].str.upper()
    aliases = (
        aliases.groupby(["input_symbol", "input_symbol_normalized", "gene_symbol", "source"])
        .size()
        .reset_index(name="probe_count")
        .sort_values(["input_symbol_normalized", "gene_symbol", "source"])
    )
    return aliases


def build_region_lookup(fetal_folders: list[Path]) -> pd.DataFrame:
    metadata_frames = []
    for fetal_folder in fetal_folders:
        metadata_frames.append(pd.read_csv(fetal_folder / "columns_metadata.csv"))

    metadata = pd.concat(metadata_frames, ignore_index=True)
    metadata["structure_name"] = metadata["structure_name"].map(strip_left_right)
    keep_cols = [
        col
        for col in ["structure_id", "structure_acronym", "structure_name"]
        if col in metadata.columns
    ]
    return metadata[keep_cols].drop_duplicates().sort_values(["structure_name", "structure_id"])


def build_region_donor_counts(fetal_folders: list[Path]) -> pd.DataFrame:
    donor_region_rows = []
    for fetal_folder in fetal_folders:
        donor_id = donor_id_from_folder(fetal_folder)
        samples = read_samples_file(fetal_folder / "columns_metadata.csv")
        donor_counts = samples.groupby("structure_name").size().reset_index(name="sample_count")
        donor_counts["donor_id"] = donor_id
        donor_region_rows.append(donor_counts)

    donor_regions = pd.concat(donor_region_rows, ignore_index=True)
    region_donor_counts = (
        donor_regions.groupby("structure_name")
        .agg(
            donor_count=("donor_id", "nunique"),
            sample_count=("sample_count", "sum"),
            donor_ids=("donor_id", lambda ids: ",".join(sorted(set(map(str, ids))))),
        )
        .reset_index()
        .rename(columns={"structure_name": "brain_area"})
        .sort_values("brain_area")
    )
    return region_donor_counts


def write_matrix(matrix: pd.DataFrame, output_path: Path) -> None:
    print(f"Writing {matrix.shape[0]:,} x {matrix.shape[1]:,} matrix to {output_path}")
    matrix.to_csv(output_path, compression="gzip")


def outputs_exist(output_dir: Path) -> bool:
    return all(
        (output_dir / file_name).exists()
        for file_name in [
            AVERAGE_EXPRESSION_FILE,
            RANK_ZSCORE_FILE,
            ALIASES_FILE,
            REGION_LOOKUP_FILE,
            REGION_DONOR_COUNTS_FILE,
            MANIFEST_FILE,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-fetal-dir", type=Path, default=RAW_FETAL_DIR)
    parser.add_argument("--annotation-file", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--probe-strategy",
        choices=["reannotator", "default"],
        default="reannotator",
        help="Use reannotated symbols by default so modern gene symbols resolve.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if outputs_exist(args.output_dir) and not args.force:
        print(f"Fetal HBA preprocessed files already exist in {args.output_dir}")
        print("Use --force to rebuild them.")
        return

    fetal_folders = find_fetal_folders(args.raw_fetal_dir)
    average_matrices = []
    rank_matrices = []

    for fetal_folder in fetal_folders:
        average_matrix, rank_matrix = process_donor(
            fetal_folder=fetal_folder,
            annotation_file=args.annotation_file,
            probe_strategy=args.probe_strategy,
        )
        average_matrices.append(average_matrix)
        rank_matrices.append(rank_matrix)
        gc.collect()

    fetal_average = average_donor_matrices(average_matrices)
    fetal_rank_zscore = average_donor_matrices(rank_matrices)

    write_matrix(fetal_average, args.output_dir / AVERAGE_EXPRESSION_FILE)
    write_matrix(fetal_rank_zscore, args.output_dir / RANK_ZSCORE_FILE)

    aliases = build_alias_table(fetal_folders, args.annotation_file, fetal_average.index)
    aliases.to_csv(args.output_dir / ALIASES_FILE, index=False, compression="gzip")
    print(f"Writing {aliases.shape[0]:,} symbol aliases to {args.output_dir / ALIASES_FILE}")

    region_lookup = build_region_lookup(fetal_folders)
    region_lookup.to_csv(args.output_dir / REGION_LOOKUP_FILE, index=False, compression="gzip")
    print(f"Writing {region_lookup.shape[0]:,} region rows to {args.output_dir / REGION_LOOKUP_FILE}")

    region_donor_counts = build_region_donor_counts(fetal_folders)
    region_donor_counts.to_csv(
        args.output_dir / REGION_DONOR_COUNTS_FILE,
        index=False,
        compression="gzip",
    )
    print(
        f"Writing {region_donor_counts.shape[0]:,} region donor-count rows to "
        f"{args.output_dir / REGION_DONOR_COUNTS_FILE}"
    )

    manifest = {
        "raw_fetal_dir": str(args.raw_fetal_dir),
        "annotation_file": str(args.annotation_file),
        "probe_strategy": args.probe_strategy,
        "donor_ids": [donor_id_from_folder(folder) for folder in fetal_folders],
        "average_expression_shape": list(fetal_average.shape),
        "rank_zscore_shape": list(fetal_rank_zscore.shape),
        "outputs": {
            "average_expression": AVERAGE_EXPRESSION_FILE,
            "rank_zscore_expression": RANK_ZSCORE_FILE,
            "gene_symbol_aliases": ALIASES_FILE,
            "region_lookup": REGION_LOOKUP_FILE,
            "region_donor_counts": REGION_DONOR_COUNTS_FILE,
        },
    }
    with open(args.output_dir / MANIFEST_FILE, "w") as outfile:
        json.dump(manifest, outfile, indent=2)
    print(f"Writing manifest to {args.output_dir / MANIFEST_FILE}")


if __name__ == "__main__":
    main()
