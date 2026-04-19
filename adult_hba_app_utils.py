"""Shared adult HBA helpers for preprocessing-compatible Streamlit analyses."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from matplotlib.colors import rgb2hex
from scipy.stats import mannwhitneyu, rankdata

try:
    from scipy.stats import false_discovery_control
except ImportError:  # pragma: no cover - for older SciPy installs.
    false_discovery_control = None


def parse_gene_text(text: str) -> list[str]:
    """Parse pasted gene symbols from common delimited gene lists."""
    genes = []
    seen = set()
    for token in re.split(r"[\s,;|]+", text or ""):
        gene = token.strip()
        if not gene:
            continue
        key = gene.upper()
        if key not in seen:
            genes.append(gene)
            seen.add(key)
    return genes


def strip_left_right(structure_name: str) -> str:
    fragments = []
    for fragment in str(structure_name).split(","):
        if fragment.strip() not in {"left", "right", "Left", "Right"}:
            fragments.append(fragment)
    return ",".join(fragments).strip()


def resolve_gene_symbols(
    input_genes: Iterable[str],
    available_symbols: Iterable[str],
    aliases: pd.DataFrame | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Resolve user input to symbols present in the preprocessed matrices."""
    available_by_normalized = {str(symbol).upper(): str(symbol) for symbol in available_symbols}

    alias_lookup: dict[str, pd.DataFrame] = {}
    if aliases is not None and not aliases.empty:
        aliases = aliases.copy()
        if "input_symbol_normalized" not in aliases.columns:
            aliases["input_symbol_normalized"] = aliases["input_symbol"].astype(str).str.upper()
        aliases = aliases[aliases["gene_symbol"].astype(str).str.upper().isin(available_by_normalized)]
        alias_lookup = {
            key: group
            for key, group in aliases.groupby("input_symbol_normalized", sort=False)
        }

    resolved: list[str] = []
    resolved_seen = set()
    detail_rows = []

    for raw_gene in input_genes:
        normalized = str(raw_gene).strip().upper()
        if not normalized:
            continue

        if normalized in available_by_normalized:
            symbol = available_by_normalized[normalized]
            if symbol not in resolved_seen:
                resolved.append(symbol)
                resolved_seen.add(symbol)
            detail_rows.append(
                {
                    "input": raw_gene,
                    "resolved_symbol": symbol,
                    "status": "exact",
                    "source": "matrix",
                }
            )
            continue

        alias_hits = alias_lookup.get(normalized)
        if alias_hits is not None and not alias_hits.empty:
            target_symbols = list(dict.fromkeys(alias_hits["gene_symbol"].astype(str)))
            for symbol in target_symbols:
                if symbol not in resolved_seen:
                    resolved.append(symbol)
                    resolved_seen.add(symbol)
            detail_rows.append(
                {
                    "input": raw_gene,
                    "resolved_symbol": ", ".join(target_symbols),
                    "status": "alias",
                    "source": ", ".join(sorted(set(alias_hits["source"].astype(str)))),
                }
            )
        else:
            detail_rows.append(
                {
                    "input": raw_gene,
                    "resolved_symbol": "",
                    "status": "missing",
                    "source": "",
                }
            )

    return resolved, pd.DataFrame(detail_rows)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction."""
    values = pvalues.astype(float).to_numpy()
    corrected = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return pd.Series(corrected, index=pvalues.index)

    finite_values = values[finite]
    if false_discovery_control is not None:
        corrected[finite] = false_discovery_control(finite_values, method="bh")
        return pd.Series(corrected, index=pvalues.index)

    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n_tests = ranked.size
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    finite_corrected = np.empty_like(finite_values)
    finite_corrected[order] = adjusted
    corrected[finite] = finite_corrected
    return pd.Series(corrected, index=pvalues.index)


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC equivalent to sklearn.metrics.roc_auc_score for binary labels."""
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")
    positive_rank_sum = ranks[labels].sum()
    u_statistic = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return float(u_statistic / (positive_count * negative_count))


def mannwhitney_pvalue(positive_scores: pd.Series, negative_scores: pd.Series) -> float:
    if positive_scores.empty or negative_scores.empty:
        return float("nan")
    result = mannwhitneyu(positive_scores, negative_scores, alternative="two-sided")
    return float(getattr(result, "pvalue", result[1]))


def rank_zscore_by_gene(exp_df: pd.DataFrame) -> pd.DataFrame:
    """Rank each region column, then z-score each gene row across regions."""
    ranked = exp_df.rank(axis=0, ascending=True, method="average")
    row_mean = ranked.mean(axis=1)
    row_std = ranked.std(axis=1, ddof=0).replace(0, np.nan)
    return ranked.sub(row_mean, axis=0).div(row_std, axis=0)


def generate_region_stats(exp_df: pd.DataFrame, gene_list: Iterable[str]) -> pd.DataFrame:
    """Calculate AUROC, Mann-Whitney p-values, and BH-FDR for every brain region."""
    gene_set = set(gene_list)
    labels = exp_df.index.isin(gene_set)
    if labels.sum() == 0:
        raise ValueError("None of the submitted genes were found in the expression matrix.")
    if labels.sum() == len(labels):
        raise ValueError("The gene list includes every gene in the expression matrix.")

    pvalues = {}
    auc_values = {}
    for brain_area in exp_df.columns:
        series = exp_df[brain_area].dropna()
        local_labels = series.index.isin(gene_set)
        positive_scores = series[local_labels]
        negative_scores = series[~local_labels]
        pvalues[brain_area] = mannwhitney_pvalue(positive_scores, negative_scores)
        auc_values[brain_area] = auc_from_scores(series.to_numpy(), local_labels)

    pvalues = pd.Series(pvalues, name="p")
    auc = pd.Series(auc_values, name="AUROC")
    table = pd.concat([auc, pvalues, benjamini_hochberg(pvalues).rename("pFDR")], axis=1)
    table.index.name = "brain_area"
    return table.sort_values("AUROC", ascending=False)


def single_gene_expression_table(expression: pd.Series) -> pd.DataFrame:
    table = expression.dropna().sort_values(ascending=False).rename("expression").reset_index()
    table.rename(columns={table.columns[0]: "brain_area"}, inplace=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    if len(table) > 1:
        table["percentile"] = 100 * (1 - (table["rank"] - 1) / (len(table) - 1))
    else:
        table["percentile"] = 100.0
    return table


def load_ontology(ontology_file: Path) -> dict:
    with open(ontology_file) as infile:
        ontology = json.load(infile)
    if isinstance(ontology, dict) and isinstance(ontology.get("msg"), list) and ontology["msg"]:
        return ontology["msg"][0]
    return ontology


def flatten_ontology(node: dict, nodes: dict[int, dict] | None = None) -> dict[int, dict]:
    if nodes is None:
        nodes = {}
    nodes[int(node["id"])] = node
    for child in node.get("children", []):
        flatten_ontology(child, nodes)
    return nodes


def get_svg_structure_ids(svg_text: str) -> list[int]:
    soup = BeautifulSoup(svg_text, "xml")
    structure_ids = []
    seen = set()
    for element in soup.find_all(attrs={"structure_id": True}):
        try:
            structure_id = int(element.attrs["structure_id"])
        except (TypeError, ValueError):
            continue
        if structure_id not in seen:
            structure_ids.append(structure_id)
            seen.add(structure_id)
    return structure_ids


def build_direct_structure_value_map(
    values_by_region: pd.Series,
    region_lookup: pd.DataFrame,
) -> dict[int, float]:
    lookup = region_lookup.copy()
    if "structure_name" not in lookup.columns:
        lookup["structure_name"] = lookup["allen_structure_name"].map(strip_left_right)

    value_map = values_by_region.dropna().to_dict()
    direct: dict[int, float] = {}
    for _, row in lookup.iterrows():
        region_name = row["structure_name"]
        if region_name not in value_map:
            continue
        try:
            structure_id = int(row["structure_id"])
        except (TypeError, ValueError):
            continue
        direct[structure_id] = float(value_map[region_name])
    return direct


def build_svg_structure_label_map(
    region_lookup: pd.DataFrame,
    ontology: dict | None,
    svg_text: str,
) -> dict[int, str]:
    """Return human-readable labels for structure IDs present in an Allen SVG."""
    lookup = region_lookup.copy()
    if "structure_name" not in lookup.columns:
        lookup["structure_name"] = lookup["allen_structure_name"].map(strip_left_right)

    direct_labels: dict[int, str] = {}
    for _, row in lookup.iterrows():
        try:
            structure_id = int(row["structure_id"])
        except (TypeError, ValueError):
            continue
        direct_labels[structure_id] = str(row["structure_name"])

    ontology_labels: dict[int, str] = {}
    if ontology is not None:
        for structure_id, node in flatten_ontology(ontology).items():
            label = node.get("name") or node.get("acronym")
            if label:
                ontology_labels[structure_id] = strip_left_right(str(label))

    labels = {}
    for structure_id in get_svg_structure_ids(svg_text):
        labels[structure_id] = (
            direct_labels.get(structure_id)
            or ontology_labels.get(structure_id)
            or f"Structure {structure_id}"
        )
    return labels


def map_region_values_to_svg_structures(
    values_by_region: pd.Series,
    region_lookup: pd.DataFrame,
    ontology: dict | None,
    svg_text: str,
) -> dict[int, float]:
    """Map brain-region values onto the structure IDs present in an Allen SVG."""
    direct_values = build_direct_structure_value_map(values_by_region, region_lookup)
    if ontology is None:
        return {
            structure_id: direct_values.get(structure_id, float("nan"))
            for structure_id in get_svg_structure_ids(svg_text)
        }

    nodes = flatten_ontology(ontology)
    children_by_id = {
        structure_id: [int(child["id"]) for child in node.get("children", [])]
        for structure_id, node in nodes.items()
    }
    parent_by_id = {
        structure_id: node.get("parent_structure_id")
        for structure_id, node in nodes.items()
    }

    @lru_cache(maxsize=None)
    def values_for_structure(structure_id: int) -> tuple[float, ...]:
        if structure_id in direct_values:
            return (direct_values[structure_id],)

        child_values = []
        for child_id in children_by_id.get(structure_id, []):
            child_values.extend(values_for_structure(child_id))
        if child_values:
            return tuple(child_values)

        parent_id = parent_by_id.get(structure_id)
        if parent_id is not None:
            try:
                parent_id = int(parent_id)
            except (TypeError, ValueError):
                parent_id = None
        if parent_id in direct_values:
            return (direct_values[parent_id],)
        return ()

    mapped = {}
    for structure_id in get_svg_structure_ids(svg_text):
        vals = values_for_structure(structure_id)
        mapped[structure_id] = float(np.mean(vals)) if vals else float("nan")
    return mapped


def make_normalizer(values: Iterable[float], mode: str) -> mpl.colors.Normalize:
    finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite_values.size == 0:
        return mpl.colors.Normalize(vmin=0, vmax=1)

    if mode == "auc":
        extreme = max(float(np.max(np.abs(finite_values - 0.5))), 0.001)
        return mpl.colors.Normalize(vmin=0.5 - extreme, vmax=0.5 + extreme)

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if math.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def color_svg_by_values(
    svg_text: str,
    structure_values: dict[int, float],
    mode: str,
    label: str | None = None,
    cmap_name: str | None = None,
    include_colorbar: bool = True,
    structure_labels: dict[int, str] | None = None,
) -> str:
    """Return SVG text with structure fill colors replaced by mapped values."""
    if cmap_name is None:
        cmap_name = "RdYlBu_r" if mode == "auc" else "viridis"

    normalizer = make_normalizer(structure_values.values(), mode=mode)
    color_map = plt.get_cmap(cmap_name)
    soup = BeautifulSoup(svg_text, "xml")

    for group in soup.find_all(attrs={"graphic_group_label": ["Atlas - Human Sulci", "Atlas - Human Hotspots"]}):
        group.decompose()

    for element in soup.find_all(attrs={"structure_id": True}):
        try:
            structure_id = int(element.attrs["structure_id"])
        except (TypeError, ValueError):
            continue

        value = structure_values.get(structure_id, float("nan"))
        fill = "#FFFFFF" if not np.isfinite(value) else rgb2hex(color_map(normalizer(value)))
        if structure_id == 9219:
            fill = "#FFFFFF"
        element.attrs["style"] = f"stroke:black;fill:{fill};pointer-events:all"
        add_svg_tooltip(
            soup,
            element,
            structure_labels.get(structure_id) if structure_labels else None,
            value,
            label or mode,
        )

    if include_colorbar:
        append_svg_colorbar(soup, structure_values.values(), mode, label or mode, cmap_name)

    ensure_svg_viewbox(soup)
    return str(soup)


def add_svg_tooltip(
    soup: BeautifulSoup,
    element,
    structure_label: str | None,
    value: float,
    value_label: str,
) -> None:
    existing_title = element.find("title", recursive=False)
    if existing_title is not None:
        existing_title.decompose()

    region_text = structure_label or "Unknown brain region"
    value_text = f"{value:.3f}" if np.isfinite(value) else "not available"
    tooltip_text = f"{region_text}, {value_label}: {value_text}"
    element.attrs["data-hba-tooltip"] = tooltip_text
    element.attrs["data-hba-region"] = region_text
    element.attrs["data-hba-value-label"] = value_label
    element.attrs["data-hba-value"] = value_text
    element.attrs["aria-label"] = tooltip_text
    element.attrs["tabindex"] = "0"


def parse_svg_dimension(value: str | int | float | None, fallback: float) -> float:
    if value is None:
        return fallback
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else fallback


def format_svg_dimension(value: float) -> str:
    return f"{value:.6g}"


def ensure_svg_viewbox(soup: BeautifulSoup) -> None:
    svg = soup.find("svg")
    if svg is None or svg.get("viewBox"):
        return

    width = parse_svg_dimension(svg.get("width"), 800)
    height = parse_svg_dimension(svg.get("height"), 260)
    svg["viewBox"] = f"0 0 {format_svg_dimension(width)} {format_svg_dimension(height)}"


def append_svg_colorbar(
    soup: BeautifulSoup,
    values: Iterable[float],
    mode: str,
    label: str,
    cmap_name: str,
) -> None:
    svg = soup.find("svg")
    if svg is None:
        return

    original_width = parse_svg_dimension(svg.get("width"), 800)
    original_height = parse_svg_dimension(svg.get("height"), 260)
    extra_width = 145
    new_width = original_width + extra_width
    bar_width = 24
    bar_height = 150
    vertical_margin = 20
    new_height = max(original_height, bar_height + 2 * vertical_margin)
    svg["width"] = format_svg_dimension(new_width)
    svg["height"] = format_svg_dimension(new_height)
    svg["viewBox"] = (
        f"0 0 {format_svg_dimension(new_width)} {format_svg_dimension(new_height)}"
    )

    normalizer = make_normalizer(values, mode=mode)
    color_map = plt.get_cmap(cmap_name)

    bar_x = original_width + 38
    bar_y = (new_height - bar_height) / 2
    steps = 36
    segment_height = bar_height / steps

    group = soup.new_tag(
        "g",
        attrs={
            "id": "hba-inline-colorbar",
            "font-family": "Arial, sans-serif",
            "font-size": "11",
            "fill": "#111827",
        },
    )

    for i in range(steps):
        frac = i / (steps - 1)
        value = normalizer.vmin + frac * (normalizer.vmax - normalizer.vmin)
        rect = soup.new_tag(
            "rect",
            attrs={
                "x": f"{bar_x:.2f}",
                "y": f"{bar_y + (steps - i - 1) * segment_height:.2f}",
                "width": f"{bar_width:.2f}",
                "height": f"{segment_height + 0.15:.2f}",
                "fill": rgb2hex(color_map(normalizer(value))),
                "stroke": "none",
            },
        )
        group.append(rect)

    outline = soup.new_tag(
        "rect",
        attrs={
            "x": f"{bar_x:.2f}",
            "y": f"{bar_y:.2f}",
            "width": f"{bar_width:.2f}",
            "height": f"{bar_height:.2f}",
            "fill": "none",
            "stroke": "#111827",
            "stroke-width": "0.5",
        },
    )
    group.append(outline)

    tick_values = [normalizer.vmax, (normalizer.vmin + normalizer.vmax) / 2, normalizer.vmin]
    tick_fracs = [1, 0.5, 0]
    for tick_value, frac in zip(tick_values, tick_fracs):
        y = bar_y + (1 - frac) * bar_height
        tick = soup.new_tag(
            "line",
            attrs={
                "x1": f"{bar_x + bar_width:.2f}",
                "x2": f"{bar_x + bar_width + 5:.2f}",
                "y1": f"{y:.2f}",
                "y2": f"{y:.2f}",
                "stroke": "#111827",
                "stroke-width": "0.5",
            },
        )
        group.append(tick)

        text = soup.new_tag(
            "text",
            attrs={
                "x": f"{bar_x + bar_width + 8:.2f}",
                "y": f"{y + 3.7:.2f}",
            },
        )
        text.string = f"{tick_value:.2f}"
        group.append(text)

    label_text = soup.new_tag(
        "text",
        attrs={
            "x": f"{bar_x - 13:.2f}",
            "y": f"{bar_y + bar_height / 2:.2f}",
            "text-anchor": "middle",
            "transform": f"rotate(-90 {bar_x - 13:.2f} {bar_y + bar_height / 2:.2f})",
        },
    )
    label_text.string = label
    group.append(label_text)
    svg.append(group)


def make_colorbar(
    values: Iterable[float],
    mode: str,
    label: str,
    cmap_name: str | None = None,
) -> plt.Figure:
    if cmap_name is None:
        cmap_name = "RdYlBu_r" if mode == "auc" else "viridis"
    normalizer = make_normalizer(values, mode=mode)
    fig, ax = plt.subplots(figsize=(0.55, 1.65))
    colorbar = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap_name), norm=normalizer)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    fig.tight_layout(pad=0.2)
    return fig


def wrap_svg_for_html(svg_text: str, max_width_px: int = 1100) -> str:
    return f"""
    <style>
      .hba-svg-wrapper {{
        width: 100%;
        overflow-x: auto;
        margin: 0 0 0.25rem 0;
      }}
      .hba-svg-wrapper svg {{
        display: block;
        width: min({max_width_px}px, 100%);
        height: auto;
        margin: 0 auto;
      }}
    </style>
    <div class="hba-svg-wrapper">
        {svg_text}
    </div>
    """
