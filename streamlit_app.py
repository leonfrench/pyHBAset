from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

import adult_hba_app_utils as hba_app


DATA_DIR = Path("data/processed")
ADULT_ONTOLOGY_PATH = Path("data/ontology.json")
ALLEN_HBA_URL = "https://human.brain-map.org/"


@dataclass(frozen=True)
class SvgConfig:
    label: str
    path: Path
    max_width_px: int
    iframe_height: int
    ontology_path: Path | None = None


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    matrix_label: str
    average_expression_path: Path
    rank_zscore_path: Path
    aliases_path: Path
    region_lookup_path: Path
    region_donor_counts_path: Path
    preprocess_command: str
    svg_configs: tuple[SvgConfig, ...]


DATASETS = {
    "adult": DatasetConfig(
        key="adult",
        label="Adult",
        matrix_label="adult Allen Human Brain Atlas",
        average_expression_path=DATA_DIR / "adult_hba_average_expression.csv.gz",
        rank_zscore_path=DATA_DIR / "adult_hba_rank_zscore_expression.csv.gz",
        aliases_path=DATA_DIR / "adult_hba_gene_symbol_aliases.csv.gz",
        region_lookup_path=DATA_DIR / "adult_hba_region_lookup.csv.gz",
        region_donor_counts_path=DATA_DIR / "adult_hba_region_donor_counts.csv.gz",
        preprocess_command="python3 preprocess_adult_hba.py",
        svg_configs=(
            SvgConfig(
                label="adult overview",
                path=Path("data/svg/human_diagram.svg"),
                max_width_px=1100,
                iframe_height=315,
                ontology_path=ADULT_ONTOLOGY_PATH,
            ),
        ),
    ),
    "fetal": DatasetConfig(
        key="fetal",
        label="Fetal",
        matrix_label="fetal Allen Human Brain Atlas",
        average_expression_path=DATA_DIR / "fetal_hba_average_expression.csv.gz",
        rank_zscore_path=DATA_DIR / "fetal_hba_rank_zscore_expression.csv.gz",
        aliases_path=DATA_DIR / "fetal_hba_gene_symbol_aliases.csv.gz",
        region_lookup_path=DATA_DIR / "fetal_hba_region_lookup.csv.gz",
        region_donor_counts_path=DATA_DIR / "fetal_hba_region_donor_counts.csv.gz",
        preprocess_command="python3 preprocess_fetal_hba.py",
        svg_configs=(
            SvgConfig(
                label="fetal 21pcw slice 0893",
                path=Path("data/svg/slices/fetal21/0893_101892619.svg"),
                max_width_px=195,
                iframe_height=530,
                ontology_path=Path("data/fetal21ontology.json"),
            ),
            SvgConfig(
                label="fetal 21pcw slice 1097",
                path=Path("data/svg/slices/fetal21/1097_101892615.svg"),
                max_width_px=215,
                iframe_height=530,
                ontology_path=Path("data/fetal21ontology.json"),
            ),
            SvgConfig(
                label="fetal 21pcw slice 1352",
                path=Path("data/svg/slices/fetal21/1352_101892610.svg"),
                max_width_px=185,
                iframe_height=530,
                ontology_path=Path("data/fetal21ontology.json"),
            ),
            SvgConfig(
                label="fetal brainstem slice 0391",
                path=Path("data/svg/slices/fetal21_brainstem/0391_102182817.svg"),
                max_width_px=205,
                iframe_height=390,
                ontology_path=Path("data/fetal_brainstem_ontology.json"),
            ),
            SvgConfig(
                label="fetal brainstem slice 0639",
                path=Path("data/svg/slices/fetal21_brainstem/0639_102182810.svg"),
                max_width_px=520,
                iframe_height=450,
                ontology_path=Path("data/fetal_brainstem_ontology.json"),
            ),
        ),
    ),
}


st.set_page_config(
    page_title="Human brain region gene expression enrichment results",
    layout="wide",
)


def apply_layout_css() -> None:
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] {
            width: 17rem !important;
            min-width: 17rem !important;
            max-width: 17rem !important;
          }
          section[data-testid="stSidebar"] > div {
            width: 17rem !important;
            min-width: 17rem !important;
            max-width: 17rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_matrix(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False)
def load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_text(path: str) -> str:
    return Path(path).read_text()


@st.cache_data(show_spinner=False)
def load_ontology(path: str) -> dict:
    return hba_app.load_ontology(Path(path))


def ontology_paths_for_config(config: DatasetConfig) -> tuple[str, ...]:
    paths = []
    seen = set()
    for svg_config in config.svg_configs:
        if svg_config.ontology_path is None:
            continue
        path = str(svg_config.ontology_path)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def coerce_structure_id(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def build_region_parent_table(region_lookup_path: str, ontology_paths: tuple[str, ...]) -> pd.DataFrame:
    region_lookup = pd.read_csv(region_lookup_path)
    lookup = region_lookup.copy()
    if "structure_name" not in lookup.columns:
        lookup["structure_name"] = lookup["allen_structure_name"].map(hba_app.strip_left_right)

    ontology_nodes = {}
    for ontology_path in ontology_paths:
        path = Path(ontology_path)
        if path.exists():
            ontology_nodes.update(hba_app.flatten_ontology(hba_app.load_ontology(path)))

    id_to_label: dict[int, str] = {}
    parent_by_id: dict[int, int | None] = {}
    for structure_id, node in ontology_nodes.items():
        label = node.get("name") or node.get("acronym")
        if label:
            id_to_label[structure_id] = hba_app.strip_left_right(str(label))
        parent_by_id[structure_id] = coerce_structure_id(node.get("parent_structure_id"))

    for _, row in lookup.iterrows():
        structure_id = coerce_structure_id(row.get("structure_id"))
        if structure_id is None:
            continue
        id_to_label[structure_id] = str(row["structure_name"])
        if "parent_structure_id" in lookup.columns:
            parent_by_id[structure_id] = coerce_structure_id(row.get("parent_structure_id"))

    def first_distinct_parent(structure_id: int | None, region_name: str) -> str | None:
        region_key = hba_app.strip_left_right(region_name).casefold()
        parent_id = parent_by_id.get(structure_id) if structure_id is not None else None
        seen = set()
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            parent_label = id_to_label.get(parent_id)
            parent_key = hba_app.strip_left_right(parent_label).casefold() if parent_label else ""
            if parent_key and parent_key != region_key and len(parent_key) > 5:
                return parent_label
            parent_id = parent_by_id.get(parent_id)
        return None

    rows = []
    for _, row in lookup.iterrows():
        structure_id = coerce_structure_id(row.get("structure_id"))
        region_name = str(row["structure_name"])
        parent_region = first_distinct_parent(structure_id, region_name)
        if parent_region:
            rows.append({"brain_area": region_name, "Parent region": parent_region})

    if not rows:
        return pd.DataFrame(columns=["brain_area", "Parent region"])

    parent_table = pd.DataFrame(rows)
    return (
        parent_table.groupby("brain_area", sort=False)["Parent region"]
        .agg(lambda values: "; ".join(dict.fromkeys(values)))
        .reset_index()
    )


def check_preprocessed_files(config: DatasetConfig) -> None:
    required_paths = [
        config.average_expression_path,
        config.rank_zscore_path,
        config.aliases_path,
        config.region_lookup_path,
        config.region_donor_counts_path,
    ]
    for svg_config in config.svg_configs:
        required_paths.append(svg_config.path)
        if svg_config.ontology_path is not None:
            required_paths.append(svg_config.ontology_path)

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        st.error(f"{config.label} Allen Human Brain Atlas app files are missing.")
        st.write("Run preprocessing from the repository root:")
        st.code(config.preprocess_command, language="bash")
        st.write("Missing files:")
        for path in missing:
            st.write(f"- `{path}`")
        st.stop()


def show_mapping_details(
    mapping: pd.DataFrame,
    *,
    title: str = "Gene resolution details",
    missing_label: str = "input gene(s)",
    matrix_label: str = "selected Allen Human Brain Atlas",
) -> None:
    if mapping.empty:
        return
    missing = mapping[mapping["status"] == "missing"]
    aliases = mapping[mapping["status"] == "alias"]

    if not aliases.empty:
        st.info("Some inputs were resolved through the Allen original-symbol alias table.")
    if not missing.empty:
        st.warning(
            f"{missing.shape[0]} {missing_label} were not found in the "
            f"{matrix_label} matrix."
        )

    status_order = {"missing": 0, "alias": 1, "exact": 2}
    display_mapping = mapping.copy()
    display_mapping["_status_order"] = display_mapping["status"].map(status_order).fillna(3)
    display_mapping["_input_order"] = range(len(display_mapping))
    display_mapping = (
        display_mapping.sort_values(["_status_order", "_input_order"])
        .drop(columns=["_status_order", "_input_order"])
    )

    with st.expander(title, expanded=False):
        st.dataframe(display_mapping, width="stretch", hide_index=True)


def add_region_coverage(
    table: pd.DataFrame,
    region_donor_counts: pd.DataFrame,
    region_parent_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    coverage_cols = ["brain_area", "donor_count", "sample_count"]
    coverage = region_donor_counts[coverage_cols].copy()
    coverage.rename(columns={"donor_count": "Donors", "sample_count": "Samples"}, inplace=True)
    table = table.merge(coverage, on="brain_area", how="left")
    if region_parent_table is not None and not region_parent_table.empty:
        table = table.merge(region_parent_table, on="brain_area", how="left")

    front_cols = [
        col for col in ["brain_area", "Parent region", "Donors", "Samples"] if col in table.columns
    ]
    remaining_cols = [col for col in table.columns if col not in front_cols]
    table = table[front_cols + remaining_cols]
    return table.rename(columns={"brain_area": "Brain region"})


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def format_probability(value: float) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if value > 0.0001:
        return f"{value:.3g}"
    return f"{value:.2e}"


def wrap_svg_for_display(
    svg_text: str,
    max_width_px: int,
    align: str = "center",
    caption: str | None = None,
) -> str:
    svg_margin = {
        "left": "0 auto 0 0",
        "right": "0 0 0 auto",
    }.get(align, "0 auto")
    caption_html = (
        f'<div class="hba-svg-label">{html.escape(caption)}</div>'
        if caption is not None
        else ""
    )

    return f"""
    <style>
      .hba-svg-wrapper {{
        width: 100%;
        overflow-x: auto;
        margin: 0 0 0.25rem 0;
        position: relative;
      }}
      .hba-svg-inner {{
        width: min({max_width_px}px, 100%);
        margin: {svg_margin};
      }}
      .hba-svg-label {{
        color: #6b7280;
        font: 1rem Arial, sans-serif;
        line-height: 1.25;
        margin: 0 0 0.75rem 0;
        text-align: center;
      }}
      .hba-svg-wrapper svg {{
        display: block;
        width: 100%;
        height: auto;
        margin: 0;
      }}
      .hba-svg-tooltip {{
        background: #111827;
        border: 1px solid #374151;
        border-radius: 4px;
        box-shadow: 0 8px 18px rgba(17, 24, 39, 0.18);
        color: #ffffff;
        display: none;
        font: 0.82rem Arial, sans-serif;
        left: 0;
        line-height: 1.35;
        max-width: 260px;
        padding: 0.35rem 0.5rem;
        pointer-events: none;
        position: absolute;
        top: 0;
        white-space: pre-line;
        z-index: 20;
      }}
    </style>
    <div class="hba-svg-wrapper">
      <div class="hba-svg-inner">
        {caption_html}
        {svg_text}
      </div>
      <div class="hba-svg-tooltip"></div>
    </div>
    <script>
      (() => {{
        const script = document.currentScript;
        const wrapper = script.previousElementSibling;
        const tooltip = wrapper.querySelector(".hba-svg-tooltip");

        function hideTooltip() {{
          tooltip.style.display = "none";
        }}

        wrapper.addEventListener("mousemove", (event) => {{
          const targetElement = event.target instanceof Element ? event.target : event.target.parentElement;
          const target = targetElement ? targetElement.closest("[data-hba-tooltip]") : null;
          if (!target || !wrapper.contains(target)) {{
            hideTooltip();
            return;
          }}

          const region = target.getAttribute("data-hba-region");
          const valueLabel = target.getAttribute("data-hba-value-label");
          const value = target.getAttribute("data-hba-value");
          tooltip.textContent = region && valueLabel && value
            ? `${{region}}, ${{valueLabel}}: ${{value}}`
            : target.getAttribute("data-hba-tooltip");
          tooltip.style.display = "block";

          const wrapperRect = wrapper.getBoundingClientRect();
          const tooltipRect = tooltip.getBoundingClientRect();
          const padding = 8;
          let left = event.clientX - wrapperRect.left + wrapper.scrollLeft + 14;
          let top = event.clientY - wrapperRect.top + wrapper.scrollTop + 14;
          const maxLeft = wrapper.clientWidth + wrapper.scrollLeft - tooltipRect.width - padding;
          const maxTop = wrapper.clientHeight + wrapper.scrollTop - tooltipRect.height - padding;
          left = Math.max(padding + wrapper.scrollLeft, Math.min(left, maxLeft));
          top = Math.max(padding + wrapper.scrollTop, Math.min(top, maxTop));
          tooltip.style.left = `${{left}}px`;
          tooltip.style.top = `${{top}}px`;
        }});

        wrapper.addEventListener("mouseleave", hideTooltip);
      }})();
    </script>
    """


def read_svg_parts(svg_text: str) -> tuple[float, float, str, str]:
    soup = BeautifulSoup(svg_text, "xml")
    svg = soup.find("svg")
    if svg is None:
        return 800.0, 260.0, "0 0 800 260", svg_text

    width = hba_app.parse_svg_dimension(svg.get("width"), 800)
    height = hba_app.parse_svg_dimension(svg.get("height"), 260)
    view_box = svg.get("viewBox") or f"0 0 {width:.6g} {height:.6g}"
    return width, height, view_box, svg.decode_contents()


def combine_colored_svgs(colored_svgs: list[tuple[SvgConfig, str]]) -> str:
    label_height = 28
    gap = 44
    row_gap = 36
    margin = 24

    panel_rows = [colored_svgs[:3], colored_svgs[3:]] if len(colored_svgs) == 5 else [colored_svgs]
    prepared_rows = []
    canvas_width = 0.0

    for row in panel_rows:
        prepared_row = []
        for svg_config, colored_svg in row:
            source_width, source_height, view_box, contents = read_svg_parts(colored_svg)
            display_width = float(svg_config.max_width_px)
            display_height = display_width * source_height / source_width
            prepared_row.append(
                {
                    "label": svg_config.label,
                    "display_width": display_width,
                    "display_height": display_height,
                    "view_box": view_box,
                    "contents": contents,
                }
            )
        row_width = sum(panel["display_width"] for panel in prepared_row)
        row_width += gap * max(0, len(prepared_row) - 1)
        canvas_width = max(canvas_width, row_width)
        prepared_rows.append((prepared_row, row_width))

    row_heights = [
        label_height + max(panel["display_height"] for panel in prepared_row)
        for prepared_row, _ in prepared_rows
    ]
    canvas_height = sum(row_heights) + row_gap * max(0, len(row_heights) - 1)
    canvas_width += 2 * margin
    canvas_height += 2 * margin

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_width:.1f}" height="{canvas_height:.1f}" '
            f'viewBox="0 0 {canvas_width:.1f} {canvas_height:.1f}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            "<style>"
            ".hba-panel-label{font:16px Arial,sans-serif;fill:#6b7280;}"
            "</style>"
        ),
    ]

    y = margin
    for (prepared_row, row_width), row_height in zip(prepared_rows, row_heights):
        x = margin + (canvas_width - 2 * margin - row_width) / 2
        for panel in prepared_row:
            label_x = x + panel["display_width"] / 2
            parts.append(
                f'<text class="hba-panel-label" x="{label_x:.1f}" y="{y + 17:.1f}" '
                f'text-anchor="middle">{html.escape(panel["label"])}</text>'
            )
            parts.append(
                f'<svg x="{x:.1f}" y="{y + label_height:.1f}" '
                f'width="{panel["display_width"]:.1f}" '
                f'height="{panel["display_height"]:.1f}" '
                f'viewBox="{html.escape(panel["view_box"])}" '
                'preserveAspectRatio="xMidYMid meet">'
            )
            parts.append(panel["contents"])
            parts.append("</svg>")
            x += panel["display_width"] + gap
        y += row_height + row_gap

    parts.append("</svg>")
    return "\n".join(parts)


def add_colored_svg_download(
    colored_svgs: list[tuple[SvgConfig, str]],
    file_name: str,
) -> None:
    if len(colored_svgs) == 1:
        button_label = "Download colored SVG"
        svg_text = colored_svgs[0][1]
    else:
        button_label = "Download combined SVG"
        svg_text = combine_colored_svgs(colored_svgs)

    st.download_button(
        button_label,
        data=svg_text.encode("utf-8"),
        file_name=file_name,
        mime="image/svg+xml",
    )


def color_and_show_svgs(
    values_by_region: pd.Series,
    mode: str,
    label: str,
    config: DatasetConfig,
) -> list[tuple[SvgConfig, str]]:
    region_lookup = load_table(str(config.region_lookup_path))
    mapped_svgs = []
    all_structure_values = []

    for svg_config in config.svg_configs:
        svg_text = load_text(str(svg_config.path))
        ontology = (
            load_ontology(str(svg_config.ontology_path))
            if svg_config.ontology_path is not None and svg_config.ontology_path.exists()
            else None
        )

        structure_values = hba_app.map_region_values_to_svg_structures(
            values_by_region=values_by_region,
            region_lookup=region_lookup,
            ontology=ontology,
            svg_text=svg_text,
        )
        structure_labels = hba_app.build_svg_structure_label_map(
            region_lookup=region_lookup,
            ontology=ontology,
            svg_text=svg_text,
        )
        mapped_svgs.append((svg_config, svg_text, structure_values, structure_labels))
        all_structure_values.extend(structure_values.values())

    shared_scale_values = {
        -(index + 1): value for index, value in enumerate(all_structure_values)
    }
    colored_svgs = []

    for index, (svg_config, svg_text, structure_values, structure_labels) in enumerate(mapped_svgs):
        colored_svg = hba_app.color_svg_by_values(
            svg_text,
            {**shared_scale_values, **structure_values},
            mode=mode,
            label=label,
            include_colorbar=index == len(mapped_svgs) - 1,
            structure_labels=structure_labels,
        )
        colored_svgs.append((svg_config, colored_svg))

    def render_svg(svg_config: SvgConfig, colored_svg: str, align: str = "center") -> None:
        components.html(
            wrap_svg_for_display(
                colored_svg,
                max_width_px=svg_config.max_width_px,
                align=align,
                caption=svg_config.label,
            ),
            height=svg_config.iframe_height,
            scrolling=False,
        )

    if len(colored_svgs) == 5:
        top_columns = st.columns(3, gap="small")
        for column, (svg_config, colored_svg) in zip(top_columns, colored_svgs[:3]):
            with column:
                render_svg(svg_config, colored_svg)

        bottom_columns = st.columns(2, gap="small")
        for column, (svg_config, colored_svg), align in zip(
            bottom_columns,
            colored_svgs[3:],
            ["right", "left"],
        ):
            with column:
                render_svg(svg_config, colored_svg, align=align)
    elif len(colored_svgs) > 1:
        columns = st.columns(len(colored_svgs), gap="small")
        for column, (svg_config, colored_svg) in zip(columns, colored_svgs):
            with column:
                render_svg(svg_config, colored_svg)
    else:
        svg_config, colored_svg = colored_svgs[0]
        components.html(
            wrap_svg_for_display(colored_svg, max_width_px=svg_config.max_width_px),
            height=svg_config.iframe_height,
            scrolling=False,
        )

    return colored_svgs


def single_gene_view(
    gene: str,
    average_expression: pd.DataFrame,
    region_donor_counts: pd.DataFrame,
    region_parent_table: pd.DataFrame,
    config: DatasetConfig,
) -> None:
    st.subheader(f"{gene} average expression ({config.label.lower()})")
    expression = average_expression.loc[gene].dropna()
    table = hba_app.single_gene_expression_table(expression)
    table = add_region_coverage(table, region_donor_counts, region_parent_table)
    table.drop(columns=["rank"], inplace=True)

    top_expression = table.iloc[0]["expression"]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Brain regions", f"{table.shape[0]:,}")
    metric_cols[1].metric("Max expression", f"{top_expression:.3f}")
    metric_cols[2].metric("Median expression", f"{table['expression'].median():.3f}")

    colored_svgs = color_and_show_svgs(
        values_by_region=expression,
        mode="expression",
        label="Expression log2 intensity",
        config=config,
    )

    add_colored_svg_download(
        colored_svgs,
        file_name=f"{gene}_{config.key}_hba_expression.svg",
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Brain region": st.column_config.TextColumn("Brain region", width="large"),
            "Parent region": st.column_config.TextColumn("Parent region", width="medium"),
            "Donors": st.column_config.NumberColumn("Donors", format="%d", width="small"),
            "Samples": st.column_config.NumberColumn("Samples", format="%d", width="small"),
            "expression": st.column_config.NumberColumn(format="%.4f"),
            "percentile": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.download_button(
        "Download expression table",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{gene}_{config.key}_hba_expression.csv",
        mime="text/csv",
    )


def multi_gene_view(
    genes: list[str],
    average_expression: pd.DataFrame,
    region_donor_counts: pd.DataFrame,
    region_parent_table: pd.DataFrame,
    config: DatasetConfig,
    background_genes: list[str] | None = None,
) -> None:
    st.subheader("Human brain region gene expression enrichment results")
    using_background = background_genes is not None

    if using_background:
        universe_genes = [gene for gene in background_genes if gene in average_expression.index]
        universe_set = set(universe_genes)
        genes_in_matrix = [gene for gene in genes if gene in universe_set]
        excluded_targets = [gene for gene in genes if gene not in universe_set]

        if excluded_targets:
            st.warning(
                f"{len(excluded_targets):,} target gene(s) were resolved in the "
                f"{config.matrix_label} matrix but are not in the background universe, "
                "so they were excluded from AUROC scoring."
            )
        if len(universe_genes) < 2:
            st.error(
                f"Background genes must resolve to at least two {config.matrix_label} "
                "matrix genes."
            )
            st.stop()
        if not genes_in_matrix:
            st.error("None of the resolved target genes are present in the background universe.")
            st.stop()
        if len(genes_in_matrix) == len(universe_genes):
            st.error("Background genes must include at least one non-target gene for AUROC scoring.")
            st.stop()

        score_matrix = hba_app.rank_zscore_by_gene(average_expression.loc[universe_genes])
        download_prefix = f"{config.key}_hba_background_limited_gene_list_auroc"
    else:
        score_matrix = load_matrix(str(config.rank_zscore_path))
        universe_genes = list(score_matrix.index)
        genes_in_matrix = [gene for gene in genes if gene in score_matrix.index]
        download_prefix = f"{config.key}_hba_gene_list_auroc"

    stats = hba_app.generate_region_stats(score_matrix, genes_in_matrix)
    stats_table = stats.reset_index()
    stats_table = add_region_coverage(stats_table, region_donor_counts, region_parent_table)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Target genes tested", f"{len(genes_in_matrix):,}")
    metric_cols[1].metric("Universe genes", f"{len(universe_genes):,}")
    metric_cols[2].metric("Brain regions", f"{stats_table.shape[0]:,}")
    metric_cols[3].metric("Top AUROC", f"{stats_table['AUROC'].max():.3f}")
    metric_cols[4].metric("FDR < 0.05", f"{(stats_table['pFDR'] < 0.05).sum():,}")

    if using_background:
        st.caption(
            "Background provided: AUROC uses only the resolved background universe, with each "
            "region column re-ranked and each gene row z-scored after that reduction."
        )

    colored_svgs = color_and_show_svgs(
        values_by_region=stats["AUROC"],
        mode="auc",
        label="AUROC",
        config=config,
    )

    add_colored_svg_download(
        colored_svgs,
        file_name=f"{download_prefix}.svg",
    )

    display_stats_table = stats_table.copy()
    display_stats_table["p"] = display_stats_table["p"].map(format_probability)
    display_stats_table["pFDR"] = display_stats_table["pFDR"].map(format_probability)

    st.dataframe(
        display_stats_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Brain region": st.column_config.TextColumn("Brain region", width="large"),
            "Parent region": st.column_config.TextColumn("Parent region", width="medium"),
            "Donors": st.column_config.NumberColumn("Donors", format="%d", width="small"),
            "Samples": st.column_config.NumberColumn("Samples", format="%d", width="small"),
            "AUROC": st.column_config.NumberColumn(format="%.4f"),
            "p": st.column_config.TextColumn("p"),
            "pFDR": st.column_config.TextColumn("pFDR"),
        },
    )
    st.download_button(
        "Download AUROC table",
        data=stats_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{download_prefix}.csv",
        mime="text/csv",
    )


def main() -> None:
    apply_layout_css()

    with st.sidebar:
        st.markdown(
            "<p style='font-size: 1.05rem; line-height: 1.3;'>"
            "<strong>HBAset</strong>: a tool for testing brain region specific expression"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<p style='font-size: 0.85rem; line-height: 1.3; color: #6b7280;'>"
                "Data are from the Allen Institute for Brain Science "
                f"<a href='{ALLEN_HBA_URL}' target='_blank'>Human Brain Atlas microarray resource</a>."
                "</p>"
            ),
            unsafe_allow_html=True,
        )
        dataset_key = st.selectbox(
            "Dataset",
            options=list(DATASETS.keys()),
            format_func=lambda key: DATASETS[key].label,
            index=0,
        )

    config = DATASETS[dataset_key]
    check_preprocessed_files(config)

    average_expression = load_matrix(str(config.average_expression_path))
    aliases = load_table(str(config.aliases_path))
    region_donor_counts = load_table(str(config.region_donor_counts_path))
    region_parent_table = build_region_parent_table(
        str(config.region_lookup_path),
        ontology_paths_for_config(config),
    )

    with st.sidebar:
        st.header("Input")
        default_gene = "NGFR"
        pasted_target_genes = st.text_area(
            "Target genes",
            value=default_gene,
            height=180,
            help="Use spaces, tabs, commas, semicolons, pipes, or one gene per line.",
        )
        pasted_background_genes = st.text_area(
            "Background genes",
            value="",
            height=180,
            help=(
                "Optional. Use spaces, tabs, commas, semicolons, pipes, "
                "or one gene per line."
            ),
        )
        st.caption(
            f"One resolved gene shows average expression from the {config.matrix_label}. "
            f"Two or more resolved genes run AUROC against the {config.label.lower()} "
            "Allen rank-z-scored matrix. "
            "A background list limits and re-ranks the AUROC universe."
        )
        st.markdown(
            "Source code is on [GitHub](https://github.com/leonfrench/pyHBAset)"
        )
        st.header("Dataset")
        st.write(f"Dataset: `{config.label}`")
        st.write(f"Average matrix: `{average_expression.shape[0]:,}` genes x `{average_expression.shape[1]:,}` regions")

    target_input_genes = hba_app.parse_gene_text(pasted_target_genes)
    background_input_genes = hba_app.parse_gene_text(pasted_background_genes)
    if not target_input_genes:
        st.info("Enter a gene symbol or paste a gene list to begin.")
        st.stop()

    resolved_genes, mapping = hba_app.resolve_gene_symbols(
        target_input_genes,
        average_expression.index,
        aliases,
    )
    show_mapping_details(
        mapping,
        title="Target gene resolution details",
        missing_label="target gene(s)",
        matrix_label=config.matrix_label,
    )

    if not resolved_genes:
        st.stop()

    resolved_background_genes: list[str] | None = None
    if background_input_genes:
        resolved_background_genes, background_mapping = hba_app.resolve_gene_symbols(
            background_input_genes,
            average_expression.index,
            aliases,
        )
        show_mapping_details(
            background_mapping,
            title="Background gene resolution details",
            missing_label="background gene(s)",
            matrix_label=config.matrix_label,
        )
        st.info(
            f"Background universe: {len(resolved_background_genes):,} {config.matrix_label} gene(s) "
            f"resolved from {len(background_input_genes):,} submitted background token(s)."
        )

    if len(resolved_genes) == 1:
        if background_input_genes:
            st.caption("Background genes are used for multi-gene AUROC scoring; single-gene expression is unchanged.")
        single_gene_view(
            resolved_genes[0],
            average_expression,
            region_donor_counts,
            region_parent_table,
            config,
        )
    else:
        multi_gene_view(
            resolved_genes,
            average_expression,
            region_donor_counts,
            region_parent_table,
            config,
            resolved_background_genes,
        )


if __name__ == "__main__":
    main()
