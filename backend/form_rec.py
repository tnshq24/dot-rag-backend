from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from tabulate import tabulate

__all__ = [
    "extract_text_and_tables",
    "text_formatting",
    "text_formatting_list",
    "save_content_to_txt",
]

###############################################################################
# Helper utilities
###############################################################################

def _first(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key among *keys* in *mapping* (or *default*)."""
    for k in keys:
        if k in mapping:
            return mapping[k]
    return default


def _clean(text: str) -> str:
    """Remove Form‑Recognizer selection markers and strip whitespace."""
    for mark in ("\n:unselected:", "\n:selected:", ":unselected:", ":selected:"):
        text = text.replace(mark, "")
    return text.strip()


@dataclass()
class CellInfo:
    row: int
    col: int
    row_span: int
    col_span: int
    content: str

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "CellInfo":
        return cls(
            row=_first(raw, "rowIndex", "row_index", default=0),
            col=_first(raw, "columnIndex", "column_index", default=0),
            row_span=_first(raw, "rowSpan", "row_span", default=1),
            col_span=_first(raw, "columnSpan", "column_span", default=1),
            content=_clean(raw.get("content", "")),
        )

###############################################################################
# Table helpers
###############################################################################

def _make_unique(columns: List[Any]) -> List[str]:
    counts = defaultdict(int)
    unique: List[str] = []
    for col in columns:
        col_str = str(col)
        counts[col_str] += 1
        unique.append(col_str + " " * counts[col_str])
    return unique


def _build_matrix(row_cnt: int, col_cnt: int, raw_cells: List[Dict[str, Any]]) -> Tuple[List[List[str]], List[str]]:
    matrix = [["" for _ in range(col_cnt)] for _ in range(row_cnt)]
    flat: List[str] = []
    for rc in raw_cells:
        cell = CellInfo.from_raw(rc)
        if cell.content:
            flat.append(cell.content)
        for r in range(cell.row_span):
            for c in range(cell.col_span):
                matrix[cell.row + r][cell.col + c] = cell.content
    return matrix, flat


def _extract_table(tbl: Dict[str, Any], *, as_: str):
    rows = _first(tbl, "rowCount", "row_count", default=0)
    cols = _first(tbl, "columnCount", "column_count", default=0)
    matrix, flat = _build_matrix(rows, cols, tbl.get("cells", []))

    header_rows = 0
    if any("kind" in c for c in tbl.get("cells", [])):
        header_rows = max((
            _first(c, "rowIndex", "row_index", default=0) + _first(c, "rowSpan", "row_span", default=1)
            for c in tbl["cells"] if c.get("kind") == "columnHeader"), default=0)

    matrix = [row for row in matrix if any(cell != "" for cell in row)]

    if header_rows:
        headers = [" : ".join(dict.fromkeys(col_vals)) for col_vals in zip(*matrix[:header_rows])]
        df = pd.DataFrame(matrix[header_rows:], columns=headers)
    else:
        df = pd.DataFrame(matrix)

    df.drop_duplicates(inplace=True)
    df.columns = _make_unique(list(df.columns))

    if as_ == "dataframe":
        return df, flat
    if as_ == "markdown":
        md = tabulate(df.to_records(index=False), headers=df.columns, tablefmt="github")
        return md, flat
    if as_ == "json":
        js = json.dumps({"fields": list(df.columns), "data": df.values.tolist()}, ensure_ascii=False)
        return js, flat
    raise ValueError("as_ must be 'dataframe', 'markdown', or 'json'.")

###############################################################################
# Page‑level grouping helpers
###############################################################################

def _tables_by_page(tables: List[Dict[str, Any]], *, as_: str):
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for t in tables:
        br = _first(t, "boundingRegions", "bounding_regions", default=[{}])
        page_no = _first(br[0], "pageNumber", "page_number", default=1) if br else 1
        content, flat = _extract_table(t, as_=as_)
        grouped.setdefault(page_no, []).append({"table_content": {"role": "table", "content": content}, "table_list": flat})
    return grouped


def _paragraphs_by_page(paragraphs: List[Dict[str, Any]]):
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in paragraphs:
        br = _first(p, "boundingRegions", "bounding_regions", default=[{}])
        page_no = _first(br[0], "pageNumber", "page_number", default=1) if br else 1
        by_page[page_no].append({"role": p.get("role", "paragraph"), "content": _clean(p.get("content", ""))})
    return by_page


def _normalise(items: List[Dict[str, Any]]):
    norm: List[Dict[str, str]] = []
    for it in items:
        if "content" in it and "role" in it:
            norm.append(it)
        elif "table_content" in it:
            norm.append(it["table_content"])
    return norm

###############################################################################
# Public API
###############################################################################

def extract_text_and_tables(analyse: Dict[str, Any], *, output: str = "markdown", include_tags: bool = False):
    """Return a page‑wise dict with `structured_content` & `unstructured_content`.

    Parameters
    ----------
    analyse : dict
        Raw JSON from Azure Document Intelligence.
    output  : {"markdown", "json", "paragraph"}
    include_tags : bool, default *False*
        If *True* prepend each block with `##TABLE##` / `##PARAGRAPH##` tags.
    """
    pages_total = len(analyse.get("pages", [])) or 1
    page_content: Dict[int, Dict[str, Any]] = {p: {"structured_content": [], "unstructured_content": ""} for p in range(1, pages_total + 1)}

    # 1) paragraphs
    for p_no, plist in _paragraphs_by_page(analyse.get("paragraphs", [])).items():
        page_content[p_no]["structured_content"].extend(plist)

    # 2) tables
    tbls = _tables_by_page(
        analyse.get("tables", []), as_="markdown" if output in {"markdown", "json"} else "dataframe"
    )

    for p in range(1, pages_total + 1):
        items = page_content[p]["structured_content"]
        for tbl in tbls.get(p, []):
            if output in {"markdown", "json"}:
                para_texts = [i["content"] for i in items]
                flat = tbl["table_list"]
                idx = [i for i, txt in enumerate(para_texts) if any(tok in txt for tok in flat)]
                if idx:
                    items[idx[0] : idx[-1] + 1] = [tbl["table_content"]]
                else:
                    items.append(tbl["table_content"])
            else:
                items.append(tbl["table_content"])

        page_content[p]["structured_content"] = _normalise(items)
        if include_tags:
            parts = [
                ("##TABLE##" if it["role"] == "table" else "##PARAGRAPH##") + "\n" + it["content"]
                for it in page_content[p]["structured_content"]
            ]
        else:
            parts = [it["content"] for it in page_content[p]["structured_content"]]
        page_content[p]["unstructured_content"] = "\n".join(parts)

    return page_content

###############################################################################
# Convenience wrappers
###############################################################################

def text_formatting(extracted):
    return {p: d["unstructured_content"] for p, d in extracted.items()}


def text_formatting_list(extracted):
    return [d["unstructured_content"] for _, d in sorted(extracted.items())]

###############################################################################
# File output helper
###############################################################################

def save_content_to_txt(page_dict: Dict[int, Dict[str, Any]], file_path: str) -> str:
    """Save *all* pages into a single UTF‑8 text file.

    Returns the written file path so callers can chain or log it.
    """
    ordered_pages = [page_dict[p]["unstructured_content"] for p in sorted(page_dict)]
    with open(file_path, "w", encoding="utf-8") as f:
        for idx, page_content in enumerate(ordered_pages, start=1):
            f.write(f"=== Page {idx} ===\n")
            f.write(page_content)
            f.write("\n\n")
    return file_path

def convert_to_txt(page_dict: Dict[int, Dict[str, Any]]) -> str:
    ordered_pages = [page_dict[p]["unstructured_content"].strip() for p in sorted(page_dict) if page_dict[p]["unstructured_content"].strip()]
    text = ""
    for idx, page_content in enumerate(ordered_pages, start=1):
        if page_content.strip() == "":
            continue
        text += f"=== Page {idx} ===\n"
        text += page_content
        text += "\n\n"
    return text.strip()