#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pyodbc
from dotenv import load_dotenv


# ============================================================
# Naming conventions (keep consistent)
# ============================================================
# Document formats (doc_format_code) and doc types (doc_type_code):
#   CCD, RBI, RBI_EVERGREENING, OTHER
#
# Domains (table_domain):
#   ASSET_REGISTER, RBI_RISK, RBI_INSPECTION
#
# IMPORTANT:
# This script currently assumes doc_type_code == doc_format_code.
# That is why we insert doc_type_code rows matching formats (e.g. RBI).
# ============================================================


# ----------------------------
# Environment and SQL connect
# ----------------------------

def load_env() -> None:
    # Loads .env from current working directory if present
    load_dotenv()


def connect_sql() -> pyodbc.Connection:
    """
    Requires:
      VIRT_SENS_SQL_SERVER   e.g. virt-sens-dev-sql-weu.database.windows.net
      VIRT_SENS_SQL_DB       e.g. virt-sens-dev-db
      VIRT_SENS_SQL_USER
      VIRT_SENS_SQL_PASSWORD
    """
    load_env()

    server = os.getenv("VIRT_SENS_SQL_SERVER")
    db = os.getenv("VIRT_SENS_SQL_DB")
    user = os.getenv("VIRT_SENS_SQL_USER")
    password = os.getenv("VIRT_SENS_SQL_PASSWORD")

    missing = [k for k, v in {
        "VIRT_SENS_SQL_SERVER": server,
        "VIRT_SENS_SQL_DB": db,
        "VIRT_SENS_SQL_USER": user,
        "VIRT_SENS_SQL_PASSWORD": password,
    }.items() if not v]

    if missing:
        raise RuntimeError(f"Set env vars {', '.join(missing)} before running.")

    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server=tcp:{server},1433;"
        f"Database={db};"
        f"Uid={user};"
        f"Pwd={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)


# ----------------------------
# JSON helpers (Document Intelligence)
# ----------------------------

def _get_analyze_result(layout: Dict[str, Any]) -> Dict[str, Any]:
    # Document Intelligence often wraps output as {"analyzeResult": {...}}
    ar = layout.get("analyzeResult") if isinstance(layout, dict) else None
    if isinstance(ar, dict):
        return ar
    return layout


def load_layout(layout_json_path: str) -> Dict[str, Any]:
    with open(layout_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tables_and_pages(layout: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ar = _get_analyze_result(layout)
    tables = ar.get("tables") or []
    pages = ar.get("pages") or []
    return tables, pages


# ----------------------------
# Text normalization / tokenization
# ----------------------------

def _norm_token(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().upper()
    # normalize whitespace
    t = re.sub(r"\s+", " ", t)
    # normalize dash variants
    t = t.replace("–", "-").replace("—", "-")
    return t


def collect_table_tokens(table: Dict[str, Any], max_cells: int = 2000) -> Tuple[set, int]:
    """
    Returns (tokens_set, max_row_cell_count_estimate).
    Tokenizes cell contents.
    """
    cells = table.get("cells") or []
    tokens = set()
    row_cell_count: Dict[int, int] = {}

    for c in cells[:max_cells]:
        txt = c.get("content") if isinstance(c, dict) else None
        if not txt:
            continue

        # split into crude tokens
        parts = re.split(r"[^A-Za-z0-9\-/]+", str(txt))
        for p in parts:
            p2 = _norm_token(p)
            if p2:
                tokens.add(p2)

        r = c.get("rowIndex") if isinstance(c, dict) else None
        if isinstance(r, int):
            row_cell_count[r] = row_cell_count.get(r, 0) + 1

    max_cols = max(row_cell_count.values()) if row_cell_count else 0
    return tokens, max_cols


def token_matches(required_token: str, tokens: set) -> bool:
    rt = _norm_token(required_token)

    # Prefix rule: if token ends with '-', match any token that starts with it (e.g. 'CL-' matches 'CL-02')
    if rt.endswith("-"):
        return any(t.startswith(rt) for t in tokens)

    return rt in tokens


# ----------------------------
# Small safety helpers
# ----------------------------

def trunc(s: Optional[str], max_len: int) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s)
    return s2[:max_len]


def dedupe_stage_rows(rows: List[Tuple[Any, Any, Any]]) -> List[Tuple[Any, Any, Any]]:
    """
    rows: [(page_no, table_title, row_json), ...]
    Dedupe on (table_title, row_json) because row_json includes rowIndex and cells.
    """
    seen = set()
    out = []
    for page_no, table_title, row_json in rows:
        key = (str(table_title), str(row_json))
        if key in seen:
            continue
        seen.add(key)
        out.append((page_no, table_title, row_json))
    return out


# ----------------------------
# SQL config loaders
# ----------------------------

def load_doc_type_id(conn: pyodbc.Connection, doc_type_code: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT doc_type_id FROM dbo.doc_type WHERE doc_type_code = ?", doc_type_code)
    r = cur.fetchone()
    if not r:
        raise RuntimeError(f"doc_type_code not found in dbo.doc_type: {doc_type_code}")
    return int(r[0])


def load_signatures(conn: pyodbc.Connection, doc_type_id: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT domain, required_tokens_json, optional_tokens_json, min_cols, max_cols, score_threshold, version
        FROM dbo.doc_table_signature
        WHERE doc_type_id = ? AND active_flag = 1
        """,
        doc_type_id
    )
    sigs: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        domain = row[0]
        required = json.loads(row[1]) if row[1] else []
        optional = json.loads(row[2]) if row[2] else []
        min_cols = int(row[3]) if row[3] is not None else None
        max_cols = int(row[4]) if row[4] is not None else None
        threshold = float(row[5]) if row[5] is not None else 0.5
        version = int(row[6]) if row[6] is not None else 1
        sigs.append({
            "domain": domain,
            "required": required,
            "optional": optional,
            "min_cols": min_cols,
            "max_cols": max_cols,
            "threshold": threshold,
            "version": version,
        })
    return sigs


def get_doc_format_code(conn: pyodbc.Connection, doc_id: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT doc_format_code FROM dbo.documents WHERE doc_id = ?", doc_id)
    r = cur.fetchone()
    if not r or not r[0]:
        raise RuntimeError(f"doc_id not found or doc_format_code is NULL in dbo.documents: {doc_id}")
    return str(r[0]).strip()


def is_domain_enabled(conn: pyodbc.Connection, doc_id: str, domain: str) -> bool:
    doc_format_code = get_doc_format_code(conn, doc_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT active_flag
        FROM dbo.format_domain
        WHERE doc_format_code = ? AND domain = ?
        """,
        doc_format_code, domain
    )
    r = cur.fetchone()
    return bool(r and int(r[0]) == 1)


def load_domain_column_map(conn: pyodbc.Connection, doc_id: str, domain: str) -> Dict[str, int]:
    doc_format_code = get_doc_format_code(conn, doc_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT target_field, source_index
        FROM dbo.domain_column_map
        WHERE doc_format_code = ? AND domain = ? AND active_flag = 1
        """,
        doc_format_code, domain
    )
    return {str(r[0]): int(r[1]) for r in cur.fetchall()}


# ----------------------------
# Classification
# ----------------------------

def classify_table(table: Dict[str, Any], signatures: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
    tokens, max_cols = collect_table_tokens(table)

    best_domain = None
    best_score = 0.0

    for sig in signatures:
        req = sig["required"]
        opt = sig["optional"]
        min_cols = sig["min_cols"]
        max_cols_allowed = sig["max_cols"]
        threshold = sig["threshold"]

        if min_cols is not None and max_cols < min_cols:
            continue
        if max_cols_allowed is not None and max_cols > max_cols_allowed:
            continue

        req_hits = sum(1 for t in req if token_matches(t, tokens))
        opt_hits = sum(1 for t in opt if token_matches(t, tokens))
        denom = max(1, len(req) + len(opt))
        score = (req_hits + opt_hits) / denom

        if req and req_hits == 0:
            # required tokens exist but none matched
            continue

        if score >= threshold and score > best_score:
            best_score = score
            best_domain = sig["domain"]

    return best_domain, float(best_score)


def update_table_domain(conn: pyodbc.Connection, doc_id: str, run_id: str, table_title: str, domain: str, score: float) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE dbo.stage_layout_table_rows
        SET table_domain = ?, table_confidence = ?
        WHERE doc_id = ? AND run_id = ? AND table_title = ?
        """,
        domain, float(score), doc_id, run_id, table_title
    )
    conn.commit()


def classify_existing_rows_only(conn: pyodbc.Connection, doc_id: str, run_id: str, layout_json_path: str) -> None:
    layout = load_layout(layout_json_path)
    tables, _pages = get_tables_and_pages(layout)
    if not tables:
        raise RuntimeError("No tables found in layout JSON.")

    # doc_type_code is assumed to match doc_format_code (CCD/RBI/RBI_EVERGREENING/OTHER)
    doc_type_code = get_doc_format_code(conn, doc_id)
    doc_type_id = load_doc_type_id(conn, doc_type_code)
    signatures = load_signatures(conn, doc_type_id)

    updated_tables = 0
    for t_idx, t in enumerate(tables):
        table_title = f"TABLE_{t_idx}"  # must match staging
        domain, score = classify_table(t, signatures)
        if domain:
            update_table_domain(conn, doc_id, run_id, table_title, domain, score)
            updated_tables += 1

    print(f"OK: classified {updated_tables} tables for {doc_id}/{run_id}")


# ----------------------------
# Staging (raw rows)
# ----------------------------

def iter_table_rows(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of rows like: {"rowIndex": int, "cells": [str, ...]}
    Based on Document Intelligence 'tables[].cells[]' structure.
    """
    cells = table.get("cells") or []
    # Build a map (rowIndex -> colIndex -> content)
    grid: Dict[int, Dict[int, str]] = {}
    max_col_by_row: Dict[int, int] = {}

    for c in cells:
        if not isinstance(c, dict):
            continue
        r = c.get("rowIndex")
        col = c.get("columnIndex")
        txt = c.get("content")
        if r is None or col is None:
            continue
        if txt is None:
            txt = ""
        r = int(r)
        col = int(col)
        grid.setdefault(r, {})[col] = str(txt)
        max_col_by_row[r] = max(max_col_by_row.get(r, -1), col)

    rows_out = []
    for r in sorted(grid.keys()):
        max_col = max_col_by_row.get(r, -1)
        row_cells = []
        for c in range(max_col + 1):
            row_cells.append(grid[r].get(c, ""))
        rows_out.append({"rowIndex": r, "cells": row_cells})

    return rows_out


def stage_layout_rows(conn: pyodbc.Connection, doc_id: str, run_id: str, layout_json_path: str) -> None:
    layout = load_layout(layout_json_path)
    tables, _pages = get_tables_and_pages(layout)
    if not tables:
        raise RuntimeError("No tables found in layout JSON.")

    cur = conn.cursor()

    # idempotent: clear stage rows for same doc/run
    cur.execute("DELETE FROM dbo.stage_layout_table_rows WHERE doc_id=? AND run_id=?", doc_id, run_id)
    conn.commit()
    print(f"Cleared existing stage rows for {doc_id}/{run_id}")

    insert_sql = """
      INSERT INTO dbo.stage_layout_table_rows (doc_id, run_id, page_no, table_title, row_json)
      VALUES (?, ?, ?, ?, ?)
    """

    inserted = 0
    for t_idx, t in enumerate(tables):
        table_title = f"TABLE_{t_idx}"

        # Page number: best effort from boundingRegions if present
        page_no = None
        br = t.get("boundingRegions") if isinstance(t, dict) else None
        if isinstance(br, list) and br:
            pn = br[0].get("pageNumber")
            if pn is not None:
                page_no = int(pn)

        rows = iter_table_rows(t)
        for row in rows:
            row_json = json.dumps(row, ensure_ascii=False)
            cur.execute(insert_sql, doc_id, run_id, page_no, table_title, row_json)
            inserted += 1

    conn.commit()
    print(f"OK: inserted {inserted} raw rows into dbo.stage_layout_table_rows")


# ----------------------------
# RBI risk facts builder
# ----------------------------

RISK_SET = {"LOW", "MEDIUM", "MEDIUM-HIGH", "HIGH"}


def _clean_risk(s: str) -> str:
    t = _norm_token(s)
    t = t.replace("MEDIUM -HIGH", "MEDIUM-HIGH").replace("MEDIUM- HIGH", "MEDIUM-HIGH")
    return t


def is_risk_value_cell(s: str) -> bool:
    return _clean_risk(s) in RISK_SET


def build_rbi_risk_facts(conn: pyodbc.Connection, doc_id: str, run_id: str) -> None:
    if not is_domain_enabled(conn, doc_id, "RBI_RISK"):
        print(f"Skip: RBI_RISK is not enabled for this document format (doc_id={doc_id}).")
        return

    cur = conn.cursor()

    # idempotent
    cur.execute("DELETE FROM dbo.rbi_risk_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    conn.commit()

    cur.execute(
        """
        SELECT DISTINCT page_no, table_title, row_json
        FROM dbo.stage_layout_table_rows
        WHERE doc_id=? AND run_id=? AND table_domain='RBI_RISK'
        """,
        doc_id, run_id
    )
    rows = dedupe_stage_rows(cur.fetchall())

    insert_sql = """
      INSERT INTO dbo.rbi_risk_facts
      (doc_id, run_id, table_title, row_index, risk_slot, page_no,
       equipment_tag, corrosion_loop, risk_label, risk_value, instr_primary, instr_secondary)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    for page_no, table_title, row_json in rows:
        obj = json.loads(row_json)
        row_index = int(obj.get("rowIndex"))

        # Skip header rows
        if row_index == 0:
            continue

        cells = obj.get("cells", [])
        if not cells or len(cells) < 4:
            continue

        equipment_tag = cells[0].strip() if len(cells) > 0 and isinstance(cells[0], str) else (str(cells[0]).strip() if len(cells) > 0 and cells[0] is not None else None)
        corrosion_loop = cells[1].strip() if len(cells) > 1 and isinstance(cells[1], str) else (str(cells[1]).strip() if len(cells) > 1 and cells[1] is not None else None)

        risk_cols = []
        for idx in range(2, len(cells)):
            v = cells[idx]
            if v is None:
                continue
            if is_risk_value_cell(str(v)):
                risk_cols.append(idx)

        if not risk_cols:
            continue

        last_risk = max(risk_cols)
        tail = []
        for x in cells[last_risk + 1:]:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                tail.append(sx)

        instr_primary = tail[0] if len(tail) > 0 else None
        instr_secondary = tail[1] if len(tail) > 1 else None

        slot = 0
        for idx in risk_cols:
            slot += 1
            risk_value = _clean_risk(str(cells[idx]))
            risk_label = f"RISK_COL_{idx}"  # placeholder, can be upgraded later

            cur.execute(
                insert_sql,
                doc_id, run_id, str(table_title), int(row_index), int(slot), int(page_no) if page_no is not None else None,
                trunc(equipment_tag, 256), trunc(corrosion_loop, 128),
                trunc(risk_label, 256), trunc(risk_value, 64),
                trunc(instr_primary, 4000), trunc(instr_secondary, 4000)
            )
            inserted += 1

    conn.commit()
    print(f"OK: inserted {inserted} rbi_risk_facts rows for {doc_id}/{run_id}")


# ----------------------------
# Asset register facts builder
# ----------------------------

def extract_cl(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\bCL-\s*\d+\b", str(text).upper())
    if not m:
        return None
    return m.group(0).replace(" ", "")


def build_asset_register_facts(conn: pyodbc.Connection, doc_id: str, run_id: str) -> None:
    if not is_domain_enabled(conn, doc_id, "ASSET_REGISTER"):
        print(f"Skip: ASSET_REGISTER is not enabled for this document format (doc_id={doc_id}).")
        return

    colmap = load_domain_column_map(conn, doc_id, "ASSET_REGISTER")

    cur = conn.cursor()

    # idempotent
    cur.execute("DELETE FROM dbo.asset_register_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    conn.commit()

    cur.execute(
        """
        SELECT DISTINCT page_no, table_title, row_json
        FROM dbo.stage_layout_table_rows
        WHERE doc_id=? AND run_id=? AND table_domain='ASSET_REGISTER'
        """,
        doc_id, run_id
    )
    rows = dedupe_stage_rows(cur.fetchall())

    insert_sql = """
      INSERT INTO dbo.asset_register_facts
      (doc_id, run_id, table_title, row_index, page_no,
       tag, cl, description, equip_type, material, thickness, insulated, raw_cells_json)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    inserted = 0
    for page_no, table_title, row_json in rows:
        obj = json.loads(row_json)
        row_index = int(obj.get("rowIndex"))

        # Skip header rows
        if row_index == 0:
            continue

        cells = obj.get("cells", [])
        if not cells or len(cells) < 4:
            continue

        def cell(idx: Optional[int]) -> Optional[str]:
            if idx is None or idx < 0 or idx >= len(cells):
                return None
            v = cells[idx]
            if v is None:
                return None
            return str(v).strip()

        tag = cell(colmap.get("tag"))
        desc = cell(colmap.get("description"))
        equip_type = cell(colmap.get("equip_type"))
        material = cell(colmap.get("material"))
        thickness = cell(colmap.get("thickness"))
        insulated = cell(colmap.get("insulated"))

        cl = extract_cl(tag)
        raw_cells_json = trunc(json.dumps(cells, ensure_ascii=False), 4000)

        cur.execute(
            insert_sql,
            doc_id, run_id, str(table_title), int(row_index), int(page_no) if page_no is not None else None,
            trunc(tag, 256), trunc(cl, 64), trunc(desc, 512), trunc(equip_type, 64),
            trunc(material, 128), trunc(thickness, 64), trunc(insulated, 32),
            raw_cells_json
        )
        inserted += 1

    conn.commit()
    print(f"OK: inserted {inserted} asset_register_facts rows for {doc_id}/{run_id}")

def build_rbi_risk_summary_facts(conn: pyodbc.Connection, doc_id: str, run_id: str) -> None:
    domain = "RBI_RISK_SUMMARY"

    if not is_domain_enabled(conn, doc_id, domain):
        fmt = get_doc_format_code(conn, doc_id)
        print(f"Skip: {domain} is not enabled for this document format (format={fmt}, doc_id={doc_id}).")
        return

    cur = conn.cursor()

    # idempotent
    cur.execute("DELETE FROM dbo.rbi_risk_summary_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    conn.commit()

    insert_sql = """
      INSERT INTO dbo.rbi_risk_summary_facts
      (doc_id, run_id, page_no, table_title, row_index, row_json, created_utc)
      SELECT
          s.doc_id,
          s.run_id,
          s.page_no,
          s.table_title,
          TRY_CONVERT(int, JSON_VALUE(s.row_json, '$.rowIndex')) AS row_index,
          s.row_json,
          SYSUTCDATETIME()
      FROM dbo.stage_layout_table_rows s
      WHERE s.doc_id = ?
        AND s.run_id = ?
        AND s.table_domain = ?;
    """

    cur.execute(insert_sql, doc_id, run_id, domain)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM dbo.rbi_risk_summary_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    count = int(cur.fetchone()[0])
    print(f"OK: inserted {count} rbi_risk_summary_facts rows for {doc_id}/{run_id}")


# ----------------------------
# RBI inspection facts builder
# ----------------------------

def build_rbi_inspection_facts(conn: pyodbc.Connection, doc_id: str, run_id: str) -> None:
    """
    Build RBI_INSPECTION facts from already-staged, already-classified rows.
    Uses dbo.domain_column_map as a required config guardrail.
    Facts table currently stores raw row_json for extensibility.
    """
    domain = "RBI_INSPECTION"

    if not is_domain_enabled(conn, doc_id, domain):
        fmt = get_doc_format_code(conn, doc_id)
        print(f"Skip: {domain} is not enabled for this document format (format={fmt}, doc_id={doc_id}).")
        return

    # Required by design: ensures format-specific mapping exists (even if we store raw row_json for now)
    colmap = load_domain_column_map(conn, doc_id, domain)
    if not colmap:
        fmt = get_doc_format_code(conn, doc_id)
        raise RuntimeError(f"No active domain_column_map rows for {fmt}/{domain}")

    cur = conn.cursor()

    # idempotent
    cur.execute("DELETE FROM dbo.rbi_inspection_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    conn.commit()

    insert_sql = """
      INSERT INTO dbo.rbi_inspection_facts
      (doc_id, run_id, page_no, table_title, row_index, row_json, created_utc)
      SELECT
          s.doc_id,
          s.run_id,
          s.page_no,
          s.table_title,
          TRY_CONVERT(int, JSON_VALUE(s.row_json, '$.rowIndex')) AS row_index,
          s.row_json,
          SYSUTCDATETIME()
      FROM dbo.stage_layout_table_rows s
      WHERE s.doc_id = ?
        AND s.run_id = ?
        AND s.table_domain = ?
        AND (
              JSON_VALUE(s.row_json, '$.rowIndex') IS NULL
              OR TRY_CONVERT(int, JSON_VALUE(s.row_json, '$.rowIndex')) NOT IN (0, 1)
            );
    """

    cur.execute(insert_sql, doc_id, run_id, domain)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM dbo.rbi_inspection_facts WHERE doc_id=? AND run_id=?", doc_id, run_id)
    count = int(cur.fetchone()[0])
    print(f"OK: inserted {count} rbi_inspection_facts rows for {doc_id}/{run_id}")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual Sensor POC loader/classifier/builders")
    p.add_argument("--doc-id", required=True, help="DOC-0001 etc")
    p.add_argument("--run-id", required=True, help="Timestamp run id like 20260220T101330832Z")
    p.add_argument("--layout", required=False, help="Path to Document Intelligence layout JSON (contains analyzeResult)")
    p.add_argument("--stage", action="store_true", help="Stage raw table rows into dbo.stage_layout_table_rows")
    p.add_argument("--classify-only", dest="classify_only", action="store_true", help="Classify staged tables using signatures")
    p.add_argument("--build-rbi-risk-summary", dest="build_rbi_risk_summary", action="store_true",
               help="Build dbo.rbi_risk_summary_facts from RBI_RISK_SUMMARY staged rows (no --layout required)")
    p.add_argument("--build-rbi-risk", dest="build_rbi_risk", action="store_true",
                   help="Build dbo.rbi_risk_facts from RBI_RISK staged rows")
    p.add_argument("--build-asset-register", dest="build_asset_register", action="store_true",
                   help="Build dbo.asset_register_facts from ASSET_REGISTER staged rows")
    p.add_argument("--build-rbi-inspection", dest="build_rbi_inspection", action="store_true",
                   help="Build dbo.rbi_inspection_facts from RBI_INSPECTION staged rows (no --layout required)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    any_flag = (
        args.stage
        or args.classify_only
        or args.build_rbi_risk
        or args.build_asset_register
        or args.build_rbi_inspection
        or args.build_rbi_risk_summary

    )
    # Default behavior: if user gives no flags, do staging.
    if not any_flag:
        args.stage = True

    conn = connect_sql()

    # Require --layout only when staging or classifying.
    if (args.stage or args.classify_only) and not args.layout:
        raise RuntimeError("--layout is required when using --stage or --classify-only")

    if args.stage:
        print("RUN: stage")
        stage_layout_rows(conn, args.doc_id, args.run_id, args.layout)

    if args.classify_only:
        print("RUN: classify-only")
        classify_existing_rows_only(conn, args.doc_id, args.run_id, args.layout)

    if args.build_rbi_risk:
        print("RUN: build-rbi-risk")
        build_rbi_risk_facts(conn, args.doc_id, args.run_id)

    if args.build_asset_register:
        print("RUN: build-asset-register")
        build_asset_register_facts(conn, args.doc_id, args.run_id)

    if args.build_rbi_inspection:
        print("RUN: build-rbi-inspection")
        build_rbi_inspection_facts(conn, args.doc_id, args.run_id)
    
    if args.build_rbi_risk_summary:
        print("RUN: build-rbi-risk-summary")
        build_rbi_risk_summary_facts(conn, args.doc_id, args.run_id)



if __name__ == "__main__":
    main()
