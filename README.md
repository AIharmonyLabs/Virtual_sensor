# Virtual Sensor POC (docs) – Config-Driven Document Intelligence Pipeline

This repository contains a proof-of-concept pipeline that:
1) Stages Azure AI Document Intelligence (prebuilt-layout) output JSON into Azure SQL.
2) Classifies extracted tables into business domains using SQL configuration (signatures).
3) Builds normalized fact tables per domain (currently: RBI Risk and Asset Register).
4) Keeps ingestion and Document Intelligence execution manual for now, but prepares for front-end upload automation later.

## What is implemented today

### Supported document formats
- **CCD** (implemented end-to-end)
- **RBI_INSPECTION**
- **RBI_EVERGREENING**

Planned formats (not yet implemented end-to-end):

- **OTHER**

### Supported domains (end-to-end)
- **RBI_RISK**
  - Output table: `dbo.rbi_risk_facts`
- **ASSET_REGISTER**
  - Output table: `dbo.asset_register_facts`
  - Column mapping is config-driven via `dbo.domain_column_map`

### High-level architecture
- Input: PDF uploaded to Azure Blob Storage
- Document Intelligence: prebuilt-layout run in Studio (manual)
- Output: `layout.json` uploaded to Blob (manual)
- Processing: local Python script reads `layout.json`, loads to SQL, classifies via SQL signatures, builds facts

---

## Azure resources

### Resource Group
- `Hakima`

### Storage account
- `alahsa`

### Blob containers
- `virt-sens-raw-pdfs` (source PDFs)
- `virt-sens-extractions-raw` (Document Intelligence outputs)
- `virt-sens-quarantine` (failed/invalid PDFs)

### Document Intelligence
- Resource: `virt-sens-dev` (S0)
- Model: `prebuilt-layout`
- Tooling: Document Intelligence Studio (manual)

### Azure SQL
- Server: `virt-sens-dev-sql-weu.database.windows.net`
- Database: `virt-sens-dev-db`

---

## Repository contents

### Main script
- `poc_load_stage_rows.py`

The script supports:
- `--stage` (stage raw rows)
- `--classify-only` (apply signatures to assign `table_domain`)
- `--build-rbi-risk` (build `dbo.rbi_risk_facts`)
- `--build-asset-register` (build `dbo.asset_register_facts`)

These flags are not mutually exclusive. You can run multiple steps in a single command.

---

## Data conventions

### Document ID and Run ID
- `doc_id`: `DOC-0001`, `DOC-0002`, etc
- `run_id`: UTC timestamp string, example: `20260220T101330832Z`

### Blob convention for extracted JSON
Upload Document Intelligence output JSON to:
- `virt-sens-extractions-raw/DOC-xxxx/<run_id>/layout.json`

---

## Database objects used by the pipeline

### Operational metadata
- `dbo.documents`
  - Holds doc registry including `doc_format_code`
- `dbo.doc_runs`
  - Holds run metadata including `layout_json_path` and status

### Staging
- `dbo.stage_layout_table_rows`
  - Stages row-wise JSON by (doc_id, run_id, table_title, row_json)
  - Updated by classification:
    - `table_domain`
    - `table_confidence`

### Format and domain configuration
- `dbo.doc_format`
  - Allowed document formats: CCD, RBI_INSPECTION, RBI_EVERGREENING, OTHER
- `dbo.format_domain`
  - Which domains are enabled per format
- `dbo.doc_type`
  - Config lookup key (doc_type_code matches doc_format_code)
- `dbo.doc_table_signature`
  - Signatures for table classification per doc_type_code and domain
- `dbo.domain_column_map`
  - Column mapping per (doc_format_code, domain, target_field)
  - Used for ASSET_REGISTER now; intended pattern for future domains

### Domain facts tables
- `dbo.rbi_risk_facts`
- `dbo.asset_register_facts`

---

## Local setup (macOS)

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
