# Real-world astronomical Parquet data sources

References for **real** Parquet catalogs (RA/Dec, large-scale) suitable for cross-matching and out-of-core testing.

---

## Parquet field analysis and implementation support

Summary of **typical Parquet columns** in the catalogs below and how the **current Pleiades implementation** handles them.

### Catalog schemas (from documentation and web search)

| Source | RA column | Dec column | ID column | Other columns | Units |
|--------|-----------|------------|-----------|----------------|-------|
| **Gaia DR3** | `ra` | `dec` | `source_id` (long) | ra_error, dec_error, parallax, pmra, pmdec, designation, … | degrees |
| **IRSA AllWISE** | `ra` | `dec` | `cntr` | Many photometry/quality columns | degrees |
| **IRSA NEOWISE** | (similar) | (similar) | (varies) | 145+ columns, HEALPix k=5 | degrees |
| **LSST / Rubin DP0.2** | `coord_ra` | `coord_dec` | `objectId` (long) | tract, patch, flux columns, shape_*, detect_*, … | **radians** |
| **SDSS** | (varies by product) | (varies) | (varies) | Parquet schema product-dependent | degrees typical |

### What the current implementation requires and does

- **Required input columns (per catalog):**
  - **ra**, **dec**: configurable via `ra_col` / `dec_col` (default `"ra"`, `"dec"`). Must be numeric; **Float32 or Float64** supported (Rust engine). Units: **degrees** (default) or **radians** via `ra_dec_units="rad"`.
  - **One ID column**: configurable via `id_col_a` / `id_col_b`, or inferred as first non‑ra/dec column. Types supported: **Int64, Int32, Utf8, LargeUtf8**.
- **Output:** Exactly three columns: **id_a**, **id_b**, **separation_arcsec**. No other catalog columns are passed through; use `merge_match_to_catalog()` to attach catalog columns by ID.

### Support status

- **Gaia DR3**: Supported. Use defaults: `ra`, `dec`, `source_id`. Other columns (parallax, pm, etc.) are ignored for the join.
- **IRSA (AllWISE / NEOWISE)**: Supported for the join if the Parquet has `ra`/`dec` (or you set `ra_col`/`dec_col`) and an ID-like column (e.g. `cntr`). Float32 ra/dec work.
- **LSST DP0.2**: Supported. Use `ra_col="coord_ra"`, `dec_col="coord_dec"`, `id_col_*="objectId"`, **`ra_dec_units="rad"`** (coordinates are in radians).
- **SDSS**: Supported as long as the specific Parquet product has numeric ra/dec (and optional ID) columns; use `ra_col`/`dec_col`/`id_col_*` to match the product’s schema.

### Optional: pass-through columns

The match output contains only the two IDs and `separation_arcsec`. To add catalog columns (e.g. magnitudes, parallax) to the result, use **merge_match_to_catalog**(matches_path, catalog_path, output_path, catalog_side=`"a"` or `"b"`) after the cross-match. A future enhancement could add optional pass-through column names to the engine.
---

## Gaia

- **Gaia DR3** (~1.8 billion sources)
  - **ESA CDN**: `http://cdn.gea.esac.esa.int/Gaia/gdr3/` (full release, multi-TB)
  - **AWS (STScI)**: `s3://stpubdata/gaia/` — HATS-partitioned Parquet, no account needed:
    ```bash
    aws s3 ls --no-sign-request s3://stpubdata/gaia/
    ```
  - **LSDB**: Use [LSDB](https://docs.lsdb.io/) for cross-matching Gaia with other HATS catalogs in the cloud.

## IRSA (IPAC) — HEALPix Parquet

- **Parquet catalog docs**: https://irsa.ipac.caltech.edu/docs/parquet_catalogs/
- **Cloud access (recommended)**: Query in place via S3; see [IRSA Cloud Data Access](https://irsa.ipac.caltech.edu/cloud_access/) for bucket/path and [notebook tutorials](https://irsa.ipac.caltech.edu/docs/notebooks/#accessing-irsa-s-cloud-holdings).
- **Catalogs** (bulk download scripts on IRSA):
  - **AllWISE Source Catalog** — Parquet, HEALPix order 5 (~340 GB)
  - **NEOWISE Single-exposure Source Table** — 200B+ rows (2015–2024)
  - **Euclid Q1 Merged Catalog**
  - **WISE All-Sky Release**

## SDSS

- **Bulk data**: https://www.sdss4.org/dr16/data_access/bulk/ (rsync/wget; total >125 TB). Parquet availability may vary by product.

## LSST / Vera Rubin

- **DP0.2** (simulated): Parquet catalogs, tract-partitioned; access via [Rubin Science Platform](https://data.lsst.cloud/) or [DP0 docs](https://dp0-2.lsst.io/) (requires DP0 registration).

## Local POC scale (real-world-like)

We use **synthetic** data at **10M rows** per catalog by default (Gaia has ~1.8B; 10M is a realistic single-node test). Example run:

```bash
# Generate 10M rows (~100 MB Parquet, 100 row-groups of 100k)
uv run python scripts/generate_large_catalog.py 10000000

# Stream process: add pixel_id, 100k rows per chunk
uv run python scripts/poc_large_files.py -b 100000
```

Typical result: **10M rows in ~1.5 s** (~6.5M rows/s), 100 chunks; input ~250 MB, output ~290 MB.

To use **real** data: download a subset from one of the sources above (e.g. one HEALPix partition from IRSA or a Gaia HATS partition from S3) and run:

```bash
uv run python scripts/poc_large_files.py /path/to/real_catalog.parquet -o /path/to/out.parquet -b 100000
```
