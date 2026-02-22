# Real-world Parquet data sources

References for **real** Parquet catalogs (RA/Dec, large-scale) for cross-matching and out-of-core testing.

---

## What Pleiades needs

**Input (per catalog)**

- **ra**, **dec** — configurable via `ra_col` / `dec_col` (default `"ra"`, `"dec"`); Float32 or Float64
- **Units:** degrees (default) or radians via `ra_dec_units="rad"`
- **One ID column** — `id_col_a` / `id_col_b`, or first non‑ra/dec; Int64, Int32, Utf8, LargeUtf8

**Output**

- Exactly: **id_a**, **id_b**, **separation_arcsec**
- To add catalog columns later: `merge_match_to_catalog(matches_path, catalog_path, output_path, catalog_side="a"|"b")`

---

## Catalog schemas (typical)

| Source | RA | Dec | ID | Units |
|--------|-----|-----|-----|-------|
| **Gaia DR3** | `ra` | `dec` | `source_id` | degrees |
| **IRSA AllWISE / NEOWISE** | `ra` | `dec` | e.g. `cntr` | degrees |
| **LSST / Rubin DP0.2** | `coord_ra` | `coord_dec` | `objectId` | **radians** |
| **SDSS** | (varies) | (varies) | (varies) | degrees typical |

---

## Support and usage

- **Gaia DR3** — Defaults: `ra`, `dec`, `source_id`
- **IRSA (AllWISE, NEOWISE)** — Use `ra`/`dec` (or set `ra_col`/`dec_col`) and an ID column (e.g. `cntr`)
- **LSST DP0.2** — `ra_col="coord_ra"`, `dec_col="coord_dec"`, `id_col_*="objectId"`, **`ra_dec_units="rad"`**
- **SDSS** — Set `ra_col`/`dec_col`/`id_col_*` to match the product’s Parquet schema

---

## Gaia

- **Gaia DR3** (~1.8B sources)
  - ESA CDN: `http://cdn.gea.esac.esa.int/Gaia/gdr3/` (multi-TB)
  - AWS (STScI): `s3://stpubdata/gaia/` — HATS-partitioned Parquet, no account:
    ```bash
    aws s3 ls --no-sign-request s3://stpubdata/gaia/
    ```
  - [LSDB](https://docs.lsdb.io/) for cross-matching Gaia with other HATS catalogs in the cloud

---

## IRSA (IPAC) — HEALPix Parquet

- Parquet catalog docs: https://irsa.ipac.caltech.edu/docs/parquet_catalogs/
- Cloud access: [IRSA Cloud Data Access](https://irsa.ipac.caltech.edu/cloud_access/), [notebook tutorials](https://irsa.ipac.caltech.edu/docs/notebooks/#accessing-irsa-s-cloud-holdings)
- Catalogs (bulk scripts on IRSA):
  - AllWISE Source Catalog — Parquet, HEALPix order 5 (~340 GB)
  - NEOWISE Single-exposure — 200B+ rows (2015–2024)
  - Euclid Q1 Merged, WISE All-Sky Release

---

## SDSS

- Bulk: https://www.sdss4.org/dr16/data_access/bulk/ (rsync/wget; >125 TB). Parquet per product.

---

## LSST / Vera Rubin

- **DP0.2** (simulated): Parquet, tract-partitioned
  - [Rubin Science Platform](https://data.lsst.cloud/), [DP0 docs](https://dp0-2.lsst.io/) (DP0 registration)

---

## Local POC (synthetic, real-world-like)

Default: **10M rows** per catalog (~100 MB Parquet). Example:

```bash
# Generate 10M rows
uv run python scripts/generate_large_catalog.py 10000000

# Stream: add pixel_id, 100k rows per chunk
uv run python scripts/poc_large_files.py -b 100000
```

- Typical: ~6.5M rows/s, 100 chunks; input ~250 MB, output ~290 MB

**Real data:** download a subset (e.g. one HEALPix partition from IRSA or Gaia HATS from S3), then:

```bash
uv run python scripts/poc_large_files.py /path/to/real_catalog.parquet -o /path/to/out.parquet -b 100000
```
