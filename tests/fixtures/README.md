# Test data

Synthetic Parquet catalogs for Pleiades tests. All use **ra** and **dec** in **degrees**.

| File | Rows | Description |
|------|------|-------------|
| `catalog_a.parquet` | 500 | Catalog A: `source_id`, `ra`, `dec` |
| `catalog_b.parquet` | 300 | Catalog B: `object_id`, `ra`, `dec` |
| `catalog_a_small.parquet` | 100 | Same schema; 10 sources have counterparts in B within ~1.5″ |
| `catalog_b_small.parquet` | 80 | Same schema; use with `catalog_a_small` and `radius_arcsec=2.0` for match tests |

Regenerate from project root:

```bash
uv run python scripts/generate_test_data.py
```
