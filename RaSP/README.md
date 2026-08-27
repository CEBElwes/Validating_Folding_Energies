# RaSP predictions - PIN1, spg, FYN

Predictions generated with `score_gene.py`, run against the RaSP repo
(KULL-Centre/_2022_ML-ddG-Blaabjerg) at commit `f587f0dfe1fd7526bbb51874e9287051b7178271`.

## Environment
See `environment_rasp.yml` (conda env `rasp`, Python 3.9).

## Structures scored
Exact PDB IDs per gene are in `pipeline_pdb_ids_by_uniprot.json` - the same
structure set used by the FoldX pipeline (see VFE-alignment.ipynb).

## Commands used
```
python score_gene.py --repo_root . --gene PIN1 \
    --pdb_ids_json pipeline_pdb_ids_by_uniprot.json --uniprot_id Q13526 \
    --pdb_dir data/pdb_cache/PIN1 \
    --out_csv output/predictions/PIN1_rasp.csv

python score_gene.py --repo_root . --gene spg \
    --pdb_ids_json pipeline_pdb_ids_by_uniprot.json --uniprot_id P06654 \
    --pdb_dir data/pdb_cache/spg \
    --out_csv output/predictions/spg_rasp.csv

python score_gene.py --repo_root . --gene FYN \
    --pdb_ids_json pipeline_pdb_ids_by_uniprot.json --uniprot_id P06241 \
    --pdb_dir data/pdb_cache/FYN \
    --out_csv output/predictions/FYN_rasp.csv
```

Run from inside a clone of the RaSP repo at the commit above, with `score_gene.py`
copied into its root.
