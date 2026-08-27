"""
score_gene.py

Runs full saturation-mutagenesis RaSP ddG prediction across ALL PDB
structures for a given gene, using the pretrained cavity + downstream
ensemble models shipped in the RaSP repo (KULL-Centre/_2022_ML-ddG-Blaabjerg).

Changes vs. the earlier score_pdb.py:
  1. Model/dataset classes (ResidueEnvironment, ResidueEnvironmentsDataset,
     CavityModel, DownstreamModel, DDGDataset, DDGToTensor) are now imported
     DIRECTLY from the repo's colab_additonals/cavity_model.py, rather than
     re-typed into this file. This removes any remaining transcription risk
     -- including the .npz-loading/padding logic, which turned out to
     already exist in the repo as ResidueEnvironmentsDataset.parse_envs()
     (we just hadn't found it originally).
  2. Cleaning (clean_pdb) and parsing (extract_environments, DSSP-free) are
     now called as imported functions rather than shelled out to, per PDB.
  3. Scores ALL structures for one gene in a single run, writing one
     combined CSV with a pdbid column (so you can median-aggregate across
     structures downstream, same as your FoldX pipeline).
  4. Reports timing per structure, per residue, and per substitution.

IMPORTANT design choice: this does NOT independently query RCSB/PDBe for
"all structures matching this UniProt ID". That could return a different
structure set than whatever your FoldX pipeline already used, which would
make the RaSP-vs-FoldX comparison unfair. Instead you explicitly supply the
PDB ID list for a gene, via one of:
  --pdb_ids "1PGA,2XYZ,..."                           (manual list)
  --pdb_ids_json path.json --uniprot_id Q13526         (reads a
                                                        {uniprot_id:
                                                        [pdb_id, ...]} JSON)

The --pdb_ids_json format matches "pipeline_pdb_ids_by_uniprot.json", an
export added to VFE-alignment.ipynb (added right after the cell that writes
Aligned_dfs/pipeline_{uniprot_id}.csv) -- it lists, per gene, exactly the
PDB structures that survived your FoldX/Mutein pipeline's own alignment and
QC filtering. Using this (rather than an independent RCSB lookup by
UniProt ID) keeps the RaSP-vs-FoldX comparison fair, since both tools then
run on the identical structure set.

Any PDB files not already present in --pdb_dir are downloaded from RCSB.

Usage:
    python score_gene.py --repo_root . --gene PIN1 \
        --pdb_ids "1PIN,1F8A,2ITD" \
        --pdb_dir data/pdb_cache/PIN1 \
        --out_csv output/predictions/PIN1_rasp.csv

    python score_gene.py --repo_root . --gene PIN1 \
        --pdb_ids_json /path/to/pipeline_pdb_ids_by_uniprot.json --uniprot_id Q13526 \
        --pdb_dir data/pdb_cache/PIN1 \
        --out_csv output/predictions/PIN1_rasp.csv
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import model/dataset classes directly from the repo (no re-typing).
# ---------------------------------------------------------------------------


def _import_repo_modules(repo_root):
    """Add the repo's module directories to sys.path and import what we need."""
    colab_dir = os.path.join(repo_root, "colab_additonals")
    parser_dir = os.path.join(repo_root, "src", "pdb_parser_scripts")

    for p in (colab_dir, parser_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    import cavity_model  # colab_additonals/cavity_model.py -- DSSP-free, no sasa/b_factors
    from clean_pdb import clean_pdb  # src/pdb_parser_scripts/clean_pdb.py

    # extract_environments_nodssp.py must exist in src/pdb_parser_scripts/
    # (a copy of colab_additonals/extract_environments.py placed there so its
    # `import grid` resolves -- grid.py lives in src/pdb_parser_scripts/, not
    # in colab_additonals/).
    if not os.path.exists(os.path.join(parser_dir, "extract_environments_nodssp.py")):
        raise FileNotFoundError(
            "src/pdb_parser_scripts/extract_environments_nodssp.py not found. "
            "Copy colab_additonals/extract_environments.py there first (see "
            "earlier setup step) so its `import grid` resolves correctly."
        )
    import extract_environments_nodssp

    return cavity_model, clean_pdb, extract_environments_nodssp.extract_environments


# ---------------------------------------------------------------------------
# PDB ID resolution + download
# ---------------------------------------------------------------------------


def resolve_pdb_ids(args):
    if args.pdb_ids:
        return [p.strip().upper() for p in args.pdb_ids.split(",") if p.strip()]

    if args.pdb_ids_json and args.uniprot_id:
        # Expected format: {uniprot_id: [pdb_id, pdb_id, ...]}, as produced by the
        # "pipeline_pdb_ids_by_uniprot.json" export added to VFE-alignment.ipynb --
        # this is the FoldX/Mutein pipeline's own post-alignment/QC structure list
        # per gene, so scoring against it keeps the RaSP-vs-FoldX comparison fair.
        with open(args.pdb_ids_json) as f:
            mapping = json.load(f)
        pdb_ids = mapping.get(args.uniprot_id)
        if not pdb_ids:
            raise ValueError(
                f"No PDB IDs found for UniProt ID {args.uniprot_id} in {args.pdb_ids_json}."
            )
        return sorted({p.upper() for p in pdb_ids})

    raise ValueError("Must supply either --pdb_ids or (--pdb_ids_json and --uniprot_id)")


def download_pdb_if_missing(pdb_id, pdb_dir):
    os.makedirs(pdb_dir, exist_ok=True)
    out_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        return out_path, False
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {pdb_id} from {url}: {e}")
    return out_path, True


# ---------------------------------------------------------------------------
# Scoring helpers (validated against the paper's own Table 1 numbers for
# 1PGA -- kept as-is rather than re-derived, since they already reproduce
# the published RaSP-vs-Rosetta / RaSP-vs-Experimental correlations closely)
# ---------------------------------------------------------------------------


def make_ddg_dataset_classes(cavity_model):
    """Returns (DDGDataset, DDGToTensor) bound to the imported cavity_model module."""
    return cavity_model.DDGDataset, cavity_model.DDGToTensor


def get_ddg_dataloader(ddg_data, DEVICE, DDGDataset, DDGToTensor, batch_size=100):
    ddg_dataset = DDGDataset(ddg_data, transformer=DDGToTensor("pred", DEVICE))
    return DataLoader(
        ddg_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=DDGToTensor("pred", DEVICE).collate_multi,
    )


def inverse_fermi_transform(x):
    alpha = 3.0
    beta = 0.4
    EPS = 10.0 ** (-12)
    if x == 1.0:
        return 40.0
    elif 0.0 < x < 1.0:
        return (alpha * beta - np.log(-1.0 + 1.0 / x + EPS)) / beta
    elif x == 0.0:
        return -40.0
    return np.nan


def ds_pred(cavity_model_net, ds_model_net, df_total, ds_model_dir, num_ensemble, DEVICE, DDGDataset, DDGToTensor):
    dataloader = get_ddg_dataloader(df_total, DEVICE, DDGDataset, DDGToTensor)

    pdbid, chainid, variant = [], [], []
    ddg_fermi_pred = torch.empty(0, 1, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        for pdbid_batch, chainid_batch, variant_batch, x_cavity_batch, x_ds_batch in dataloader:
            ddg_fermi_pred_batch_ensemble = torch.empty(len(variant_batch), 0).to(DEVICE)
            for i in range(num_ensemble):
                ds_model_net.load_state_dict(
                    torch.load(
                        os.path.join(ds_model_dir, f"ds_model_{i}", "model.pt"),
                        map_location=torch.device(DEVICE),
                    )
                )
                ds_model_net.eval()
                cavity_pred_batch = cavity_model_net(x_cavity_batch)
                ddg_fermi_pred_batch = ds_model_net(torch.cat((cavity_pred_batch, x_ds_batch), 1))
                ddg_fermi_pred_batch_ensemble = torch.cat((ddg_fermi_pred_batch_ensemble, ddg_fermi_pred_batch), 1)

            ddg_fermi_pred_batch = torch.median(ddg_fermi_pred_batch_ensemble, 1, keepdim=True)[0]
            pdbid += pdbid_batch
            chainid += chainid_batch
            variant += variant_batch
            ddg_fermi_pred = torch.cat((ddg_fermi_pred, ddg_fermi_pred_batch), 0)

    df_ml = pd.DataFrame(ddg_fermi_pred.cpu().numpy(), columns=["score_ml_fermi"])
    df_ml["score_ml"] = df_ml["score_ml_fermi"].apply(inverse_fermi_transform)
    df_ml.insert(0, "pdbid", np.array(pdbid))
    df_ml.insert(1, "chainid", np.array(chainid))
    df_ml.insert(2, "variant", np.array(variant))
    return df_ml


AA_LIST = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]


def build_saturation_df(parsed_files, pdb_nlfs, ResidueEnvironmentsDataset):
    dataset_structure = ResidueEnvironmentsDataset(parsed_files, transformer=None)

    rows = []
    for resenv in dataset_structure:
        wt_aa = index_to_one(resenv.restype_index)
        rows.append(
            {
                "pdbid": resenv.pdb_id,
                "chainid": resenv.chain_id,
                "pos": resenv.pdb_residue_number,
                "wt_AA": wt_aa,
                "resenv": resenv,
            }
        )
    df_no_mt = pd.DataFrame(rows)

    expanded = df_no_mt.loc[df_no_mt.index.repeat(20)].reset_index(drop=True)
    expanded["mt_AA"] = AA_LIST * len(df_no_mt)
    expanded["variant"] = expanded["wt_AA"] + expanded["pos"].astype(str) + expanded["mt_AA"]

    expanded["wt_idx"] = expanded["wt_AA"].apply(one_to_index)
    expanded["mt_idx"] = expanded["mt_AA"].apply(one_to_index)
    expanded["wt_nlf"] = expanded["wt_idx"].apply(lambda i: pdb_nlfs[i])
    expanded["mt_nlf"] = expanded["mt_idx"].apply(lambda i: pdb_nlfs[i])

    return expanded, len(df_no_mt)  # also return residue count for timing stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".", help="Path to _2022_ML-ddG-Blaabjerg repo root")
    parser.add_argument("--gene", type=str, required=True, help="Gene name, used only for output file naming/logging")
    parser.add_argument("--pdb_ids", type=str, default=None, help="Comma-separated PDB IDs, e.g. '1PIN,1F8A'")
    parser.add_argument("--pdb_ids_json", type=str, default=None,
                         help="Path to pipeline_pdb_ids_by_uniprot.json, format {uniprot_id: [pdb_id, ...]}")
    parser.add_argument("--uniprot_id", type=str, default=None, help="UniProt ID to look up in --pdb_ids_json")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Folder to cache/store raw PDB files")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--num_ensemble", type=int, default=10)
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    repo_root = args.repo_root
    cavity_model, clean_pdb, extract_environments = _import_repo_modules(repo_root)
    DDGDataset, DDGToTensor = make_ddg_dataset_classes(cavity_model)
    ResidueEnvironmentsDataset = cavity_model.ResidueEnvironmentsDataset
    CavityModel = cavity_model.CavityModel
    DownstreamModel = cavity_model.DownstreamModel

    reduce_exe = os.path.join(repo_root, "src", "pdb_parser_scripts", "reduce", "reduce_src", "reduce")

    pdb_ids = resolve_pdb_ids(args)
    print(f"Gene {args.gene}: {len(pdb_ids)} PDB structure(s) to process: {pdb_ids}")

    clean_dir = os.path.join(args.pdb_dir, "cleaned")
    parsed_dir = os.path.join(args.pdb_dir, "parsed")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)

    per_structure_timing = []
    parsed_files = []

    for pdb_id in pdb_ids:
        t_struct_start = time.time()
        try:
            raw_path, was_downloaded = download_pdb_if_missing(pdb_id, args.pdb_dir)
            t_download = time.time()

            clean_pdb(raw_path, clean_dir, reduce_exe)
            clean_path = os.path.join(clean_dir, f"{pdb_id}_clean.pdb")
            t_clean = time.time()

            extract_environments(clean_path, f"{pdb_id}_clean", out_dir=parsed_dir)
            parsed_path = os.path.join(parsed_dir, f"{pdb_id}_clean_coordinate_features.npz")
            t_parse = time.time()

            parsed_files.append(parsed_path)
            per_structure_timing.append(
                {
                    "pdb_id": pdb_id,
                    "status": "ok",
                    "downloaded": was_downloaded,
                    "download_s": t_download - t_struct_start,
                    "clean_s": t_clean - t_download,
                    "parse_s": t_parse - t_clean,
                    "total_s": t_parse - t_struct_start,
                }
            )
            print(f"  [{pdb_id}] cleaned + parsed in {t_parse - t_struct_start:.1f}s")
        except Exception as e:
            per_structure_timing.append(
                {
                    "pdb_id": pdb_id,
                    "status": f"FAILED: {e}",
                    "downloaded": None,
                    "download_s": None,
                    "clean_s": None,
                    "parse_s": None,
                    "total_s": time.time() - t_struct_start,
                }
            )
            print(f"  [{pdb_id}] FAILED: {e}")

    if not parsed_files:
        print("No structures parsed successfully -- aborting.")
        sys.exit(1)

    pdb_nlfs = -np.log(
        np.load(os.path.join(repo_root, "data", "train", "cavity", "pdb_frequencies.npz"))["frequencies"]
    )

    df_structure, n_residues_total = build_saturation_df(parsed_files, pdb_nlfs, ResidueEnvironmentsDataset)
    print(f"Built {len(df_structure)} substitution rows across {n_residues_total} residues "
          f"from {len(parsed_files)} structure(s)")

    best_model_path = open(os.path.join(repo_root, "output", "cavity_models", "best_model_path.txt")).read().strip()
    cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
    cavity_model_net.load_state_dict(
        torch.load(os.path.join(repo_root, "output", "cavity_models", best_model_path), map_location=torch.device(DEVICE))
    )
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(DEVICE)

    t0 = time.time()
    df_ml = ds_pred(
        cavity_model_net,
        ds_model_net,
        df_structure,
        os.path.join(repo_root, "output", "ds_models"),
        args.num_ensemble,
        DEVICE,
        DDGDataset,
        DDGToTensor,
    )
    scoring_time = time.time() - t0
    n_substitutions = len(df_structure)

    df_out = df_structure.drop(columns=["resenv"]).merge(df_ml, on=["pdbid", "chainid", "variant"], how="inner")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    n_nan = df_out["score_ml"].isna().sum()
    print(f"\nWrote {len(df_out)} predictions to {args.out_csv}")
    print(f"NaN predictions: {n_nan} / {len(df_out)}")

    # --- Timing summary ---
    n_structures_ok = sum(1 for t in per_structure_timing if t["status"] == "ok")
    prep_time_total = sum(t["total_s"] for t in per_structure_timing if t["status"] == "ok")

    print("\n=== Timing summary ===")
    print(f"Structures processed OK: {n_structures_ok} / {len(pdb_ids)}")
    print(f"Prep (download+clean+parse) total: {prep_time_total:.1f}s "
          f"({prep_time_total / max(n_structures_ok,1):.2f}s / structure)")
    print(f"Scoring total: {scoring_time:.1f}s")
    print(f"Scoring per residue: {scoring_time / max(n_residues_total,1):.3f}s")
    print(f"Scoring per substitution: {scoring_time / max(n_substitutions,1):.4f}s")
    print(f"End-to-end per gene ({args.gene}): {prep_time_total + scoring_time:.1f}s")

    timing_df = pd.DataFrame(per_structure_timing)
    timing_csv = os.path.splitext(args.out_csv)[0] + "_timing.csv"
    timing_df.to_csv(timing_csv, index=False)
    with open(timing_csv, "a") as f:
        f.write(f"\n# scoring_total_s,{scoring_time:.3f}\n")
        f.write(f"# scoring_per_residue_s,{scoring_time / max(n_residues_total,1):.4f}\n")
        f.write(f"# scoring_per_substitution_s,{scoring_time / max(n_substitutions,1):.5f}\n")
    print(f"Timing details written to {timing_csv}")


if __name__ == "__main__":
    main()
