"""Summarize matched endpoints only; print pending arms without treating them as zeros."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean


def read(path):
    return json.loads(path.read_text())


def radial(subset):
    ref = [float(r["F_radial_ref"]) for r in subset]
    err = [float(r["radial_error"]) for r in subset]
    mse = fmean(x*x for x in err)
    ref_ms = fmean(x*x for x in ref)
    return dict(n=len(ref), rmse_meV_A=1000*math.sqrt(mse),
                skill_vs_zero=1-mse/ref_ms if ref_ms else None)


def summarize(root):
    result = {}
    for system in ("nacl","aumgo","cumulene","sasa","sabs","sadma","dimers"):
        records, keys = {}, {}
        for arm in ("baseline","gate","direction"):
            run = root/system/arm
            if not (run/"complete.json").is_file():
                records[arm] = {"pending":True}
                continue
            final = read(run/"test_final.json")
            valid = read(run/"validation_final.json")
            ident = final["identity"]
            assert final["checkpoint_sha256"] == valid["checkpoint_sha256"]
            assert ident["arm"] == arm and ident["system"] == system
            keys[arm] = tuple(ident[k] for k in ("source_checkpoint_sha256","source_resume_sha256",
                                                  "split_sha256","seed","budget")) + (
                tuple(r["source_index"] for r in final["frames"]), tuple(r["source_index"] for r in valid["frames"]))
            rec = records[arm] = {"step":final["joint_step"],"training_seconds":final["training_seconds"]}
            for label, report in (("validation",valid),("test",final)):
                d = report["summary"]
                rec[label] = {**d["proposal"],"rms_V":d["stationarity_RMS_V"],
                              "q_vs_newton_pct":d["newton_panel"]["mean_frame_relative_transfer_error_percent"],
                              "newton_certified":d["newton_panel"]["certified"],
                              "uniform_q":d["uniform"]}
            if (run/"timing.json").exists():
                rec["timing"] = read(run/"timing.json")
            if system == "cumulene":
                rec["high_angle"] = read(run/"torsion/high_angle.json")
            if system in ("sasa","sabs","sadma","dimers"):
                with (run/"fragments/heldout_radial.csv").open() as fp:
                    rows = list(csv.DictReader(fp))
                assert {int(r["source_index"]) for r in rows} == set(keys[arm][-2])
                rec["radial_all"] = radial(rows)
                far = [r for r in rows if r.get("outside_cutoff")=="True"]
                if far:
                    rec["radial_disconnected"] = radial(far)
                curves = read(run/"fragments/summary.json").get("curves")
                if curves:
                    rec["labeled_dimer_curves"] = curves
        if keys and any(k != next(iter(keys.values())) for k in keys.values()):
            raise ValueError(f"{system}: incompatible checkpoint, split, budget or frame panel")
        base = records["baseline"]
        if "test" in base:
            for arm, value in records.items():
                if arm != "baseline" and "test" in value:
                    value["test_change_vs_baseline"] = {
                        k:value["test"][k]-base["test"][k]
                        for k in ("energy_RMSE_meV_atom","force_RMSE_meV_A","rms_V")}
        result[system] = records
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", type=Path)
    args = ap.parse_args()
    if not args.runs.is_dir():
        ap.error("runs directory does not exist")
    print(json.dumps(summarize(args.runs), indent=2, allow_nan=False))
