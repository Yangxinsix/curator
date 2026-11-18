"""Summarize DPA4C screening, confirmation and transfer experiments from their saved measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
from tempfile import NamedTemporaryFile


NAMES = dict(aumgo="AuMgO", sasa="SA-SA", nacl="NaCl", sabs="SA-BS",
             cumulene="Cumulene", dimers="Dimers", sadma="SA-DMA")
ARMS = dict(A0="纯 DPA4C", A1="共享 RSA", A2="一个增益", A3="四组增益", A4="四组径向谱",
            B0="纯 DPA4C", B1="原分子 ACVR", B2="初始 β × 2", B3="主干冻结 256 步",
            B4="β × 2 加冻结", pure="纯 DPA4C", acvr="DPA4C + ACVR", les="DPA4C + LES")
SCREEN = dict(aumgo=[f"A{i}" for i in range(5)], sasa=[f"B{i}" for i in range(5)])
SEEDS = (3407, 3408, 3409)
FOLDS = ("validation", "test")
E, F = "energy_RMSE_meV_atom", "force_RMSE_meV_A"


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def measurements(summary, arm):
    mode = "pure" if arm in ("A0", "B0", "pure") else "les" if arm == "les" else "proposal"
    primary = summary[mode]
    clamp = summary.get("clamp_minus_proposal", {})
    newton = summary.get("newton_panel", {})
    return {
        E: primary[E], F: primary[F], "frames": summary["frames"],
        "uniform": summary.get("uniform"), "clamp_delta_energy_meV_atom": clamp.get(E),
        "clamp_delta_force_meV_A": clamp.get(F), "stationarity_RMS_V": summary.get("stationarity_RMS_V"),
        "proposal_transfer_RMS_e": summary.get("proposal_transfer_RMS_e"),
        "drive_RMS_V": newton.get("uniform_drive_RMS_V"),
        "newton": newton or None, "dipole": summary.get("dipole"),
    }


def endpoint(path, arm, fold):
    doc = read(path)
    if doc is None:
        return None
    if doc["split"] != fold:
        raise ValueError(f"Unexpected fold in {path}")
    ids = sorted(x["source_index"] for x in doc["frames"])
    if len(ids) != doc["summary"]["frames"] or len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"Missing or duplicate frame identities in {path}")
    return dict(measurements(doc["summary"], arm),
                source_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                checkpoint_sha256=doc["checkpoint_sha256"])


def scan(root):
    records, comparisons, decisions, bases = [], [], {}, {}
    for system in NAMES:
        path = root/system
        arms = SCREEN.get(system, ("pure", "acvr", "les"))
        for stage in ("select-test", "screen"):
            pick = read(root.parent/"dispatch"/"manifests"/f"{system}-{stage}.json")
            if pick:
                dec = pick["systems"][system]
                decisions[system] = dict(retained=dec["retained"], chosen_acvr=dec.get("chosen_acvr"))
                break
        for seed in SEEDS if system in SCREEN else SEEDS[:1]:
            for arm in arms:
                dest = path/str(seed)/arm
                if seed != SEEDS[0] and not dest.is_dir():
                    continue
                status, done = read(dest/"status.json") or {}, read(dest/"done.json") or {}
                folds = {f: endpoint(dest/(f + "_final.json"), arm, f) for f in FOLDS}
                state = ("独立测试已评分" if folds["test"] else "验证已评分" if folds["validation"]
                         else "训练已完成" if done else "有训练记录" if status or (dest/"config.json").exists()
                         else "无训练记录")
                row = dict(system=system, arm=arm, seed=seed, state=state,
                           joint_steps=done.get("updates", done.get("steps", status.get("step"))),
                           training_seconds=done.get("training_seconds", done.get("seconds", status.get("training_seconds"))),
                           **folds)
                curve = read(dest/"validation_curve.json")
                if curve:
                    row["validation_panel_curve"] = [
                        dict(joint_step=p["joint_step"], ef_step=p["ef_step"], training_seconds=p["training_seconds"],
                             **measurements(p["summary"], arm)) for p in curve["points"]]
                wall = read(dest/"validation_wall_curve.json")
                if wall:
                    row["wall_budget_validation"] = [
                        {k: v for k, v in p.items() if k != "summary"} |
                        (measurements(p["summary"], arm) if p.get("summary") else {}) for p in wall["points"]]
                records.append(row)
        base = path/str(SEEDS[0])/"base"
        bases[system] = dict(status=read(base/"status.json"), done=read(base/"done.json"))
    lookup = {(r["system"], r["arm"], r["seed"]): r for r in records}
    for system, arms in SCREEN.items():
        # Each intervention against pure; direct mechanism comparisons across ACVR arms.
        side = [("A2", "A1"), ("A3", "A2"), ("A4", "A3")] if system == "aumgo" else [
            ("B2", "B1"), ("B3", "B1"), ("B4", "B1")]
        for candidate, ref in [(a, arms[0]) for a in arms[1:]] + side:
            for fold in FOLDS:
                observations = []
                for seed in SEEDS:
                    a = lookup.get((system,candidate,seed), {}).get(fold)
                    b = lookup.get((system,ref,seed), {}).get(fold)
                    if a is None or b is None:
                        continue
                    if a["source_ids_sha256"] != b["source_ids_sha256"]:
                        raise ValueError(f"Unequal {fold} membership for {system} {candidate}/{ref}, seed {seed}")
                    observations.append(dict(seed=seed, **{E:a[E]-b[E], F:a[F]-b[F]}))
                if observations:
                    complete = len(observations)==len(SEEDS)
                    comparisons.append(dict(system=system, candidate=candidate, reference=ref, fold=fold,
                        paired_seeds=len(observations), paired_differences=observations,
                        three_seed_mean={k:mean(x[k] for x in observations) for k in (E,F)} if complete else None,
                        three_seed_sd={k:stdev(x[k] for x in observations) for k in (E,F)} if complete else None))
    return dict(records=records, paired_comparisons=comparisons, decisions=decisions, base_training=bases)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def cell(d, key):
    return fmt((d or {}).get(key))


def markdown(report):
    rows, decisions = report["records"], report["decisions"]
    lines = ["# DPA4C-Small 与 ACVR 实验结果", "",
             "能量和力均为 RMSE，单位分别为 meV/atom 和 meV/Å。表中横线表示无对应测量。进度来自已写入的训练和评分文件。",
             "", "## 首轮十组", "", "种子为 3407。验证和独立测试均取完整划分，只有入选组进入独立测试。", "",
             "| 体系 | 组别 | 已记录阶段 | 联合步数 | 验证 E | 验证 F | 测试 E | 测试 F |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    first = [r for r in rows if r["system"] in SCREEN and r["seed"] == SEEDS[0]]
    for r in first:
        a, v, t = r["arm"], r["validation"], r["test"]
        lines.append(f"| {NAMES[r['system']]} | {a} · {ARMS[a]} | {r['state']} | {fmt(r['joint_steps'],0)} | "
                     f"{cell(v,E)} | {cell(v,F)} | {cell(t,E)} | {cell(t,F)} |")
    for system in SCREEN:
        if system in decisions:
            dec = decisions[system]
            lines += ["", f"{NAMES[system]} 已选复验组 {', '.join(dec['retained'])}。"
                      + (f"进入独立测试的 ACVR 为 {dec['chosen_acvr']}。" if dec["chosen_acvr"] else "")]
    lines += ["", "## 电荷转移的贡献", "",
              "下表取种子 3407 的完整验证折。钳制将所有原子电荷固定为 Q/N，并重新计算能量及力。正的 ΔF 表示钳制增大了相对参考力的误差。",
              "", "| 体系 | 组别 | 正常 F | 钳制 F | ΔF | ΔE | 驻定 RMS（μV）| Newton 转移 RMS（e）| 相对 Newton 误差（%）|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in first:
        if r["arm"].endswith("0"):
            continue
        v = r["validation"] or {}
        n = v.get("newton") or {}
        residual = v.get("stationarity_RMS_V")
        lines.append(f"| {NAMES[r['system']]} | {r['arm']} | {cell(v,F)} | {cell(v.get('uniform'),F)} | "
                     f"{cell(v,'clamp_delta_force_meV_A')} | {cell(v,'clamp_delta_energy_meV_atom')} | "
                     f"{fmt(residual*1e6 if residual is not None else None,2)} | "
                     f"{fmt(n.get('equilibrium_transfer_RMS_e'),5)} | {cell(n,'mean_frame_relative_transfer_error_percent')} |")
    lines += ["", "## SA-SA 的驱动力与偶极", "",
              "驱动力取均匀电荷处的投影梯度，单位 mV。偶极取完整验证折，单位 D。原始符号用于分量相关和 RMSE。",
              "", "| 组别 | 驱动力 | 预测平均模长 | DFT 平均模长 | 分量 RMSE | 有符号分量 r |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in first:
        if r["system"] != "sasa" or r["arm"] == "B0":
            continue
        v = r["validation"] or {}
        d = v.get("dipole") or {}
        drive = v.get("drive_RMS_V")
        lines.append(f"| {r['arm']} | {fmt(drive*1000 if drive is not None else None)} | "
                     f"{cell(d,'mean_proposal_magnitude_D')} | {cell(d,'mean_DFT_magnitude_D')} | "
                     f"{cell(d,'component_RMSE_D')} | {cell(d,'signed_component_pearson_r')} |")
    lines += ["", "## 三种子配对差值", "",
              "差值为候选组减去对照组，负值表示误差下降。三个种子共用同一份预热主干。均值及样本标准差只在 3407、3408、3409 三对结果齐全后计算。",
              "", "| 体系 | 候选 − 对照 | 数据 | 已有配对 | 三个种子的 ΔF | 平均 ΔF ± SD | 平均 ΔE ± SD |",
              "|---|---|---|---:|---|---|---|"]
    for comp in report["paired_comparisons"]:
        vals = comp["paired_differences"]
        by_seed = "，".join(f"{p['seed']} {fmt(p[F])}" for p in vals)
        mu, sd = comp["three_seed_mean"], comp["three_seed_sd"]
        def measure(key):
            return f"{fmt(mu[key])} ± {fmt(sd[key])}" if mu else "—"
        phase = "验证" if comp["fold"] == "validation" else "独立测试"
        lines.append(f"| {NAMES[comp['system']]} | {comp['candidate']} − {comp['reference']} | {phase} | "
                     f"{len(vals)}/3 | {by_seed} | {measure(F)} | {measure(E)} |")
    lines += ["", "## 扩展五体系", "", "三个独立训练的模型均使用种子 3407。", "",
              "| 体系 | 模型 | 已记录阶段 | 联合步数 | 验证 E | 验证 F | 测试 E | 测试 F |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["system"] in SCREEN:
            continue
        v,t = r["validation"],r["test"]
        lines.append(f"| {NAMES[r['system']]} | {ARMS[r['arm']]} | {r['state']} | {fmt(r['joint_steps'],0)} | "
                     f"{cell(v,E)} | {cell(v,F)} | {cell(t,E)} | {cell(t,F)} |")
    lines += ["", "逐种子完整终点评分、固定验证子集的训练曲线、同时间预算数据及 GPU 型号见同目录的 results.json。"]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path("runs"))
    parser.add_argument("--out",type=Path,default=Path("summary"))
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report=scan(args.runs)
    for name,payload in (("results.json",json.dumps(report,allow_nan=False,indent=2)+"\n"),
                         ("REPORT.md",markdown(report))):
        with NamedTemporaryFile(mode="w",dir=args.out,delete=False) as f:
            f.write(payload)
            tmp=f.name
        os.replace(tmp,args.out/name)
    print(args.out/"REPORT.md")


if __name__=="__main__":
    main()
