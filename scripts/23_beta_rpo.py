#!/usr/bin/env python3
"""The beta-sensitivity and RPO studies, analysed as they were declared.

Both studies were declared in ``sentalign.plan`` (``beta_sensitivity``, ``rpo_comparison``)
on 28 September 2026, before any of their runs, together with the comparisons computed here:

B   descriptive. LFM2.5, seeds 13, 21 and 34, at beta 0.03 and 0.3, with the paper's own
    beta = 0.1 runs restricted to the same seeds as the reference row: KTO and R-DPO against
    SFT, and each anchored arm against its unanchored twin at the same beta, on AUROC,
    E-AURC and accuracy. Seed-paired mean difference, 95 percent interval, uncorrected p and
    the number of seeds that agree in sign with the mean.
R1  the declared family, Holm over its four tests, seed-paired t on AUROC: R-DPO+RPO against
    R-DPO, and R-DPO+reg against R-DPO+RPO, on LFM2.5 and Qwen3.5, five seeds.
R2  descriptive: the same pairs on E-AURC and accuracy, and R-DPO+RPO against SFT on all
    three metrics.

Every metric is the per-seed mean over the four DynaSent sets (scripts/06), on the exact
order of c. A declared comparison without its runs stops the analysis unless ``--interim``,
because a family that shrinks silently is corrected less strictly than the one declared.
An RPO run is admitted only if its manifest shows the term was applied: ``rpo_alpha`` 1.0
and at least one training log checked for TRL's ``nll_loss``. A run whose batch plan differs
from the arm it is compared with (the OOM backoff halves the micro-batch) is reported, not
dropped: the effective batch is the same, and the reader is told which cells it touches.

    python scripts/23_beta_rpo.py --runs runs_final --out results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentalign.config import SEEDS                                    # noqa: E402
from sentalign.modeling import display_name                          # noqa: E402

ENDPOINT = "mean4"
METRICS = (("auroc", "AUROC", True), ("eaurc", "E-AURC", False), ("acc", "Acc", True))

BETA_MODEL = "lfm-1.2b"
BETA_SEEDS = (13, 21, 34)
#: 0.1 is the paper's own setting, whose runs already exist; the study adds the other two.
BETAS = (0.03, 0.1, 0.3)
DECLARED_BETAS = (0.03, 0.3)

RPO_MODELS = ("lfm-1.2b", "qwen-2b")
RPO_ARM = "rdpo-rpo1.0"
R1 = ((RPO_ARM, "rdpo"), ("rdpo-dreg1.0", RPO_ARM))
R2_EXTRA = ((RPO_ARM, "sft"),)

LABELS = {"sft": "SFT", "kto": "KTO", "rdpo": "R-DPO", "kto-dreg1.0": "KTO+reg",
          "rdpo-dreg1.0": "R-DPO+reg", RPO_ARM: "R-DPO+RPO"}


def _script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def beta_arm(objective: str, beta: float, anchored: bool) -> str:
    """The directory name of an arm at a beta: the paper's runs carry no beta suffix."""
    base = objective if beta == 0.1 else f"{objective}-beta{beta}"
    return base + ("-dreg1.0" if anchored else "")


def beta_contrasts(beta: float):
    return ((beta_arm("kto", beta, False), "sft"), (beta_arm("rdpo", beta, False), "sft"),
            (beta_arm("kto", beta, True), beta_arm("kto", beta, False)),
            (beta_arm("rdpo", beta, True), beta_arm("rdpo", beta, False)))


def label(arm: str) -> str:
    base, _, beta = arm.partition("-beta")
    beta, _, reg = beta.partition("-dreg")
    name = LABELS.get(base + ("-dreg1.0" if reg else ""), arm)
    return name


def restrict(mean: dict, model: str, seeds) -> dict:
    """Each arm of ``model`` on ``seeds`` only, so every row is averaged over the same seeds."""
    return {key: {s: v for s, v in by_seed.items() if s in seeds}
            for key, by_seed in mean.items() if key[0] == model}


def compare(mean, model, first, second, metric, *, stats):
    row = stats.paired(mean, model, ENDPOINT, first, second, metric)
    if row is None:
        return None
    a, b = mean[(model, ENDPOINT, first)], mean[(model, ENDPOINT, second)]
    diffs = [a[s][metric] - b[s][metric] for s in row["seeds"]]
    row["agree"] = int(sum(np.sign(d) == np.sign(row["mean_diff"]) for d in diffs))
    return row


def beta_rows(mean, *, stats, interim=False) -> tuple[list, list]:
    """Section B: descriptive, uncorrected, on the three seeds the study declared."""
    local = restrict(mean, BETA_MODEL, BETA_SEEDS)
    rows, missing = [], []
    for beta in BETAS:
        for first, second in beta_contrasts(beta):
            for metric, _, _ in METRICS:
                row = compare(local, BETA_MODEL, first, second, metric, stats=stats)
                if row is None or row["n"] != len(BETA_SEEDS):
                    missing.append(f"beta {beta}: {first} vs {second} {metric}")
                    continue
                rows.append(row | {"beta": beta})
    if missing and not interim:
        raise SystemExit("declared comparisons without their runs:\n  " + "\n  ".join(missing))
    return rows, missing


def rpo_rows(mean, *, stats, selective, interim=False) -> tuple[dict, list]:
    """R1, Holm over its four declared tests, and R2, uncorrected."""
    rows, missing = {"R1": [], "R2": []}, []
    for model in RPO_MODELS:
        local = restrict(mean, model, SEEDS)
        pairs = [(pair, metric, "R1" if metric == "auroc" else "R2")
                 for pair in R1 for metric, _, _ in METRICS]
        pairs += [(pair, metric, "R2") for pair in R2_EXTRA for metric, _, _ in METRICS]
        for (first, second), metric, family in pairs:
            row = compare(local, model, first, second, metric, stats=stats)
            if row is None or row["n"] != len(SEEDS):
                missing.append(f"{model}: {first} vs {second} {metric}")
                continue
            rows[family].append(row)
    if missing and not interim:
        raise SystemExit("declared comparisons without their runs:\n  " + "\n  ".join(missing))
    adjusted = selective.holm({i: r["p"] for i, r in enumerate(rows["R1"])})
    for i, row in enumerate(rows["R1"]):
        row["p_holm"] = adjusted[i]
        row["family_size"] = len(rows["R1"])
    return rows, missing


def beta_confidence(runs: Path, *, mech=None) -> dict:
    """How confident each beta arm is and how its confidence follows agreement.

    The inflation and agreement-sensitivity readouts of scripts/22 (median log-odds of c,
    and the Spearman correlation of c with annotator agreement on correct answers), for
    every arm of section B on the declared seeds. Descriptive: they say how the change in
    ranking shows up, and enter no comparison.
    """
    mech = mech or _script("mechanism", "22_mechanism.py")
    arms = ["sft"] + sorted({arm for beta in BETAS for pair in beta_contrasts(beta)
                             for arm in pair if arm != "sft"})
    means = mech.collect(runs, models=(BETA_MODEL,), arms=tuple(arms), seeds=BETA_SEEDS)
    return {key.split("|", 1)[1]: {k: v[k] for k in ("auroc", "log_odds", "sensitivity")}
            for key, v in means.items()}


def rpo_manifest_problems(runs: Path) -> list[str]:
    """RPO runs whose manifest does not show the term in the loss."""
    problems = []
    for model in RPO_MODELS:
        for seed in SEEDS:
            run = runs / f"main__{model}__{RPO_ARM}__eps0.0__tau0.2__n8000__s{seed}"
            if not (run / "eval").exists():
                continue                        # not scored yet: counted as missing instead
            path = run / "manifest.json"
            record = json.loads(path.read_text()) if path.exists() else {}
            if record.get("rpo_alpha") != 1.0 or not record.get("rpo_nll_logs"):
                problems.append(f"{run.name}: rpo_alpha={record.get('rpo_alpha')}, "
                                f"rpo_nll_logs={record.get('rpo_nll_logs')}")
    return problems


def batch_plan_differences(runs: Path, comparisons) -> list[str]:
    """Compared cells whose runs trained with different micro-batches."""
    def plan(model, arm, seed):
        path = runs / f"main__{model}__{arm}__eps0.0__tau0.2__n8000__s{seed}" / "manifest.json"
        return json.loads(path.read_text()).get("batch_plan") if path.exists() else None

    out = set()
    for model, first, second, seeds in comparisons:
        for seed in seeds:
            a, b = plan(model, first, seed), plan(model, second, seed)
            if a and b and a != b:
                out.add(f"{model} s{seed}: {first} {a} vs {second} {b}")
    return sorted(out)


def _p(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def _diff(row, ci=False) -> str:
    text = f"{row['mean_diff']:+.3f}"
    if ci:
        text += f" [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]"
    return text


DIRECTION = ("$\\uparrow$: higher is better and $\\downarrow$: lower is better, so a "
             "positive difference in AUROC or accuracy and a negative one in E-AURC favour "
             "the first-named arm.")


def beta_table(rows) -> str:
    by = {(r["beta"], r["first"], r["second"], r["metric"]): r for r in rows}
    body = []
    for i, beta in enumerate(BETAS):
        if i:
            body.append(r"\addlinespace")
        for j, (first, second) in enumerate(beta_contrasts(beta)):
            cells = by.get((beta, first, second, "auroc"))
            if cells is None:
                continue
            e, a = by[(beta, first, second, "eaurc")], by[(beta, first, second, "acc")]
            n = cells["n"]
            body.append(" & ".join([
                f"{beta:g}" if j == 0 else "", f"{label(first)} vs {label(second)}",
                _diff(cells, ci=True), _p(cells["p"]), f"{cells['agree']}/{n}",
                f"{_diff(e)} ({e['agree']}/{n})", f"{_diff(a)} ({a['agree']}/{n})"]) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{Sensitivity to the preference temperature $\\beta$, "
        f"{display_name(BETA_MODEL)}, seeds {', '.join(map(str, BETA_SEEDS))}. Every arm "
        "of the paper trains at $\\beta = 0.1$; those rows are the paper's runs restricted "
        "to the same three seeds. Per-seed mean over the four evaluation sets; seed-paired "
        "mean difference with its 95\\% interval and uncorrected $p$ for AUROC, and for "
        "every metric the number of seeds that agree in sign with the mean. The anchor "
        "keeps $\\lambda = 1$ at every $\\beta$. Declared before the runs; descriptive, "
        f"outside the declared families. {DIRECTION}}}",
        r"\label{tab:beta}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"$\beta$ & Comparison & $\Delta$AUROC$\uparrow$ [95\% CI] & $p$ & Seeds & "
        r"$\Delta$E-AURC$\downarrow$ & $\Delta$Acc$\uparrow$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def rpo_table(rows) -> str:
    body = []
    for i, model in enumerate(RPO_MODELS):
        if i:
            body.append(r"\addlinespace")
        # The model as a panel row: a model column costs more width than the page has.
        body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{display_name(model)}}}}} \\\\")
        mine = [r for r in rows["R1"] + rows["R2"] if r["model"] == model]
        for first, second in R1 + R2_EXTRA:
            cell = {r["metric"]: r for r in mine
                    if (r["first"], r["second"]) == (first, second)}
            if "auroc" not in cell:
                continue
            auroc = cell["auroc"]
            p = (_p(auroc["p_holm"]) + r"$^{\dagger}$") if "p_holm" in auroc else _p(auroc["p"])
            body.append(" & ".join([
                f"{label(first)} vs {label(second)}", _diff(auroc, ci=True), p,
                f"{_diff(cell['eaurc'])} ({_p(cell['eaurc']['p'])})",
                f"{_diff(cell['acc'])} ({_p(cell['acc']['p'])})"]) + r" \\")
    size = len(rows["R1"])
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        "\\caption{The anchor against TRL's own likelihood term. R-DPO+RPO adds, with "
        "\\texttt{rpo\\_alpha} $= 1$, the negative log-likelihood of the preferred "
        "completion, averaged over its two tokens (the label and the end of sequence), to "
        "every pair's R-DPO loss. Per-seed mean over the four evaluation "
        "sets, five seeds, seed-paired $t$. $^{\\dagger}$: the declared family, "
        f"Holm-adjusted over its {size} tests; every other $p$ is uncorrected. "
        f"{DIRECTION}}}",
        r"\label{tab:rpo}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Comparison & $\Delta$AUROC$\uparrow$ [95\% CI] & $p$ & "
        r"$\Delta$E-AURC$\downarrow$ ($p$) & $\Delta$Acc$\uparrow$ ($p$) \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs_final"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--interim", action="store_true",
                    help="report what has run; the declared families may be incomplete")
    args = ap.parse_args(argv)
    stats = _script("regulariser_stats", "06_regulariser_stats.py")
    selective = _script("selective", "05_selective.py")

    problems = rpo_manifest_problems(args.runs)
    if problems:
        raise SystemExit("RPO runs whose manifest does not show the term:\n  "
                         + "\n  ".join(problems))
    scores, items, excluded = stats.load(args.runs, set(RPO_MODELS) | {BETA_MODEL})
    for (model, name), digests in sorted(items.items()):
        if len(digests) != 1:
            raise SystemExit(f"{model} {name}: runs scored {len(digests)} different item sets")
    mean = stats.average_over_sets(scores)

    beta, beta_missing = beta_rows(mean, stats=stats, interim=args.interim)
    rpo, rpo_missing = rpo_rows(mean, stats=stats, selective=selective, interim=args.interim)
    compared = [(BETA_MODEL, f, s, BETA_SEEDS) for b in DECLARED_BETAS
                for f, s in beta_contrasts(b)]
    compared += [(m, f, s, SEEDS) for m in RPO_MODELS for f, s in R1 + R2_EXTRA]
    result = {"interim": bool(beta_missing or rpo_missing), "excluded": excluded,
              "missing": beta_missing + rpo_missing,
              "batch_plan_differences": batch_plan_differences(args.runs, compared),
              "beta": beta, "rpo": rpo}
    if not beta_missing:
        result["beta_confidence"] = beta_confidence(args.runs)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "beta_rpo.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "table_beta.tex").write_text(beta_table(beta) + "\n")
    (args.out / "table_rpo.tex").write_text(rpo_table(rpo) + "\n")
    for row in beta:
        print(f"  beta {row['beta']:<5g} {row['first']:22} vs {row['second']:14} "
              f"{row['metric']:5} d={row['mean_diff']:+.4f} p={row['p']:.4f} "
              f"agree {row['agree']}/{row['n']}")
    for family in ("R1", "R2"):
        for row in rpo[family]:
            tail = f" holm={row['p_holm']:.4f}" if "p_holm" in row else ""
            print(f"  {family} {row['model']:10} {row['first']:13} vs {row['second']:13} "
                  f"{row['metric']:5} d={row['mean_diff']:+.4f} p={row['p']:.4f}{tail} "
                  f"agree {row['agree']}/{row['n']}")
    for arm, v in result.get("beta_confidence", {}).items():
        print(f"  confidence {arm:22} AUROC={v['auroc']:.3f} median log-odds="
              f"{v['log_odds']:5.1f} rho(c, agreement | correct)={v['sensitivity']:.3f}")
    for line in result["batch_plan_differences"]:
        print(f"  batch plan differs: {line}")
    if result["missing"]:
        print(f"  INTERIM: {len(result['missing'])} declared comparisons have no runs yet")
    print(f"wrote {args.out / 'table_beta.tex'} and {args.out / 'table_rpo.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
