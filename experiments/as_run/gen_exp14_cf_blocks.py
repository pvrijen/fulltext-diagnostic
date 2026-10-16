r"""Render the LaTeX blocks for the cross-fitted second encoder family (exp14) from ../../2.8_jamia/exp14_stats.json.

Writes ../tables/exp14_cf_blocks.tex with three snippets separated by marker lines:
  %% TAB3    : the complete main-text Table 3 (PFD controls and replication, both families, both metrics, seed-averaged)
  %% PERSEED : Supplementary table with the primary family's per-seed values (moved out of Table 3)
  %% SUPTAB  : Supplementary Table S9 (second family, complete) with the cross-fitted PFD column
Primary-family numbers are the published ones (exp11; Table 3 of v2.9 at 7519d47); second-family numbers come from the JSON.
Conventions: $+$0.074 sign macro, \sd{.xxx}, no leading zero in P, P floor <.001.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE.parent.parent / "2.8_jamia" / "exp14_stats.json"))
assert S["n_folds"] == 30, S["n_folds"]
DS = {"pfas": "PFAS", "bpa": "BPA", "cd012661": "CD012661", "cd012069": "CD012069", "cd008122": "CD008122", "cd011431": "CD011431"}


def sg(x, nd=3):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0.0:
        return s
    return ("$-$" if x < 0 else "$+$") + s


def sd(x):
    return "\\sd{" + f"{x:.3f}"[1:] + "}"


def ci(e):
    return f"{sg(e['ci'][0])} to {sg(e['ci'][1])}"


def pfmt(p):
    if p < 0.001:
        return "$<$.001"
    s = f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    return s[1:] if s.startswith("0") else s


def val(x):
    return f"{x:.3f}"


P, W = S["prauc"], S["wss95"]
L = []

# ---------------- Table 3 (main text) ----------------
L.append("%% TAB3")
L.append(r"\begin{table}[t]")
L.append(r"	\caption{\textbf{PFD controls and replication.} Gain over the abstract baseline for the cross-fitted PFD student, the hard-label control (same epochs from the same checkpoints, $\alpha = 0$), and their paired difference, in both encoder families and on both metrics. Primary family: mean over three student seeds (per-seed values in Supplementary \Cref{tab:controls-seeds}); second family: one seed.}\label{tab:controls}")
L.append(r"	\centering")
L.append(r"	\small")
L.append(r"	\resizebox{\columnwidth}{!}{%")
L.append(r"	\begin{tabular}{@{}llrlr@{}}")
L.append(r"		\toprule")
L.append(r"		 & \multicolumn{2}{c}{\textbf{PR-AUC}} & \multicolumn{2}{c}{\textbf{WSS@95}} \\")
L.append(r"		\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
L.append(r"		\textbf{Arm} & \textbf{Gain (95\% CI)} & $P$ & \textbf{Gain (95\% CI)} & $P$ \\")
L.append(r"		\midrule")
L.append(r"		\multicolumn{5}{@{}l}{\textit{BioMedBERT (primary pipeline), three student seeds}} \\")
L.append(r"		PFD student ($\alpha = 0.1$) & $+$0.015 ($+$0.006 to $+$0.028) & $<$.001 & $+$0.056 ($+$0.027 to $+$0.090) & $<$.001 \\")
L.append(r"		Hard-label control ($\alpha = 0$) & $+$0.003 ($-$0.005 to $+$0.012) & .45 & $-$0.028 ($-$0.071 to $+$0.014) & .02 \\")
L.append(r"		Soft minus hard (paired) & $+$0.012 ($-$0.002 to $+$0.030) & .10 & $+$0.083 ($+$0.040 to $+$0.133) & $<$.001 \\")
L.append(r"		\midrule")
L.append(r"		\multicolumn{5}{@{}l}{\textit{BiocModernBERT (second family), one seed}} \\")
L.append(f"		PFD student ($\\alpha = 0.1$) & {sg(P['CF']['mean'])} ({ci(P['CF'])}) & {pfmt(P['CF']['p_wilcoxon'])} & {sg(W['CF']['mean'])} ({ci(W['CF'])}) & {pfmt(W['CF']['p_wilcoxon'])} \\\\")
L.append(f"		Hard-label control ($\\alpha = 0$) & {sg(P['HARD']['mean'])} ({ci(P['HARD'])}) & {pfmt(P['HARD']['p_wilcoxon'])} & {sg(W['HARD']['mean'])} ({ci(W['HARD'])}) & {pfmt(W['HARD']['p_wilcoxon'])} \\\\")
L.append(f"		Soft minus hard (paired) & {sg(P['CF_minus_hard']['mean'])} ({ci(P['CF_minus_hard'])}) & {pfmt(P['CF_minus_hard']['p_boot'])} & {sg(W['CF_minus_hard']['mean'])} ({ci(W['CF_minus_hard'])}) & {pfmt(W['CF_minus_hard']['p_boot'])} \\\\")
L.append(r"		\bottomrule")
L.append(r"	\end{tabular}}")
L.append(r"	\par\smallskip")
L.append(r"	\noindent\footnotesize{Gain = mean paired fold-level difference over 30 folds (primary family: averaged over seeds 42, 123, and 456) with two-stage cluster-bootstrap 95\% CI. $P$ = pooled two-sided Wilcoxon ($n = 30$) for the student and control rows; for the paired soft-minus-hard rows the two-stage cluster-bootstrap $P$ (the 90 primary-family seed-folds are not independent). Both families use the same folds and distillation settings; in-sample values are in Supplementary \Cref{tab:stats-reanalysis} and \Cref{tab:second-family}.}")
L.append(r"\end{table}")

# ---------------- Per-seed supplementary table (primary family; values from v2.9 Table 3) ----------------
L.append("%% PERSEED")
L.append(r"\begin{table}[H]")
L.append(r"	\caption{\textbf{PFD student and hard-label control per student seed (primary pipeline).} Gain in PR-AUC and WSS@95 over $\Mabs$ for the cross-fitted student and the hard-label control under three student seeds (same folds and soft labels); the seed-averaged values are in \Cref{tab:controls}.}\label{tab:controls-seeds}")
L.append(r"	\centering")
L.append(r"	\small")
L.append(r"	\begin{tabular}{@{}lrrr@{}}")
L.append(r"		\toprule")
L.append(r"		 & \multicolumn{3}{c}{\textbf{Seed}} \\")
L.append(r"		\cmidrule(lr){2-4}")
L.append(r"		\textbf{Arm} & \textbf{42} & \textbf{123} & \textbf{456} \\")
L.append(r"		\midrule")
L.append(r"		\multicolumn{4}{@{}l}{\textit{PR-AUC}} \\")
L.append(r"		PFD student ($\alpha = 0.1$) & $+$0.016\sd{.028} & $+$0.017\sd{.037} & $+$0.013\sd{.027} \\")
L.append(r"		Hard-label control ($\alpha = 0$) & $+$0.004\sd{.027} & $+$0.006\sd{.030} & 0.000\sd{.024} \\")
L.append(r"		\midrule")
L.append(r"		\multicolumn{4}{@{}l}{\textit{WSS@95}} \\")
L.append(r"		PFD student ($\alpha = 0.1$) & $+$0.035\sd{.081} & $+$0.078\sd{.165} & $+$0.053\sd{.133} \\")
L.append(r"		Hard-label control ($\alpha = 0$) & $-$0.034\sd{.115} & $-$0.030\sd{.193} & $-$0.020\sd{.104} \\")
L.append(r"		\bottomrule")
L.append(r"	\end{tabular}")
L.append(r"	\par\smallskip")
L.append(r"	\noindent\footnotesize{Cells: mean $\pm$SD of the paired fold-level gain over 30 folds (ddof = 1). Seed 42 is a rerun of the published configuration (\Cref{tab:decomposition}, $+$0.017 PR-AUC); it reproduced the published fold-level gains within $-$0.001. Cluster-robust $t$ for the seed-averaged PR-AUC gain: $P = .045$.}")
L.append(r"\end{table}")

# ---------------- Supplementary Table S9 (second family, complete, with cross-fitted PFD) ----------------
L.append("%% SUPTAB")
L.append(r"\begin{table*}[t]")
L.append(r"	\caption{\textbf{Second encoder family: training vs.\ inference decomposition, fourth cell, PFD (cross-fitted and in-sample), and hard-label control for BiocModernBERT (8,192 tokens), on PR-AUC and WSS@95.} "
         r"One truncated classifier reads 512 tokens ($\Mabs$, $\Mfta$, students) or 8,192 tokens ($\Aft$, $\Mft$); no chunk pooling. "
         r"$\Mabs$ = trained and tested on abstracts; $\Aft$ = abstract-trained model tested on full text (fourth cell); $\Mfta$ = full-text-trained model tested on abstracts; "
         r"$\Mft$ = full-text-trained model tested on full text; $\Mfta^{512}$ = full-text-trained model tested on full text truncated to its first 512 tokens (length-matched check, "
         f"{P['length_check']['n_folds']} folds; the first folds ran before this evaluation was added); "
         r"$\Mpfd$ = abstract student after cross-fitted PFD; IS = the same student with in-sample soft labels; Hard = same fine-tuning with hard labels only. "
         r"TE $= \Mfta - \Mabs$; IE $= \Mft - \Mfta$; IE$' = \Aft - \Mabs$; TE$' = \Mft - \Aft$; $\Delta$PFD $= \Mpfd - \Mabs$; $\Delta$IS $=$ IS $- \Mabs$; $\Delta$Hard $=$ Hard $- \Mabs$.}\label{tab:second-family}")
L.append(r"	\centering")
L.append(r"	\small")
L.append(r"	\resizebox{\textwidth}{!}{%")
L.append(r"	\begin{tabular}{@{}lrrrrrrrr rrrrrrr@{}}")
L.append(r"		\toprule")
L.append(r"		\textbf{Dataset} & $\Mabs$ & $\Aft$ & $\Mfta$ & $\Mfta^{512}$ & $\Mft$ & $\Mpfd$ & IS & Hard & \textbf{TE} & \textbf{IE} & \textbf{IE$'$} & \textbf{TE$'$} & \textbf{$\Delta$PFD} & \textbf{$\Delta$IS} & \textbf{$\Delta$Hard} \\")
CELLS = ("C", "Aft", "P0", "P0trunc", "P100", "CF", "IS", "HARD")
BEST = ("C", "P0", "P100", "CF")
EFF = ("TE", "IE", "IEp", "TEp", "CF", "IS", "HARD")
for metric, label in (("prauc", "PR-AUC"), ("wss95", "WSS@95")):
    m = S[metric]; c = m["cells"]
    L.append(r"		\midrule")
    L.append(f"		\\multicolumn{{16}}{{@{{}}l}}{{\\textit{{{label}}}}} \\\\")
    for d, name in DS.items():
        cells = {k: c[k]["per_dataset"].get(d) for k in CELLS}
        best = max(BEST, key=lambda k: cells[k])
        def fmt(k):
            v = cells[k]
            if v is None:
                return "--"
            return ("\\textbf{" + val(v) + "}") if k == best else val(v)
        row = f"		{name} & " + " & ".join(fmt(k) for k in CELLS)
        row += " & " + " & ".join(sg(m[k]["per_dataset"][d]) for k in EFF) + " \\\\"
        L.append(row)
    L.append(r"		\midrule")
    means = {k: c[k]["mean"] for k in CELLS}
    bestm = max(BEST, key=lambda k: means[k])
    mrow = "		Mean & " + " & ".join(("\\textbf{" + val(means[k]) + "}") if k == bestm else val(means[k]) for k in CELLS)
    mrow += " & " + " & ".join(sg(m[k]["mean"]) + sd(m[k]["sd"]) for k in EFF) + " \\\\"
    L.append(mrow)
    L.append("		95\\% CI & & & & & & & & & " + " & ".join(ci(m[k]) for k in EFF) + " \\\\")
    L.append("		Wilcoxon $P$, $n = 30$ & & & & & & & & & " + " & ".join(pfmt(m[k]["p_wilcoxon"]) for k in EFF) + " \\\\")
    L.append("		Datasets with effect $> 0$ & & & & & & & & & " + " & ".join(f"{m[k]['n_datasets_pos']}/{m[k]['n_datasets']}" for k in EFF) + " \\\\")
    lc = m["length_check"]
    L.append(f"		\\multicolumn{{16}}{{@{{}}l}}{{\\footnotesize Length-matched check ({lc['n_folds']} folds): $\\Mfta^{{512}} - \\Mfta$ (format) {sg(lc['format_component_trunc_minus_abs']['mean'])} ({ci(lc['format_component_trunc_minus_abs'])}); "
             f"$\\Mft - \\Mfta^{{512}}$ (length) {sg(lc['length_component_ft_minus_trunc']['mean'])} ({ci(lc['length_component_ft_minus_trunc'])}).}} \\\\")
    inter, smh, cfis, par = m["Int"], m["CF_minus_hard"], m["CF_minus_IS"], m["CF_vs_P100"]
    L.append(f"		\\multicolumn{{16}}{{@{{}}l}}{{\\footnotesize Interaction TE$' -$ TE $=$ {sg(inter['mean'])} ({ci(inter)}; {inter['n_pos_folds']}/{inter['n_folds']} folds); "
             f"cross-fitted soft minus hard {sg(smh['mean'])} ({ci(smh)}; cluster-bootstrap $P$ {pfmt(smh['p_boot'])}; {smh['n_datasets_pos']}/{smh['n_datasets']} datasets); "
             f"cross-fitted minus in-sample {sg(cfis['mean'])} ({ci(cfis)}); $\\Mpfd - \\Mft$ {sg(par['mean'])} ({ci(par)}).}} \\\\")
L.append(r"		\bottomrule")
L.append(r"	\end{tabular}}")
L.append(r"	\par\smallskip")
L.append(r"	\noindent\footnotesize{Bold = best value per row across $\Mabs$, $\Mfta$, $\Mft$, and $\Mpfd$. Dataset rows are five-fold means; mean rows pool 30 folds; effect columns carry $\pm$SD of the paired fold-level difference; "
         r"95\% CI = two-stage cluster bootstrap (datasets then folds, 20,000 replicates); Wilcoxon = pooled two-sided test over 30 folds. "
         r"Same folds, hyperparameters, and distillation settings as the BioMedBERT pipeline (\Cref{tab:hp}); the full-text teachers used batch 8 $\times$ 8 accumulation, bf16, gradient checkpointing. "
         r"Cross-fitted soft labels from five inner teachers per fold (Supplementary \Cref{app:crossfit}); in-sample soft labels from the teacher trained on the full training split.}")
L.append(r"\end{table*}")

(HERE.parent / "tables").mkdir(exist_ok=True)
(HERE.parent / "tables" / "exp14_cf_blocks.tex").write_text("\n".join(L) + "\n")
print("written", HERE.parent / "tables" / "exp14_cf_blocks.tex", len(L), "lines")
