# Scripts as run for the paper

These are the exact scripts used for the analyses added after the first manuscript version: the fourth cell of the two-by-two design, the hard-label control and student-seed replicate, the second encoder family (BiocModernBERT) with its cross-fitting, the inverse-route control, the soft-label memorisation check, and the statistical summaries behind the tables. They are kept verbatim for provenance.

Unlike the standalone scripts in the parent directory, they were run inside the authors' thesis codebase and keep its conventions: absolute paths on the authors' server, data loading through `run_rq2_unified.load_data`, `make_cv_splits` and `make_val_split` (seed 42; the resulting outer folds are the ones in `../fold_indices.json`), and the `belt_plus` package in this repository for the models. To rerun them, point the path constants at your copies of the datasets and result folders. The standalone equivalents for the primary pipeline remain `../run_crossfit.py` and companions.

| Script | Paper artifact | Reads | Writes |
|---|---|---|---|
| `exp11_alpha0_seeds.py` | Table 3 primary-family rows, Supplementary Table S8 (hard-label control, alpha = 0; student seeds 123 and 456) | cached cross-fitted teacher logits of the primary pipeline | one JSON per seed, dataset and fold |
| `exp12_factorial_cell.py` | Table 2 fourth cell (pooled architecture trained on abstracts, tested on abstracts and full text), Figure 2, Appendix H | datasets, fold indices | one JSON per dataset and fold |
| `exp14_biocmodernbert_pfd.py` | Second encoder family: Table 3 second-family rows, Supplementary Table S11, Appendix N (two-by-two cells, in-sample and cross-fitted PFD, hard-label control, 512-token length-matched check) | datasets, fold indices | one JSON per dataset and fold; cached in-sample and cross-fitted teacher logits |
| `exp15_inverse_pfd.py` | Supplementary Table S8 and the inverse-route control (abstract-model teacher, full-text student: cross-fitted and in-sample abstract soft labels, hard-label control; baseline and full-text model retrained) | datasets, fold indices | one JSON per dataset and fold; cached abstract-teacher logits and full-text encoder checkpoints |
| `stats_exp15.py` | Statistics for the inverse route (cluster bootstrap, Wilcoxon, per-dataset means) | the exp15 JSON files | `exp15_stats.json` |
| `stats_exp14.py` | Statistics for the second family (cluster bootstrap, Wilcoxon, per-dataset means, length check, margin predictors) | the exp14 JSON files | `exp14_stats.json` |
| `gen_exp14_cf_blocks.py` | LaTeX for Table 3, Supplementary Tables S8 and S11 | `exp14_stats.json` | `tables/exp14_cf_blocks.tex` |
| `stats_reanalysis.py` | Supplementary Table S9 (five inference schemes for every effect on both metrics), Appendix J | fold-level results of the primary pipeline | the table and its JSON |
| `stats_exp11_wss95.py`, `stats_wss_block.py` | WSS@95 blocks of Tables 2 and 3 | exp8 and exp11 results | JSON |
| `stats_pfd_vs_ft.py` | Parity contrast (PFD student vs full-text model), Table 4 note | exp8 results | JSON |
| `stats_recall_workload.py` | Recall at 20 percent workload (Table 4), Appendix M | recall-curve results | JSON |
| `stats_table1.py` | Table 1 and Supplementary Table S4 (paired contrasts vs BioMedBERT) | encoder-comparison results | JSON |
| `stats_ie_margin.py` | Step 2 heuristic (Spearman of the PFD gain with the teacher margin and with IE), Appendix N | exp8 results | JSON |
| `svm_decomposition_wss.py` | SVM rows of Supplementary Table S10 | datasets, fold indices | JSON |
| `stats_fill_gaps.py`, `stats_v26.py` | SDs and CIs filled into tables; v2.6 integration statistics | result JSON files | JSON |
| `softlabel_memorisation_check.py` | Appendix L, "Why cross-fit, measured" (in-sample vs cross-fitted teacher soft labels against the ground truth, both families, 30 folds each) | cached teacher logits of both pipelines | `softlabel_memorisation.json` |

Statistical conventions used throughout: paired fold-level differences; two-stage cluster bootstrap (datasets, then folds; 20,000 replicates; seed 42) for 95 percent confidence intervals and bootstrap P values; pooled two-sided Wilcoxon signed-rank test over the 30 dataset-folds; SD with ddof = 1.
