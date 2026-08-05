# Revision 2.4: compression, utility, and learned correction

This directory implements the Section 2.4 evidence chain:

1. identify wire-visible compressed regions and their exposure \(e_i\) in the
   model's raw-content window;
2. estimate conditional content utility with concat and behavior-only
   classifiers that never use \(r_{\mathrm{stat}}\);
3. measure whether learned compatibility corrects the calibrated Shannon prior;
4. test whether the full model preserves content utility and improves over a
   same-checkpoint stat-only intervention.

The three reported positive exposure bins remain:

\[
(0,0.25],\qquad(0.25,0.50],\qquad(0.50,1.00].
\]

## 1. Compression audit

If `flow_details.csv` from the previous audit is still available, reuse it.
There is no need to rerun this step merely because \(u_i\) was added.

For a new dataset:

```bash
python revision/2.4/compression_audit.py \
  /path/to/DATASET \
  --compression-level 3 \
  --output-dir /path/to/DATASET_compression_audit
```

The input directory must have the form `DATASET/label/*.pcap`.

## 2. Aligned checkpoint inference

Required model/data inputs:

- the Stage-1 concat classifier checkpoint (`modality=both`);
- the Stage-1 behavior-only checkpoint (`modality=size`);
- the full fine-tuned ITGCA checkpoint;
- the processed `val.pkl` used only for temperature scaling;
- `label2id.pkl`;
- the audited PCAP directory, unless paths stored in `flow_details.csv` are
  still valid.

The concat checkpoint must use the same behavior architecture as the
behavior-only checkpoint.  The full ITGCA model is built from these same two
configuration files; there is no separate ITGCA config.  Raw and behavior
encoder depths come from the two JSON files, while the number of fusion layers
is inferred from `fusion.fusion_layers.*` in the ITGCA checkpoint.  All depths,
shared dimensions, and checkpoint keys are validated strictly before inference.

A typical run is:

```bash
python revision/2.4/compression_checkpoint_inference.py \
  /path/to/DATASET_compression_audit/flow_details.csv \
  --dataset-dir /path/to/DATASET \
  --concat-checkpoint /path/to/concat.bin \
  --behavior-checkpoint /path/to/behavior.bin \
  --itgca-checkpoint /path/to/itgca.bin \
  --validation-path /path/to/val.pkl \
  --label2id-path /path/to/label2id.pkl \
  --raw-config models/bert/base_config.json \
  --behavior-config models/bert/behavior_6_config.json \
  --utility-threshold 0.01 \
  --evaluation-scope audit_directory_diagnostic \
  --device cuda:0 \
  --output-dir /path/to/DATASET_revision_2_4
```

Use `--evaluation-scope heldout_test` only when the audited PCAP directory is
genuinely the held-out test set.  The default
`audit_directory_diagnostic` may include training flows, so its accuracy must
not be described as held-out performance.

Temperature scaling is recommended.  `--skip-temperature-scaling` permits a
diagnostic run without `val.pkl`, but log-score comparisons from that run are
weaker.

The command writes:

- `flow_results.csv`: one row per audited compressed flow;
- `summary.csv`: preliminary aggregate groups;
- `gate_layers.csv`: one row per flow and fusion layer;
- `calibration.json`: fitted temperatures and validation NLL.

The existing `data/summary_*.csv` files were produced by the earlier raw-vs-
behavior definition.  Keep them for provenance, but do not mix their
`mean_content_utility` or rescue columns with v2 results.

Key definitions in `flow_results.csv` are:

\[
u_i
=
\log p_{\mathrm{concat}}(y_i)
-
\log p_{\mathrm{behavior}}(y_i),
\]

\[
v_i
=
\log p_{\mathrm{ITGCA}}(y_i)
-
\log p_{\mathrm{behavior}}(y_i),
\]

\[
c_i
=
\log p_{\mathrm{ITGCA}}(y_i)
-
\log p_{\mathrm{stat-only}}(y_i).
\]

`delta_learned` is
\(r_{\mathrm{learned}}-r_{\mathrm{calibrated}}\), while `delta_mod` is
\(r_{\mathrm{mod}}-r_{\mathrm{calibrated}}\).

## 3. Three-row formal summary

Analyze Browser and ITC together by supplying both `flow_results.csv` files:

```bash
python revision/2.4/compression_utility_analysis.py \
  /path/to/Browser_revision_2_4/flow_results.csv \
  /path/to/ITC_revision_2_4/flow_results.csv \
  --utility-threshold 0.01 \
  --bootstrap-repetitions 2000 \
  --seed 42 \
  --output-dir /path/to/revision_2_4_analysis
```

It writes:

- `utility_summary.csv`: exactly three positive \(e_i\) rows per dataset, with
  label-stratified bootstrap 95% confidence intervals;
- `trend_statistics.csv`: Spearman/Kendall trends, high-minus-low exposure
  contrasts, and the helpful-vs-harmful learned-correction contrast.

For threshold sensitivity, repeat the analysis with:

```bash
--utility-threshold 0
```

and:

```bash
--utility-threshold 0.05
```

Do not choose a threshold after inspecting which one produces the strongest
result; use `0.01` as the prespecified main analysis and the other two only as
sensitivity checks.

## 4. Incremental no-r_learned ablation

When the preceding results already exist, do not rerun Sections 1-3.  Use
`compression_r_learned_ablation.py` to run only the independently trained and
fine-tuned `--ablate_r_learned` classifier on the same positive-exposure flows.
Rows with \(e_i=0\) are skipped.

The checkpoint passed here must be a fine-tuned classifier checkpoint, not a
Stage 2 pre-training checkpoint.  Fine-tune the ablated pre-training checkpoint
with the same split, label mapping, seed, optimizer settings, epochs, and model
selection rule used by the full model, adding:

```bash
--use_itgca --ablate_r_learned
```

The two pre-training checkpoints must also be matched by training progress.  A
120,000-step full checkpoint must not be compared with a 100,000-step ablation;
compare both at 100,000 steps or continue the ablation to 120,000 steps.

Run the incremental inference once for Browser and once for ITC-Net-Blend:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python3 revision/2.4/compression_r_learned_ablation.py \
  /path/to/existing/flow_results.csv \
  --dataset-dir /path/to/DATASET \
  --dataset-name DATASET \
  --no-r-learned-checkpoint /path/to/fine_tuned_no_r_learned.bin \
  --label2id-path /path/to/label2id.pkl \
  --raw-config models/bert/base_config.json \
  --behavior-config models/bert/base_behavior_config.json \
  --batch-size 64 \
  --device cuda:0 \
  --output-dir /path/to/DATASET_r_learned_ablation
```

This command does not regenerate the compression audit, concat/behavior/full
ITGCA predictions, utility statistics, gate statistics, or the previous formal
summary.  It only recreates input tensors for the selected PCAPs because the
existing CSV does not store model tensors.

It writes:

- `r_learned_ablation_flows.csv`: reused full predictions paired with the new
  no-r_learned predictions;
- `r_learned_ablation_summary.csv`: exactly three rows containing
  `delta_macro_f1_pp_full_minus_no_r_learned` and a label-stratified bootstrap
  95% interval;
- `r_learned_ablation_provenance.json`: checkpoint and source provenance.

The delta column is already expressed in percentage points and can fill the
pending final column in `data/main_text_table.csv`.

## Files to share for result interpretation

To inspect the original results or prepare the existing table/figure, the
following small files are sufficient:

- `flow_results.csv`;
- `gate_layers.csv`;
- `calibration.json`;
- optionally `utility_summary.csv` and `trend_statistics.csv`.

For the incremental ablation, only the two
`r_learned_ablation_summary.csv` files and their provenance JSON files are
needed.

Checkpoints and PCAPs are not needed for downstream statistical analysis once
these files have been produced.

`compression_pick_flows.py` and `compression_weighted_sample.py` are optional
case-selection/visualization utilities from the earlier workflow.  Their
outcome-selected summaries are not used by the formal three-row analysis above.
