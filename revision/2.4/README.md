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

## Files to share for result interpretation

To inspect results or prepare the final table/figure, the following small files
are sufficient:

- `flow_results.csv`;
- `gate_layers.csv`;
- `calibration.json`;
- optionally `utility_summary.csv` and `trend_statistics.csv`.

Checkpoints and PCAPs are not needed for downstream statistical analysis once
these files have been produced.

`compression_pick_flows.py` and `compression_weighted_sample.py` are optional
case-selection/visualization utilities from the earlier workflow.  Their
outcome-selected summaries are not used by the formal three-row analysis above.
