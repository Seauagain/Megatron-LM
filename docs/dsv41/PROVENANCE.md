# DeepSeek-V4.1 on Megatron-Core: provenance log

This document records where every idea in the `dev-dsv41` branch comes from and what
was borrowed from which public source. The branch is based only on merged upstream code;
draft pull requests were read for design and interface shape but their code was not
merged, copied or adapted. Verbatim similarity checks against those drafts are part of
every milestone review.

## Base

| Item | Value |
| --- | --- |
| Upstream repository | `NVIDIA/Megatron-LM`, branch `dev` |
| Base commit | `0cd11658f44350a141656751259cfe1f72398e9f` ("[Dev] Remove unused legacy hybrid CP implementation (#7203)", 2026-09-11) |
| Fork | `Seauagain/Megatron-LM`, branch `dev-dsv41` |
| Rebase policy | weekly onto `upstream/dev`; this table is updated on every rebase |

## Ground truth for semantics

| Source | Used for |
| --- | --- |
| `deepseek-ai/DeepSeek-V4.1-Flash` (HF snapshot, commit `dba1be0a`), `inference/model.py` | CSA2 layer roles, compressor / indexer / candidate-block / sparse-attention math, single-pass hyper-connection ordering, Engram gating |
| same, `inference/engram.py` | compressed token map, prime bucket layout, hash multipliers, n-gram rolling hash |
| same, `inference/kernel.py` | attention-sink softmax convention, Sinkhorn split of the hyper-connection mixes, quantization block sizes (import work) |
| same, `config.json`, `model.safetensors.index.json` | hyper-parameters, parameter families and their layer placement |
| same, `DeepSeek_V41_Tech_Report.pdf` | background |

Numerical alignment tests compare against this reference implementation, never against
another port.

## Public drafts read for design only

| Draft | What was looked at | What was *not* taken |
| --- | --- | --- |
| Megatron-LM PR #7224 (DeepSeek-V4.1-Flash without vision and Engram) | field names `dsv4_version`, `csa2_kv_source_layers`, `csa2_index_source_layers`, `csa2_candidate_*`; the four-mode split; the shape of a shared-state object passed between layers; coverage of its configuration rejection tests | any code; its `CSA2State`, `csa2.py`, tests or pipeline changes |
| Megatron-LM PR #7231 (Engram support) | expert-parallel row sharding as the sharding axis; existence of a tokenizer-map generation tool | any code |
| Megatron-LM PR #7224, design review 2026-09-12 (local ref `reference-prs/7224`, read by a review agent; findings recorded in `DESIGN.md` §8) | which computations it keeps in fp32 (head contraction, indexer head-weight scale, fp32 gather before the sparse-attention backward); the guards it enforces (`attention_dropout`, clamp value in its tests); its indexer KL loss recipe (teacher from the detached same-layer query and compressed KV with window + sink mass in the denominator, head-sum, L1 renormalisation, candidate / top-k restriction, one loss per Full and Reindex layer, compressor latent detached before `wk`) as the design to follow when the loss is implemented | any code; its loss, state, kernel or test implementations |
| NeMo AutoModel PR #3855 (DeepSeek-V4.1-Flash training) | evaluation protocol for logit alignment (4,096 positions, full-vocabulary KL and top-1) | any code; parameter-name mapping is derived from the official index file, the draft is used only to cross-check for omissions during the import milestone |

## Merged upstream components reused as is

| Component | Location | Role in V4.1 |
| --- | --- | --- |
| HybridModel / HybridStack / hybrid layer patterns | `megatron/core/models/hybrid/` | model skeleton, pipeline segments (`|`), MoE (`E`) layers |
| DeepSeek-V4 self-attention projections | `experimental_attention_variant/deepseek_v4_hybrid_attention.py` | low-rank query, single-head KV, grouped low-rank output, RoPE application; subclassed by `DSv41SelfAttention` |
| Hyper-connection module | `megatron/core/transformer/hyper_connection.py` | mapping projection, Sinkhorn, aggregation, residual update (parameterisation matches the official `hc_*` tensors) |
| YaRN rotary embedding | `megatron/core/models/common/embeddings/` | compressed-position RoPE |
| CSA v1 context-parallel utilities | `experimental_attention_variant/csa_utils/cp_utils.py` | design basis for the CSA2 CP scheme (milestone M1) |

## Files added by this branch

Every new file carries a header naming the reference it follows. Summary:

| File | Follows |
| --- | --- |
| `csa2/roles.py` | official `ModelArgs` / `Attention.__init__` role rules |
| `csa2/reference.py` | official `Attention.forward`, `Indexer.forward`, `select_candidate_blocks`, `sparse_attn` kernel |
| `csa2/state.py` | official `SharedAttentionRuntime` (re-designed for recompute and pipelining) |
| `csa2/compressor.py` | official `Compressor` |
| `csa2/indexer.py` | official `Indexer` |
| `csa2/attention.py` | official `Attention`; upstream `DSv4HybridSelfAttention` (subclassed) |
| `models/deepseek_v41/engram.py` | official `engram.py`, `Engram`, `ParallelEngramEmbedding` |
| `models/deepseek_v41/hyper_connection.py` | official `Block.forward` ordering; upstream `HyperConnectionHybridLayer` (subclassed) |
| `models/deepseek_v41/hybrid_stack.py` | upstream `HybridStack` (subclassed) |
| `models/deepseek_v41/layer_specs.py` | upstream `hybrid_dsv4_stack_spec` structure |
| `csa2/candidate_kernels.py` | own Triton kernel for the candidate-set indexer scores of the Reindex layers (`relu(q·k)` head-weighted sum over the candidate blocks only, causal and block validity masked in-kernel) plus the top-k over the candidate width; semantics from the official `Indexer.forward` restricted to `select_candidate_blocks` output; no public kernel implements this step |
| `models/deepseek_v41/fused_proj_rms.py` | own Triton kernels for the mHC mapping projection with the V4.1 factor `rsqrt(mean(x^2) + eps)` (forward, `grad_x`, deterministic `grad_W`); the merged `fused_mhc_kernels.fused_proj_rms_compute_h` implements the V4 factor `norm / sqrt(K)` and was used only to confirm that the V4 form is not reusable |

## Similarity check (milestone M0, commit of this branch vs. the drafts)

`tools/dsv41/check_reference_similarity.py` reports runs of at least 10 consecutive
identical code lines (whitespace-normalised, blanks and comments dropped) between the files
added by this branch and the files touched by #7224, #7231 (fetched read-only as
`refs/reference-prs/<n>`) and #3855 (directory checkout). Result for M0: 8 runs, all of
them interface text rather than algorithm:

| Branch file | Matches | Classification |
| --- | --- | --- |
| `models/deepseek_v41/hybrid_stack.py` (constructor and `forward` parameter lists) | upstream `HybridStack` signatures as kept by #7224 | inherited public API |
| `models/deepseek_v41/hyper_connection.py` (`forward` parameter list) | upstream `HyperConnectionHybridLayer.forward` signature | inherited public API |
| `csa2/attention.py` (`CSA2Attention.forward` and `DSv41SelfAttention.__init__` parameter lists) | upstream `CompressedSparseAttention.forward` / `DSv4HybridSelfAttention.__init__` signatures | interface required by the merged `DSv4HybridAttention` caller |
| `csa2/indexer.py` (`build_module(...)` keyword block) | upstream `CSAIndexer` construction boilerplate | merged upstream boilerplate |
| `models/deepseek_v41/layer_specs.py` (`DSv4HybridSelfAttentionSubmodules(...)` wiring) | upstream `get_dsv4_hybrid_module_spec_for_backend` | merged upstream spec wiring |
| `tools/dsv41/build_engram_token_map.py` (normaliser sequence) | #7231 tool, itself mirroring the official `build_compressed_token_map` | must-match constant: the hash is defined by this normaliser |

Re-run at the end of milestone M1 (branch head `c6ad2b0f2`, 2026-09-12, refs #7224 and #7231):
25 branch files against 62 reference files, again 8 runs, the same six classes as above
(`hybrid_stack.py` constructor / `forward` signatures, `hyper_connection.py` `forward`
signature, `layer_specs.py` spec wiring, `csa2/attention.py` `forward` / `__init__` signatures,
`csa2/indexer.py` `build_module` block, the token-map normaliser). None of the M1 code (THD
layout, contiguous CP, shared-state checkpoint transport, fused-kernel adapter, alignment tool)
matches any reference run.

## Changes to shared upstream files (hook points only)

| File | Change |
| --- | --- |
| `transformer_config.py` | V4.1 fields on `MLATransformerConfig`; ratio check relaxed for `dsv4_version='v4.1'`; `_validate_dsv41` |
| `models/hybrid/hybrid_block.py` | four overridable hooks in `HybridStack` (layer wrapping, output contraction parameters, output contraction, extra layer kwargs) |
| `models/hybrid/hybrid_model.py` | pass `input_ids` to the decoder when Engram layers are configured |
| `recompute.py` | `extra_layer_kwargs` forwarded to layers that opt in |
| `pipeline_parallel/schedules.py` | extra handoff channels in the inter-stage tensor for `dsv4_version='v4.1'` |
| `training/argument_utils.py` | V4.1 form of `--csa-compress-ratios` (one entry per model layer) |
| `core/optimizer/distrib_optimizer.py` | precision-aware path of `_copy_model_params_to_main_params`: re-points the optimizer parameter views at the model parameters if their storage changed, refreshes any already materialised master weight and prints the counts (TE FusedAdam creates its master weights lazily in the first `step()`, so after a `--finetune` load nothing is normally materialised). The weight corruption seen at 128K was traced to `--main-grads-dtype bf16` with the precision-aware optimizer, not to this code path; fp32 main gradients are used instead. `load_state_dict`: with the precision-aware TE optimizer and empty state, allocate the state once via `init_state_fn` and copy only the per-group scalars instead of `FusedAdam.load_state_dict` (which held a second full copy of the optimizer state; resume OOM at 128K) |
| `training/training.py` | `_dsv41_self_attention_flops` (own FLOPs coefficients for the CSA2 layers) wired into the hybrid FLOPs path for `dsv4_version='v4.1'` |
| `training/checkpointing.py` | two rank-0 diagnostics on load: why optimizer state is not requested, and the families of optimizer keys requested |
| `core/optimizer/clip_grads.py` | `get_grad_norm_fp32` uses `torch._foreach_norm` when any gradient is not fp32 (bf16 main grads); the TE multi-tensor L2-norm kernel faulted there at 128K |
| `training/datasets/varlen_dataset.py` | `packed-bins` schema: a parquet with `input_ids` / `loss_mask` / `seq_start_id` rows (pre-tokenised packed SFT bins) expands into its sequences; masked-out tokens become `IGNORE_INDEX` targets so the label mask is `loss_mask[1:]` |
| `experimental_attention_variant/deepseek_v4_hybrid_attention.py` | class attribute `query_head_rms_norm` (default `True`, V4 behaviour unchanged) so the V4.1 subclass can skip the per-head query RMS norm |
