# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for the CSA2 packed-sequence (THD) helpers."""

import torch

from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    compressed_visible_counts,
    sliding_window_indices,
)
from megatron.core.transformer.experimental_attention_variant.csa2.thd import (
    clear_layout_cache,
    compressed_cu_seqlens,
    packed_layout_cached,
    compressed_entry_metadata,
    compressor_group_rows,
    owned_compressed_entries,
    row_metadata,
    segment_rows,
    shift_compressed_indices,
    visible_compressed_counts,
    window_indices_thd,
)

CU = torch.tensor([0, 5, 12, 14])  # three segments: 5, 7, 2 tokens


class TestRowMetadata:
    def test_segments_positions_validity(self):
        meta = row_metadata(CU, local_rows=16)  # two padding rows past the pack
        assert meta.segment_ids.tolist() == [0] * 5 + [1] * 7 + [2] * 2 + [2, 2]
        assert meta.positions.tolist() == list(range(5)) + list(range(7)) + [0, 1, 0, 0]
        assert meta.valid.tolist() == [True] * 14 + [False, False]

    def test_global_start(self):
        meta = row_metadata(CU, local_rows=4, global_start=7)
        assert meta.segment_ids.tolist() == [1, 1, 1, 1]
        assert meta.positions.tolist() == [2, 3, 4, 5]
        assert segment_rows(meta, 1).tolist() == [0, 1, 2, 3]

    def test_padded_pack(self):
        # physical starts [0, 8, 16], valid lengths [3, 6]: rows 3..7 and 14..15 are padding
        cu_phys = torch.tensor([0, 8, 16])
        lens = torch.tensor([3, 6])
        meta = row_metadata(cu_phys, local_rows=16, seq_lens=lens)
        assert meta.valid.tolist() == [True] * 3 + [False] * 5 + [True] * 6 + [False] * 2
        assert meta.positions.tolist() == [0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        cu_comp = compressed_cu_seqlens(cu_phys, ratio=2, seq_lens=lens)
        assert cu_comp.tolist() == [0, 1, 4]  # 3//2, 6//2
        _, grp, first = compressed_entry_metadata(cu_phys, cu_comp, ratio=2)
        assert first.tolist() == [0, 8, 10, 12]  # physical rows, never inside the padding
        assert grp.tolist() == [0, 0, 1, 2]
        idx = window_indices_thd(meta, window=3, local_row_base=0, halo=0)
        assert idx[3].tolist() == [-1, -1, -1]  # padding row attends nothing
        assert idx[8].tolist() == [8, -1, -1]  # segment 1 starts at its physical row

    def test_packed_layout_from_params(self):
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.transformer.experimental_attention_variant.csa2.thd import packed_layout

        cu = torch.tensor([0, 3, 9], dtype=torch.int32)
        starts, lens = packed_layout(
            PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_q_padded=None)
        )
        assert starts.tolist() == [0, 3, 9] and lens.tolist() == [3, 6]
        starts, lens = packed_layout(
            PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu,
                cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int32),
            )
        )
        assert starts.tolist() == [0, 8, 16] and lens.tolist() == [3, 6]


class TestCompressedLayout:
    def test_cu_and_entries(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        assert cu_comp.tolist() == [0, 2, 5, 6]  # 5//2, 7//2, 2//2
        seg, grp, first = compressed_entry_metadata(CU, cu_comp, ratio=2)
        assert seg.tolist() == [0, 0, 1, 1, 1, 2]
        assert grp.tolist() == [0, 1, 0, 1, 2, 0]
        assert first.tolist() == [0, 2, 5, 7, 9, 12]
        rows = compressor_group_rows(first, ratio=2)
        assert rows.tolist() == [[0, 1], [2, 3], [5, 6], [7, 8], [9, 10], [12, 13]]
        # row 4 (tail of segment 0) and row 11 (tail of segment 1) are never pooled
        assert 4 not in rows.flatten().tolist() and 11 not in rows.flatten().tolist()

    def test_ratio_one_is_identity(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=1)
        assert cu_comp.tolist() == CU.tolist()
        _, _, first = compressed_entry_metadata(CU, cu_comp, ratio=1)
        assert first.tolist() == list(range(14))

    def test_ownership_across_cp_blocks(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        _, _, first = compressed_entry_metadata(CU, cu_comp, ratio=2)
        owned = [
            owned_compressed_entries(first, 2, global_start=s, local_rows=7).tolist()
            for s in (0, 7)
        ]
        # every entry owned exactly once; group [5,6] belongs to the block holding row 6
        assert [a or b for a, b in zip(*owned)] == [True] * 6
        assert not any(a and b for a, b in zip(*owned))
        assert owned[0] == [True, True, True, False, False, False]


class TestVisibilityAndIndices:
    def test_visible_counts_match_single_sequence(self):
        meta = row_metadata(CU, local_rows=14)
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        counts = visible_compressed_counts(meta, cu_comp, ratio=2)
        expected = torch.cat(
            [
                compressed_visible_counts(5, 2).clamp_max(2),
                compressed_visible_counts(7, 2).clamp_max(3),
                compressed_visible_counts(2, 2).clamp_max(1),
            ]
        )
        assert counts.tolist() == expected.tolist()

    def test_window_indices_respect_segments(self):
        meta = row_metadata(CU, local_rows=14)
        idx = window_indices_thd(meta, window=3, local_row_base=0, halo=0)
        assert idx.shape == (14, 3)
        # row 5 is position 0 of segment 1: only itself
        assert idx[5].tolist() == [5, -1, -1]
        # row 7 (position 2): rows 7, 6, 5
        assert idx[7].tolist() == [7, 6, 5]
        # single-segment pack equals the SBHD helper
        single = row_metadata(torch.tensor([0, 9]), local_rows=9)
        assert torch.equal(window_indices_thd(single, 4, 0, 0), sliding_window_indices(9, 4))

    def test_window_indices_with_halo(self):
        # rank block starting at global row 7 (segment 1 position 2) with a 3-row halo
        meta = row_metadata(CU, local_rows=5, global_start=7)
        idx = window_indices_thd(meta, window=4, local_row_base=0, halo=3)
        # local row 0 = position 2: keys at positions 2,1,0 are buffer rows 3,2,1; k=3 leaves
        # the segment
        assert idx[0].tolist() == [3, 2, 1, -1]
        # local row 4 = global 11 = position 6: buffer rows 7,6,5,4
        assert idx[4].tolist() == [7, 6, 5, 4]

    def test_shift_compressed(self):
        local = torch.tensor([[0, 2, -1]], dtype=torch.int32)
        out = shift_compressed_indices(local, segment_comp_start=5, compressed_base=100)
        assert out.tolist() == [[105, 107, -1]]


class _Params:
    def __init__(self, cu_q, cu_pad=None):
        self.cu_seqlens_q = cu_q
        self.cu_seqlens_q_padded = cu_pad
        self.cp_partition_mode = "contiguous"


class TestPackedLayoutCache:
    """The per-microbatch layout cache must reproduce the direct computations exactly and
    must not serve stale entries."""

    def setup_method(self):
        clear_layout_cache()

    def test_matches_direct_computation_under_cp(self):
        cu = torch.tensor([0, 5, 12, 14], dtype=torch.int32)
        params = _Params(cu)
        total_q, cp_size, ratio = 7, 2, 2
        for cp_rank in range(cp_size):
            layout = packed_layout_cached(params, total_q, cp_rank * total_q, cp_rank, cp_size)
            assert layout.max_position == 7
            meta = row_metadata(cu, total_q, cp_rank * total_q)
            assert torch.equal(layout.row_metadata().positions, meta.positions)
            comp = layout.comp(ratio)
            cu_comp = compressed_cu_seqlens(cu, ratio)
            assert torch.equal(comp.cu_comp, cu_comp)
            assert comp.cu_comp_list == cu_comp.tolist()
            assert comp.n_comp == int(cu_comp[-1])
            assert comp.segments == torch.unique(meta.segment_ids[meta.valid]).tolist()
            for segment in comp.segments:
                rows = segment_rows(meta, segment)
                assert torch.equal(comp.seg_rows(segment), rows)
                assert comp.seg_first_position(segment) == int(meta.positions[rows[0]])
            _, _, first_rows = compressed_entry_metadata(cu, cu_comp, ratio)
            masks = [
                owned_compressed_entries(first_rows, ratio, r * total_q, total_q)
                for r in range(cp_size)
            ]
            assert comp.owned_counts() == [int(m.sum()) for m in masks]
            capacity = max(comp.owned_counts())
            dest, src = comp.gather_maps()
            assert dest.tolist() == torch.cat([torch.nonzero(m).squeeze(1) for m in masks]).tolist()
            assert src.tolist() == torch.cat(
                [torch.arange(n) + r * capacity for r, n in enumerate(comp.owned_counts())]
            ).tolist()
            owned_first = first_rows[masks[cp_rank]]
            expected = (int(owned_first.min()), int(owned_first.max())) if owned_first.numel() else (0, -1)
            assert comp.owned_first_rows_range() == expected

    def test_hit_same_params_miss_other_layout_and_version(self):
        cu = torch.tensor([0, 4, 9])
        params = _Params(cu)
        a = packed_layout_cached(params, 9, 0, 0, 1)
        assert packed_layout_cached(params, 9, 0, 0, 1) is a
        # a different rank view is a different entry
        assert packed_layout_cached(params, 9, 9, 1, 2) is not a
        # an in-place edit of cu_seqlens bumps the version counter and misses
        cu.add_(0)
        assert packed_layout_cached(params, 9, 0, 0, 1) is not a
        # a fresh tensor with the same values is another entry (never a stale hit)
        b = packed_layout_cached(_Params(torch.tensor([0, 4, 9])), 9, 0, 0, 1)
        assert b is not a

    def test_padded_layout_uses_physical_starts(self):
        cu_valid = torch.tensor([0, 3, 9])
        cu_phys = torch.tensor([0, 8, 16])
        layout = packed_layout_cached(_Params(cu_valid, cu_phys), 16, 0, 0, 1)
        assert torch.equal(layout.cu, cu_phys)
        assert layout.seq_lens.tolist() == [3, 6]
        assert layout.max_position == 6
        comp = layout.comp(3)
        assert comp.cu_comp_list == [0, 1, 3]
        assert comp.segments == [0, 1]
