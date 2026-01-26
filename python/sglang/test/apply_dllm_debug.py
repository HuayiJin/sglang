#!/usr/bin/env python3
"""
Apply/remove debug patches for DLLM empty answer investigation.

Usage:
    # Apply debug patches
    python -m sglang.test.apply_dllm_debug --apply
    
    # Remove debug patches  
    python -m sglang.test.apply_dllm_debug --remove
    
    # Check status
    python -m sglang.test.apply_dllm_debug --status

After applying patches, restart the sglang server and run your test.
Debug logs will appear with [DLLM_DEBUG] prefix.
"""

import argparse
import os
import shutil
import sys


def get_sglang_path():
    """Get the sglang installation path."""
    import sglang
    return os.path.dirname(sglang.__file__)


def get_file_paths():
    """Get paths to files that need patching."""
    sglang_path = get_sglang_path()
    return {
        'algorithm': os.path.join(sglang_path, 'srt', 'dllm', 'algorithm', 'low_confidence_fdfo.py'),
        'processor': os.path.join(sglang_path, 'srt', 'managers', 'scheduler_output_processor_mixin.py'),
    }


# ============================================================================
# Debug version of low_confidence_fdfo.py
# ============================================================================
ALGORITHM_DEBUG_CODE = '''"""Debug version of LowConfidenceFDFO - AUTO-GENERATED, DO NOT EDIT"""
from typing import List, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

# Debug logger
_logger = logging.getLogger("sglang.dllm.debug")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[DLLM_DEBUG %(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    _logger.addHandler(_h)


class LowConfidenceFDFO(DllmAlgorithm):
    """Low Confidence FDFO algorithm - DEBUG VERSION"""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.first_done_first_out_mode = True
        self._call_count = 0
        _logger.info(f"[INIT] threshold={self.threshold}, block_size={self.block_size}, mask_id={self.mask_id}")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], bool]:
        self._call_count += 1
        cid = self._call_count
        
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_size = forward_batch.batch_size

        mask_counts_cpu = (forward_batch.input_ids == self.mask_id).view(batch_size, self.block_size).sum(dim=1).tolist()
        _logger.debug(f"[RUN#{cid}] batch_size={batch_size}, mask_counts={mask_counts_cpu}")
        
        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
        
        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
            block_mask_index = block_input_ids == self.mask_id
            mask_count = torch.sum(block_mask_index).item()
            
            if mask_count == 0:
                _logger.debug(f"[RUN#{cid}] b{batch_id}: NO MASKS, will accept")
                continue
            
            curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]
            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(torch.gather(F.softmax(curr_logits, dim=-1), dim=-1, index=torch.unsqueeze(x, -1)), -1)
            x = torch.where(block_mask_index, x, block_input_ids)
            confidence = torch.where(block_mask_index, p, -np.inf)
            transfer_index = confidence > self.threshold
            num_above = transfer_index.sum().item()

            mask_confs = confidence[block_mask_index].tolist()[:5]
            _logger.debug(f"[RUN#{cid}] b{batch_id}: {mask_count} masks, confs={[f'{c:.3f}' for c in mask_confs]}, {num_above} above thresh")

            if num_above == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True
                _logger.debug(f"[RUN#{cid}] b{batch_id}: FALLBACK pos={select_index.item()}")

            block_input_ids[transfer_index] = x[transfer_index]
            remaining = (block_input_ids == self.mask_id).sum().item()
            _logger.debug(f"[RUN#{cid}] b{batch_id}: {remaining} masks remaining")
        
        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size).tolist()
        next_token_ids_list = []
        accept_length_per_req_cpu = []
        
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            accept_len = self.block_size if mask_counts_cpu[i] == 0 else 0
            accept_length_per_req_cpu.append(accept_len)
            if accept_len > 0:
                _logger.info(f"[RUN#{cid}] b{i}: ACCEPT (no masks at start)")
            else:
                _logger.debug(f"[RUN#{cid}] b{i}: INCOMPLETE ({mask_counts_cpu[i]} masks)")

        return logits_output, next_token_ids_list, accept_length_per_req_cpu, can_run_cuda_graph


Algorithm = LowConfidenceFDFO
'''


# ============================================================================
# Patch for scheduler_output_processor_mixin.py
# ============================================================================
PROCESSOR_PATCH_MARKER = "# === DLLM_DEBUG_PATCH_START ==="
PROCESSOR_PATCH_END = "# === DLLM_DEBUG_PATCH_END ==="

PROCESSOR_DEBUG_IMPORT = '''
# === DLLM_DEBUG_PATCH_START ===
import logging as _dllm_logging
_dllm_proc_logger = _dllm_logging.getLogger("sglang.dllm.debug.processor")
_dllm_proc_logger.setLevel(_dllm_logging.DEBUG)
if not _dllm_proc_logger.handlers:
    _dllm_h = _dllm_logging.StreamHandler()
    _dllm_h.setFormatter(_dllm_logging.Formatter('[DLLM_PROC %(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    _dllm_proc_logger.addHandler(_dllm_h)
# === DLLM_DEBUG_PATCH_END ===
'''

PROCESSOR_FUNC_ORIGINAL = '''    def process_batch_result_dllm_fdfo(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()
        block_size = batch.dllm_config.block_size

        for idx in range(batch.batch_size()):
            req = batch.reqs[idx]
            next_token_ids:list = result.next_token_ids[idx]
            len_cur_tokens = len(next_token_ids)

            assert len_cur_tokens == block_size
            if result.accept_length_per_req_cpu[idx] == 0:
                req.dllm_incomplete_ids = next_token_ids
                # release current kv cache since they are incomplete
                old_prefix_len = len(req.prefix_indices) if hasattr(req, 'prefix_indices') and req.prefix_indices is not None else 0
                new_fill_len = len(req.fill_ids)
                if new_fill_len > old_prefix_len:
                    kv_indices_to_free = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, old_prefix_len:new_fill_len
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices_to_free)
                continue
            
            req.dllm_incomplete_ids = []
            len_input = len(req.origin_input_ids)
            len_fill = len(req.fill_ids)
            if (len_fill < len_input):
                continue # prefill

            if len_fill - len_cur_tokens < len_input:
                next_token_ids = next_token_ids[len_input-len_fill:] # avoid mix input and output
            self.num_generated_tokens += len_cur_tokens

            finished=False
            for next_token_id in next_token_ids:
                # dllm_debug_print(f"Trying to append output_ids on {idx=} with {next_token_id=}")
                req.output_ids.append(next_token_id)
                req.check_finished()
                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                    finished = True
                    break
            if not finished:
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()'''

PROCESSOR_FUNC_DEBUG = '''    def process_batch_result_dllm_fdfo(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        # === DLLM_DEBUG_PATCH_START ===
        import logging as _log
        _plog = _log.getLogger("sglang.dllm.debug.processor")
        # === DLLM_DEBUG_PATCH_END ===
        
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()
        block_size = batch.dllm_config.block_size
        
        _plog.debug(f"[PROC] Processing batch of {batch.batch_size()} reqs")

        for idx in range(batch.batch_size()):
            req = batch.reqs[idx]
            next_token_ids:list = result.next_token_ids[idx]
            len_cur_tokens = len(next_token_ids)
            accept_len = result.accept_length_per_req_cpu[idx]
            
            _plog.debug(f"[PROC] req={req.rid[:8]}... idx={idx} accept={accept_len} output_ids_len={len(req.output_ids)}")

            assert len_cur_tokens == block_size
            if accept_len == 0:
                req.dllm_incomplete_ids = next_token_ids
                _plog.debug(f"[PROC] req={req.rid[:8]}... INCOMPLETE, will retry")
                old_prefix_len = len(req.prefix_indices) if hasattr(req, 'prefix_indices') and req.prefix_indices is not None else 0
                new_fill_len = len(req.fill_ids)
                if new_fill_len > old_prefix_len:
                    kv_indices_to_free = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, old_prefix_len:new_fill_len
                    ]
                    self.token_to_kv_pool_allocator.free(kv_indices_to_free)
                continue
            
            req.dllm_incomplete_ids = []
            len_input = len(req.origin_input_ids)
            len_fill = len(req.fill_ids)
            
            _plog.debug(f"[PROC] req={req.rid[:8]}... len_input={len_input} len_fill={len_fill}")
            
            if (len_fill < len_input):
                _plog.debug(f"[PROC] req={req.rid[:8]}... PREFILL stage, skip output")
                continue

            if len_fill - len_cur_tokens < len_input:
                old_len = len(next_token_ids)
                next_token_ids = next_token_ids[len_input-len_fill:]
                _plog.debug(f"[PROC] req={req.rid[:8]}... Trimmed {old_len} -> {len(next_token_ids)} tokens")
                
            self.num_generated_tokens += len_cur_tokens

            finished=False
            tokens_added = 0
            for next_token_id in next_token_ids:
                req.output_ids.append(next_token_id)
                tokens_added += 1
                req.check_finished()
                if req.finished():
                    _plog.info(f"[PROC] req={req.rid[:8]}... FINISHED after {tokens_added} tokens, "
                              f"reason={req.finished_reason}, total={len(req.output_ids)}")
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                    finished = True
                    break
                    
            if not finished:
                _plog.debug(f"[PROC] req={req.rid[:8]}... Added {tokens_added} tokens, total={len(req.output_ids)}")
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()'''


def apply_patches():
    """Apply debug patches."""
    paths = get_file_paths()
    
    print("=" * 60)
    print("Applying DLLM debug patches...")
    print("=" * 60)
    
    # Patch algorithm file
    algo_path = paths['algorithm']
    backup_path = algo_path + '.orig'
    
    if os.path.exists(backup_path):
        print(f"[WARN] Backup already exists: {backup_path}")
        print("       Patches may already be applied. Use --status to check.")
    else:
        shutil.copy(algo_path, backup_path)
        print(f"[OK] Backed up: {algo_path}")
    
    with open(algo_path, 'w') as f:
        f.write(ALGORITHM_DEBUG_CODE)
    print(f"[OK] Patched: {algo_path}")
    
    # Patch processor file
    proc_path = paths['processor']
    proc_backup = proc_path + '.orig'
    
    with open(proc_path, 'r') as f:
        content = f.read()
    
    if PROCESSOR_PATCH_MARKER in content:
        print(f"[SKIP] Processor already patched: {proc_path}")
    else:
        if not os.path.exists(proc_backup):
            shutil.copy(proc_path, proc_backup)
            print(f"[OK] Backed up: {proc_path}")
        
        # Add import at the top (after existing imports)
        import_pos = content.find('\nfrom sglang.srt')
        if import_pos == -1:
            import_pos = 0
        content = content[:import_pos] + PROCESSOR_DEBUG_IMPORT + content[import_pos:]
        
        # Replace the function
        content = content.replace(PROCESSOR_FUNC_ORIGINAL, PROCESSOR_FUNC_DEBUG)
        
        with open(proc_path, 'w') as f:
            f.write(content)
        print(f"[OK] Patched: {proc_path}")
    
    print("=" * 60)
    print("Debug patches applied!")
    print("")
    print("Next steps:")
    print("1. Restart your sglang server")
    print("2. Run your test: python -m sglang.test.few_shot_gsm8k --num-questions 5")
    print("3. Look for [DLLM_DEBUG] and [DLLM_PROC] logs")
    print("4. When done: python -m sglang.test.apply_dllm_debug --remove")
    print("=" * 60)


def remove_patches():
    """Remove debug patches and restore originals."""
    paths = get_file_paths()
    
    print("=" * 60)
    print("Removing DLLM debug patches...")
    print("=" * 60)
    
    for name, path in paths.items():
        backup = path + '.orig'
        if os.path.exists(backup):
            shutil.copy(backup, path)
            os.remove(backup)
            print(f"[OK] Restored: {path}")
        else:
            print(f"[SKIP] No backup found for: {path}")
    
    print("=" * 60)
    print("Debug patches removed!")
    print("=" * 60)


def check_status():
    """Check if patches are applied."""
    paths = get_file_paths()
    
    print("=" * 60)
    print("DLLM Debug Patch Status")
    print("=" * 60)
    
    for name, path in paths.items():
        backup = path + '.orig'
        
        with open(path, 'r') as f:
            content = f.read()
        
        is_patched = PROCESSOR_PATCH_MARKER in content or 'DEBUG VERSION' in content
        has_backup = os.path.exists(backup)
        
        status = "PATCHED" if is_patched else "ORIGINAL"
        backup_status = "(backup exists)" if has_backup else "(no backup)"
        
        print(f"  {name}: {status} {backup_status}")
        print(f"    Path: {path}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Apply/remove DLLM debug patches')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--apply', action='store_true', help='Apply debug patches')
    group.add_argument('--remove', action='store_true', help='Remove debug patches')
    group.add_argument('--status', action='store_true', help='Check patch status')
    
    args = parser.parse_args()
    
    if args.apply:
        apply_patches()
    elif args.remove:
        remove_patches()
    elif args.status:
        check_status()


if __name__ == '__main__':
    main()
