"""
Debug script to investigate why DLLM produces empty answers.

This script adds detailed logging to trace the DLLM generation process
and identify why some requests end up with empty output_ids.

Usage:
1. First, apply the debug patches by running:
   python -m sglang.test.debug_dllm_empty_answer --patch

2. Then run your evaluation:
   python -m sglang.test.few_shot_gsm8k --num-questions 10

3. Check the debug output in the server logs

4. To remove patches:
   python -m sglang.test.debug_dllm_empty_answer --unpatch
"""

import argparse
import os

# Debug patch for low_confidence_fdfo.py
PATCH_LOW_CONFIDENCE_FDFO = """
# === DEBUG PATCH START ===
import logging
_dllm_debug_logger = logging.getLogger("dllm_debug")
_dllm_debug_logger.setLevel(logging.DEBUG)
if not _dllm_debug_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[DLLM_DEBUG] %(message)s'))
    _dllm_debug_logger.addHandler(_handler)

def _debug_log(msg):
    _dllm_debug_logger.debug(msg)
# === DEBUG PATCH END ===
"""

PATCH_RUN_METHOD = """
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], bool]:
        # Forward pass through the model (no preprocessing needed)
        # each block may contain mask or finished tokens
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_size = forward_batch.batch_size

        # Compute mask counts per block BEFORE _pick_tokens modifies input_ids
        # Convert mask counts to CPU once to avoid multiple D2H transfers
        mask_counts_cpu = (forward_batch.input_ids == self.mask_id).view(batch_size, self.block_size).sum(dim=1).tolist()

        # === DEBUG: Log mask counts ===
        _debug_log(f"[run] batch_size={batch_size}, mask_counts_cpu={mask_counts_cpu}")

        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[
                curr_block_start:curr_block_end,
            ]
            block_mask_index = block_input_ids == self.mask_id
            mask_count = torch.sum(block_mask_index).item()

            # === DEBUG: Log per-batch info ===
            _debug_log(f"[run] batch_id={batch_id}, mask_count_in_block={mask_count}")

            if mask_count == 0:
                _debug_log(f"[run] batch_id={batch_id} - No masks, skipping unmask logic")
                continue

            curr_logits = logits_output.full_logits[
                curr_block_start:curr_block_end,
            ]

            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(block_mask_index, x, block_input_ids)
            confidence = torch.where(block_mask_index, p, -np.inf)

            transfer_index = confidence > self.threshold
            num_transferred = transfer_index.sum().item()

            # === DEBUG: Log confidence and transfer info ===
            mask_confidences = confidence[block_mask_index].tolist()
            _debug_log(f"[run] batch_id={batch_id}, threshold={self.threshold}, "
                      f"mask_confidences={mask_confidences[:5]}..., "  # Show first 5
                      f"num_above_threshold={num_transferred}")

            if num_transferred == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True
                _debug_log(f"[run] batch_id={batch_id} - No tokens above threshold, "
                          f"forcing top-1 at index {select_index.item()}")

            block_input_ids[transfer_index] = x[transfer_index]

            # === DEBUG: Log remaining masks after transfer ===
            remaining_masks = (block_input_ids == self.mask_id).sum().item()
            _debug_log(f"[run] batch_id={batch_id}, remaining_masks_after_transfer={remaining_masks}")

        # Build output token IDs list with decode markers
        # Reshape and convert to CPU list in one operation
        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size).tolist()
        next_token_ids_list = []
        accept_length_per_req_cpu = []
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            accept_len = self.block_size if mask_counts_cpu[i] == 0 else 0
            accept_length_per_req_cpu.append(accept_len)

            # === DEBUG: Log final accept decision ===
            _debug_log(f"[run] batch_id={i}, accept_length={accept_len}, "
                      f"reason={'no_masks_at_start' if accept_len > 0 else 'had_masks_at_start'}")

        return logits_output, next_token_ids_list, accept_length_per_req_cpu, can_run_cuda_graph
"""


# Debug patch for scheduler_output_processor_mixin.py
PATCH_PROCESS_BATCH = """
    def process_batch_result_dllm_fdfo(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        # === DEBUG IMPORT ===
        import logging
        _dllm_debug_logger = logging.getLogger("dllm_debug")
        def _debug_log(msg):
            _dllm_debug_logger.debug(msg)
        # === END DEBUG IMPORT ===

        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()
        block_size = batch.dllm_config.block_size

        for idx in range(batch.batch_size()):
            req = batch.reqs[idx]
            next_token_ids:list = result.next_token_ids[idx]
            len_cur_tokens = len(next_token_ids)
            accept_len = result.accept_length_per_req_cpu[idx]

            # === DEBUG: Log processing info ===
            _debug_log(f"[process] req_id={req.rid[:8]}..., idx={idx}, "
                      f"accept_len={accept_len}, len_cur_tokens={len_cur_tokens}, "
                      f"current_output_ids_len={len(req.output_ids)}")

            assert len_cur_tokens == block_size
            if accept_len == 0:
                req.dllm_incomplete_ids = next_token_ids
                # === DEBUG: Log incomplete ===
                _debug_log(f"[process] req_id={req.rid[:8]}... - INCOMPLETE, "
                          f"setting dllm_incomplete_ids, will retry")
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

            # === DEBUG: Log fill info ===
            _debug_log(f"[process] req_id={req.rid[:8]}..., len_input={len_input}, "
                      f"len_fill={len_fill}")

            if (len_fill < len_input):
                _debug_log(f"[process] req_id={req.rid[:8]}... - PREFILL stage, skipping output")
                continue # prefill

            if len_fill - len_cur_tokens < len_input:
                old_len = len(next_token_ids)
                next_token_ids = next_token_ids[len_input-len_fill:] # avoid mix input and output
                _debug_log(f"[process] req_id={req.rid[:8]}... - Trimmed tokens from {old_len} to {len(next_token_ids)}")

            self.num_generated_tokens += len_cur_tokens

            finished=False
            tokens_added = 0
            for next_token_id in next_token_ids:
                req.output_ids.append(next_token_id)
                tokens_added += 1
                req.check_finished()
                if req.finished():
                    _debug_log(f"[process] req_id={req.rid[:8]}... - FINISHED after adding {tokens_added} tokens, "
                              f"reason={req.finished_reason}, total_output_ids={len(req.output_ids)}")
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                    finished = True
                    break

            if not finished:
                _debug_log(f"[process] req_id={req.rid[:8]}... - Added {tokens_added} tokens, "
                          f"total_output_ids={len(req.output_ids)}, continuing...")
                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()
"""


def get_file_paths():
    """Get the paths to the files we need to patch."""
    import sglang

    sglang_path = os.path.dirname(sglang.__file__)

    return {
        "low_confidence_fdfo": os.path.join(
            sglang_path, "srt", "dllm", "algorithm", "low_confidence_fdfo.py"
        ),
        "scheduler_output_processor": os.path.join(
            sglang_path, "srt", "managers", "scheduler_output_processor_mixin.py"
        ),
    }


def create_backup(filepath):
    """Create a backup of the original file."""
    backup_path = filepath + ".backup"
    if not os.path.exists(backup_path):
        with open(filepath, "r") as f:
            content = f.read()
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup: {backup_path}")
    return backup_path


def restore_backup(filepath):
    """Restore the original file from backup."""
    backup_path = filepath + ".backup"
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            content = f.read()
        with open(filepath, "w") as f:
            f.write(content)
        os.remove(backup_path)
        print(f"Restored from backup: {filepath}")
        return True
    else:
        print(f"No backup found for: {filepath}")
        return False


def apply_patches():
    """Apply debug patches to the DLLM files."""
    paths = get_file_paths()

    print("Applying debug patches...")
    print("=" * 60)

    # Patch low_confidence_fdfo.py
    filepath = paths["low_confidence_fdfo"]
    create_backup(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Add debug imports after the existing imports
    import_end = content.find("class LowConfidenceFDFO")
    if import_end == -1:
        print(f"ERROR: Could not find class definition in {filepath}")
        return False

    # Insert debug code before class definition
    new_content = (
        content[:import_end] + PATCH_LOW_CONFIDENCE_FDFO + "\n" + content[import_end:]
    )

    # Replace the run method
    # Find the run method
    run_start = new_content.find("    def run(")
    if run_start == -1:
        print(f"ERROR: Could not find run method in {filepath}")
        return False

    # Find the end of the run method (next method or end of class)
    run_end = new_content.find("\n\nAlgorithm = ", run_start)
    if run_end == -1:
        run_end = len(new_content)

    new_content = (
        new_content[:run_start] + PATCH_RUN_METHOD + "\n" + new_content[run_end:]
    )

    with open(filepath, "w") as f:
        f.write(new_content)
    print(f"Patched: {filepath}")

    print("=" * 60)
    print("Debug patches applied successfully!")
    print("\nNow restart your sglang server and run your evaluation.")
    print("Debug logs will appear with [DLLM_DEBUG] prefix.")
    return True


def remove_patches():
    """Remove debug patches and restore original files."""
    paths = get_file_paths()

    print("Removing debug patches...")
    print("=" * 60)

    for name, filepath in paths.items():
        restore_backup(filepath)

    print("=" * 60)
    print("Debug patches removed successfully!")
    return True


def print_analysis():
    """Print analysis of the potential issue."""
    print(
        """
================================================================================
DLLM Empty Answer Debug Analysis
================================================================================

Based on code analysis, the issue is in the DLLM FDFO algorithm:

1. In `low_confidence_fdfo.py` line 78:
   ```python
   accept_length_per_req_cpu.append(self.block_size if mask_counts_cpu[i] == 0 else 0)
   ```

   This means `accept_length` is only non-zero when there are NO masks in the block
   at the START of the iteration.

2. In `scheduler_output_processor_mixin.py` line 344-354:
   ```python
   if result.accept_length_per_req_cpu[idx] == 0:
       req.dllm_incomplete_ids = next_token_ids
       continue  # Tokens are NOT added to output_ids!
   ```

POTENTIAL ROOT CAUSES:

A) The request finishes (hits EOS or stop token) on the FIRST token of the first
   complete block, resulting in empty output.

B) The unmasking process never completes - masks are never fully removed, so
   accept_length stays 0 forever.

C) There's a race condition or state issue where the request is marked finished
   before any tokens are added.

DEBUGGING STEPS:

1. Apply the debug patches:
   python -m sglang.test.debug_dllm_empty_answer --patch

2. Restart your sglang server

3. Run a small test:
   python -m sglang.test.few_shot_gsm8k --num-questions 5

4. Look for patterns in the [DLLM_DEBUG] logs:
   - Are mask_counts always > 0?
   - Is accept_length always 0?
   - Are requests finishing immediately?

5. Remove patches when done:
   python -m sglang.test.debug_dllm_empty_answer --unpatch

================================================================================
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Debug DLLM empty answer issue")
    parser.add_argument("--patch", action="store_true", help="Apply debug patches")
    parser.add_argument("--unpatch", action="store_true", help="Remove debug patches")
    parser.add_argument("--analyze", action="store_true", help="Print analysis")

    args = parser.parse_args()

    if args.patch:
        apply_patches()
    elif args.unpatch:
        remove_patches()
    elif args.analyze:
        print_analysis()
    else:
        print_analysis()
        print("\nUsage:")
        print("  --patch    Apply debug patches to DLLM files")
        print("  --unpatch  Remove debug patches and restore originals")
        print("  --analyze  Print detailed analysis of the issue")


if __name__ == "__main__":
    main()
