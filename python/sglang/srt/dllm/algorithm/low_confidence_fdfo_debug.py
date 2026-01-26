"""
Debug version of LowConfidenceFDFO algorithm.

This file is a drop-in replacement for low_confidence_fdfo.py with added debug logging.

To use:
1. Backup the original file:
   cp python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py.bak

2. Replace with this debug version:
   cp python/sglang/srt/dllm/algorithm/low_confidence_fdfo_debug.py python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py

3. Restart the server and run your test

4. Restore the original:
   cp python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py.bak python/sglang/srt/dllm/algorithm/low_confidence_fdfo.py
"""

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

# Setup debug logger
logger = logging.getLogger("sglang.dllm.debug")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[DLLM_DEBUG %(asctime)s] %(message)s', 
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)


class LowConfidenceFDFO(DllmAlgorithm):
    """Low Confidence FDFO algorithm let done unmasking reqs go first - DEBUG VERSION"""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.first_done_first_out_mode = True
        self._call_count = 0
        logger.info(f"[INIT] LowConfidenceFDFO initialized with threshold={self.threshold}, "
                   f"block_size={self.block_size}, mask_id={self.mask_id}")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], bool]:
        self._call_count += 1
        call_id = self._call_count
        
        # Forward pass through the model
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_size = forward_batch.batch_size

        # Compute mask counts per block BEFORE modifying input_ids
        mask_counts_cpu = (forward_batch.input_ids == self.mask_id).view(batch_size, self.block_size).sum(dim=1).tolist()
        
        logger.debug(f"[RUN #{call_id}] batch_size={batch_size}, mask_counts={mask_counts_cpu}")
        
        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
        
        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
            block_mask_index = block_input_ids == self.mask_id
            mask_count = torch.sum(block_mask_index).item()
            
            if mask_count == 0:
                # No masks in this block - will be accepted
                logger.debug(f"[RUN #{call_id}] batch={batch_id}: NO MASKS, block will be accepted. "
                           f"First 5 tokens: {block_input_ids[:5].tolist()}")
                continue
            
            curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]

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
            num_above_threshold = transfer_index.sum().item()

            # Get mask positions and their confidences
            mask_positions = block_mask_index.nonzero(as_tuple=True)[0].tolist()
            mask_confidences = confidence[block_mask_index].tolist()
            
            logger.debug(f"[RUN #{call_id}] batch={batch_id}: {mask_count} masks at positions {mask_positions[:5]}..., "
                        f"confidences={[f'{c:.3f}' for c in mask_confidences[:5]]}..., "
                        f"{num_above_threshold} above threshold {self.threshold}")

            if num_above_threshold == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True
                selected_pos = select_index.item()
                selected_conf = confidence[selected_pos].item()
                selected_token = x[selected_pos].item()
                logger.debug(f"[RUN #{call_id}] batch={batch_id}: FALLBACK - selecting pos {selected_pos} "
                           f"with conf {selected_conf:.3f}, token={selected_token}")

            block_input_ids[transfer_index] = x[transfer_index]
            
            # Count remaining masks after transfer
            remaining_masks = (block_input_ids == self.mask_id).sum().item()
            logger.debug(f"[RUN #{call_id}] batch={batch_id}: After transfer, {remaining_masks} masks remaining")
        
        # Build output
        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size).tolist()
        next_token_ids_list = []
        accept_length_per_req_cpu = []
        
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            # Key logic: accept only if NO masks at the START of this iteration
            accept_len = self.block_size if mask_counts_cpu[i] == 0 else 0
            accept_length_per_req_cpu.append(accept_len)
            
            if accept_len > 0:
                logger.info(f"[RUN #{call_id}] batch={i}: ACCEPT block (no masks at start), "
                          f"tokens={next_token_ids[i][:5]}...")
            else:
                logger.debug(f"[RUN #{call_id}] batch={i}: INCOMPLETE (had {mask_counts_cpu[i]} masks at start)")

        return logits_output, next_token_ids_list, accept_length_per_req_cpu, can_run_cuda_graph


Algorithm = LowConfidenceFDFO
