from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceFDFO(DllmAlgorithm):
    """Low Confidence FDFO algorithm let done unmasking reqs go first"""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.first_done_first_out_mode = True

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
        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[
                curr_block_start:curr_block_end,
            ]
            block_mask_index = block_input_ids == self.mask_id
            if torch.sum(block_mask_index).item() == 0:
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

            if transfer_index.sum().item() == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True

            block_input_ids[transfer_index] = x[transfer_index]
        
        # Build output token IDs list with decode markers
        # Reshape and convert to CPU list in one operation
        next_token_ids = forward_batch.input_ids.view(batch_size, self.block_size).tolist()
        next_token_ids_list = []
        accept_length_per_req_cpu = []
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            accept_length_per_req_cpu.append(self.block_size if mask_counts_cpu[i] == 0 else 0)

        return logits_output, next_token_ids_list, accept_length_per_req_cpu, can_run_cuda_graph


Algorithm = LowConfidenceFDFO
