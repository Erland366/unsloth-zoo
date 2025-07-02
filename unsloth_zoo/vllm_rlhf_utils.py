# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
__all__ = [
    "WorkerExtension",
    "ColocateWorkerExtension",
]

def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform
        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def update_weights_from_ipc_handles(self, ipc_handles):
        handles = ipc_handles[self.device_uuid]
        device_id = self.device.index
        weights = []
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # the key is to change device id to the current device id
            # in case two processes have different CUDA_VISIBLE_DEVICES
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

    def get_model_runner(self):
        from vllm.model_executor.models.utils import PPMissingLayer

        vllm_model = self.model_runner.model
        model_loras_A, model_loras_B = [], []
        vllm_loras_A,  vllm_loras_B  = [], []
        parameters = []
        for v_layer in vllm_model.model.layers:
            if isinstance(v_layer, PPMissingLayer):
                continue
            print(v_layer.self_attn.qkv_proj.lora_a_stacked[0])
            vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[0])
            vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[1])
            vllm_loras_A .append(v_layer.self_attn.qkv_proj.lora_a_stacked[2])

            # parameters.append((name, param))
        torch.cuda.synchronize()
        return vllm_loras_A

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated

    def get_weight_ipc_handles(self):
        from torch.multiprocessing.reductions import reduce_tensor
        data = {}
        vllm_model = self.model_runner.model
        for name, p in vllm_model.named_parameters():
            # the training actor might only have a subset of the weights
            # and need to all-gather the weights from all the actors.
            # for demonstration, here we assume all training actors have
            # the full weights.
            data[name] = reduce_tensor(p.detach())
        return {self.device_uuid: data}

    def get_bnb_quant_state_and_offsets(self) -> list:
        results = {}
        quant_state = {}
        output_sizes = {}
        offsets = {}
        vllm_model = self.model_runner.model
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()

        for name, module in vllm_model.named_modules():
            if hasattr(module, "weight") and hasattr(module.weight, "bnb_quant_state"):
                
                # === START NEW DEEP DEBUGGING BLOCK ===
                # Let's check a specific layer, like the first down_proj layer, to reduce log spam
                # We check the .base_layer since that's where the quant_state attribute lives
                if name == "model.layers.0.mlp.down_proj.base_layer":
                    qs_dict = module.weight.bnb_quant_state
                    
                    print(f"\n\n[Unsloth Deep Debug] Rank {rank}, Layer {name}", flush=True)
                    print(f"[Unsloth Deep Debug] Type of quant_state: {type(qs_dict)}", flush=True)
                    
                    if isinstance(qs_dict, dict):
                        print(f"[Unsloth Deep Debug] Dictionary Keys: {qs_dict.keys()}", flush=True)
                        # Also print some important values
                        print(f"[Unsloth Deep Debug] Blocksize: {qs_dict.get('blocksize')}", flush=True)
                        print(f"[Unsloth Deep Debug] Quant Type: {qs_dict.get('quant_type')}", flush=True)
                        print(f"[Unsloth Deep Debug] Dtype: {qs_dict.get('dtype')}", flush=True)
                    print("[Unsloth Deep Debug] End of block\n\n", flush=True)
                # === END NEW DEEP DEBUGGING BLOCK ===

                # Original logic continues here
                quant_state[name] = module.weight.bnb_quant_state
                offsets[name] = module.weight.bnb_shard_offsets

            elif hasattr(module, "weight") and hasattr(module, "output_sizes"):
                output_sizes[name] = module.output_sizes

        results["quant_state"] = quant_state
        results["offsets"] = offsets
        results["output_sizes"] = output_sizes
        return results