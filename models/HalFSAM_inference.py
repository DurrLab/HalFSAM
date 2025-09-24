import sys

import logging
from functools import partial
from typing import List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from models.backbones.hieradet_adapt import Hiera_adapt
from models.backbones.DexiNed import UpConvBlock, SingleConvBlock, weight_init

class HalFSAM_inference(nn.Module):
    """ Definition of the HalFSAM inference network. """
    def __init__(
        self,
        trunk,
        neck,
        memory_encoder,
        memory_attention,
        num_maskmem = 5,
        memory_temporal_stride_for_eval = 1,
        track_in_reverse = False,
        sigmoid_scale_for_mem_enc = 1,
        sigmoid_bias_for_mem_enc = 0,
        offload_mem_to_cpu = False,
    ):
        super(HalFSAM_inference, self).__init__()

        # SAM trunk setup
        self.trunk = trunk
        # SAM neck setup
        self.neck = neck

        # SAM memory module
        self.memory_encoder   = memory_encoder
        self.memory_attention = memory_attention
        # memory setup 
        self.track_in_reverse = track_in_reverse
        self.offload_mem_to_cpu = offload_mem_to_cpu
        if self.offload_mem_to_cpu:
            self.storage_device = torch.device("cpu")
        else: 
            self.storage_device = self.device

        self.mem_dim = self.memory_encoder.out_dim
        self.num_maskmem = num_maskmem
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.mem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.mem_tpos_enc, std=0.02)

        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc

        self.stages_num = len(self.trunk.stage_ends)


        # DexiNed upblocks setup
        self.upblocks = nn.ModuleList()
        down_scale_times = 2
        dim_out = self.neck.d_model
        for stage in range(self.stages_num):
            self.upblocks.append(UpConvBlock(in_features=dim_out, up_scale = down_scale_times))
            down_scale_times += 1
            
        self.block_cat = SingleConvBlock(self.stages_num, 1, stride=1, use_bs=False) 

        self.apply(weight_init)
        self.mems = dict()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def reset_mem(self):
        self.mems.clear()

    def forward(self, fid, frame, track_in_reverse=False):
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        _, C, H, W = frame.size()
        edge_pred, mem = self.run_single_frame_inference(
            fid, 
            frame,
            track_in_reverse,
            memory=self.mems,
        )
        if mem is not None:
            self.mems[fid] = mem
        if len(self.mems.keys()) > self.num_maskmem:
            self.mems.pop(list(self.mems.keys())[0])
        return edge_pred

    def run_single_frame_inference(self, frame_idx, x, track_in_reverse = False, memory = None):
        stage_outs, stage_pos_embs = self.neck(self.trunk(x))
        # memory attention
        if (self.num_maskmem > 0) and (memory is not None and len(memory)>0):
            stage_outs_conditioned = self.memory_condition(
                frame_idx, 
                stage_outs, 
                stage_pos_embs,
                memory, 
                track_in_reverse
            )
        else:
            stage_outs_conditioned = stage_outs
        
        # edge decoder
        results = []
        for i, stage_out in enumerate(stage_outs_conditioned):
            o = self.upblocks[i](stage_out)
            results.append(-o)
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        results = torch.cat(results, dim=0)

        # memory encoding
        memory = None
        if self.num_maskmem > 0:
            memory = self.encode_memory(stage_outs, results[-1])
        return results, memory

    def memory_condition(self, frame_idx, current_feats, current_pos_embeds, memory, track_in_reverse = False):
        device = current_feats[0].device
        r = self.memory_temporal_stride_for_eval
        # choose frames for mems
        t_pos_and_prevs = []
        for t_pos in range(1, self.num_maskmem + 1):
            t_rel = self.num_maskmem + 1 - t_pos  # how many frames before current frame
            if t_rel == 1:
                # for t_rel == 1, we take the last frame (regardless of r)
                if not track_in_reverse:
                    # the frame immediately before this frame (i.e. frame_idx - 1)
                    prev_frame_idx = frame_idx - t_rel
                else:
                    # the frame immediately after this frame (i.e. frame_idx + 1)
                    prev_frame_idx = frame_idx + t_rel
            else:
                # for t_rel >= 2, we take the memory frame from every r-th frames
                if not track_in_reverse:
                    # first find the nearest frame among every r-th frames before this frame
                    # for r=1, this would be (frame_idx - 2)
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    # first find the nearest frame among every r-th frames after this frame
                    # for r=1, this would be (frame_idx + 2)
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
            prev = memory.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, prev))
        
        to_cat_memories, to_cat_memory_pos_embeds = [], []
        for stage_i in range(self.stages_num):
            to_cat_memories.append([])
            to_cat_memory_pos_embeds.append([])

        for t_pos, prev in t_pos_and_prevs:
            if prev is None:
                continue  
            maskmem_feats = prev["maskmem_features"]
            maskmem_pos_embeds = prev["maskmem_pos_enc"]
            maskmem_feat_sizes = prev["maskmem_feat_size"]
            for stage_i, (mem_feat, mem_pos_embed, mem_feat_size) in enumerate(zip(maskmem_feats, maskmem_pos_embeds, maskmem_feat_sizes)):
                # Ensure contiguous memory layout before reshaping
                mem_feat = mem_feat.contiguous()
                mem_pos_embed = mem_pos_embed.contiguous()
                
                # B, C, H, W -> (HW), B, C
                mem_feat_flat = mem_feat.flatten(2).permute(2, 0, 1)
                mem_feat_flat = mem_feat_flat.to(device, non_blocking=True)
                to_cat_memories[stage_i].append(mem_feat_flat)
                
                mem_pos_embed_flat = mem_pos_embed.flatten(2).permute(2, 0, 1)
                mem_pos_embed_flat = mem_pos_embed_flat.to(device, non_blocking=True)
                # Temporal positional encoding
                mem_pos_embed_flat = (
                    mem_pos_embed_flat + self.mem_tpos_enc[self.num_maskmem - t_pos]
                )
                to_cat_memory_pos_embeds[stage_i].append(mem_pos_embed_flat)

        # Ensure contiguous memory layout after concatenation
        memory = [torch.cat(to_cat_memory, dim=0).contiguous() 
                  for to_cat_memory in to_cat_memories]
        memory_pos_embeds = [torch.cat(to_cat_memory_pos_embed, dim=0).contiguous() 
                            for to_cat_memory_pos_embed in to_cat_memory_pos_embeds]

        # Ensure current features are contiguous
        current_feats = [feat.contiguous() for feat in current_feats]
        current_pos_embeds = [pos.contiguous() for pos in current_pos_embeds]

        pix_feat_with_mem = self.memory_attention(
            currs = current_feats,
            memories = memory,
            curr_poss = current_pos_embeds,
            memory_poss = memory_pos_embeds,
            curr_feat_sizes = maskmem_feat_sizes,
            num_obj_ptr_tokens = 0,
        )
        return pix_feat_with_mem

    def encode_memory(self, feats, mask):
        # Ensure input tensors are contiguous
        feats = [f.contiguous() for f in feats]
        mask = mask.contiguous()
        
        # prepare mask for memory encoder
        if not self.training:
            mask_for_mem = (mask > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(mask)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        maskmem_out = self.memory_encoder(
            feats, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )

        if self.offload_mem_to_cpu:
            for key, value in maskmem_out.items():
                if isinstance(value, list):
                    maskmem_out[key] = [v.contiguous().to(self.storage_device, non_blocking=True) 
                                       for v in value]
                elif isinstance(value, tuple):
                    maskmem_out[key] = tuple(v.contiguous().to(self.storage_device, non_blocking=True) 
                                           for v in value)
                elif isinstance(value, torch.Tensor):
                    maskmem_out[key] = value.contiguous().to(self.storage_device, non_blocking=True)

        return maskmem_out

    
from hydra import compose, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

def build_HalFSAM_inference(config_name, ckpt_path=None):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../cfgs"): 
        model_cfg = compose(config_name=config_name)
        OmegaConf.resolve(model_cfg)
    model = instantiate(model_cfg.model, _recursive_=True)

    # load pretrained SAM_adapt weights
    if ckpt_path is not None:
        print('Loading pretrained weigths from: '+ckpt_path)
        state_dict = torch.load(ckpt_path, weights_only=True, map_location='cpu')#['model']
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # trunk weights
        trunk_weights = {k.split('trunk.')[-1]: v for k, v in state_dict.items() if 'trunk.' in k}
        result = model.trunk.load_state_dict(trunk_weights, strict=False)
        # adaptor weigths
        adaptor_weights = {k.split('prompt_generator.')[-1]: v for k, v in state_dict.items() if 'prompt_generator' in k}
        if len(adaptor_weights) > 0:
            model.trunk.prompt_generator.load_state_dict(adaptor_weights, strict=True)
        # neck weights
        neck_weights = {k.split('neck.')[-1]: v for k, v in state_dict.items() if 'neck' in k}
        result = model.neck.load_state_dict(neck_weights, strict=True)
        # upblock weights
        upblock_weights = {k.split('upblocks.')[-1]: v for k, v in state_dict.items() if 'upblocks' in k}
        
        # print('\n'.join(missing_keys))

    # freeze trunk & neck
    model.requires_grad_(True)
    model.trunk.requires_grad_(False)
    # model.trunk.prompt_generator.requires_grad_(True)
    model.neck.requires_grad_(False)
    # count learnable weigths
    num_leanrnables = count_learnable_parameters(model)
    print(f'Learnable weights: {num_leanrnables}')
    return model

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




        