import torch
import numpy as np
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1_cb, conv3x3_hgd_cb, DSE_cb, DTA_cb, DTA_cb1, BasicConv3d
from typing import Callable, Optional
from einops import rearrange


class HGD(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        stage: int = 1,
        enhance: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(HGD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if enhance == True:
            self.pose_guided_offset = DSE_cb()

        self.conv1 = nn.ModuleList([
            nn.Sequential(
                conv3x3_hgd_cb(inplanes, planes, stride, halving=i),
                norm_layer(planes),
                nn.ReLU(inplace=True),
                BasicConv3d(planes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                norm_layer(planes),
                nn.ReLU(inplace=True)
            )
        for i in range(stage)])

        # self.conv2 = nn.Sequential(
        #         conv3x3_hgd_cb(planes, planes, halving=0),
        #         norm_layer(planes),
        #         nn.ReLU(inplace=True)
        #     )

        # self.conv2_t = PackSequenceWrapperConv3d_Temporal(planes, planes)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(conv1x1_cb(inplanes, planes, stride=stride),
                                        nn.BatchNorm3d(planes))
        self.stride = stride

        self.enhance = enhance
        self.stage = stage

    def forward(self, x):
        x, pose = x

        if self.enhance == True:
            x = self.pose_guided_offset(x, pose)

        print(x.size()) # n c t h w
        out_hih = []
        for i in range(self.stage):
            out = self.conv1[i](x)
            out_hih.append(out)

        out = sum(out_hih)

        # out = self.conv2(out)
        # out = self.conv2_t(out, seqL)[0]

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class HiH_CASIAB(BaseModel):

   def build_network(self, model_cfg):
       #B, C = [1, 1, 1, 1], 64
       in_C, B, C, enhance = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C'], model_cfg['Backbone']['enhance']
       self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

       self.inplanes = 1 * C

       self.conv0 = BasicConv3d(in_C, self.inplanes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
       self.bn0 = nn.BatchNorm3d(self.inplanes)
       self.relu = nn.ReLU(inplace=True)


       self.layer1 = nn.ModuleList(
           [_ for _ in self.make_layer(HGD, 1 * C, 1 * C, stride=1, stage=1, enhance=False, blocks_num=B[0])]
       )

       self.layer2 = nn.ModuleList(
           [_ for _ in self.make_layer(HGD, 1 * C, 1 * C, stride=1, stage=2, enhance=False, blocks_num=B[1])]
       )

       if enhance == True:
           self.dta = DTA_cb()
       else:
           self.dta = DTA_cb1()

       self.layer3 = nn.ModuleList(
           [_ for _ in self.make_layer(HGD, 1 * C, 2 * C, stride=1, stage=3, enhance=False, blocks_num=B[2])]
       )

       self.layer4 = nn.ModuleList(
           [_ for _ in self.make_layer(HGD, 2 * C, 4 * C, stride=1, stage=3, enhance=False, blocks_num=B[3])]
       )

       self.blocks_num = B

       self.FCs = SeparateFCs(64, 4*C, 2*C)
       self.BNNecks = SeparateBNNecks(64, 2*C, class_num=model_cfg['SeparateBNNecks']['class_num'])

       self.TP = PackSequenceWrapper(torch.max)
       self.HPP = HorizontalPoolingPyramid(bin_num=[64])


   def make_layer(self, block, inplanes ,planes, stride, stage, enhance,blocks_num):
       layers = [block(inplanes, planes, stride=stride, stage=stage, enhance=enhance)]
       s = [1]
       for i in range(1, blocks_num):
           layers.append(
                   block(planes, planes, stride=s)
           )
       newlayers = nn.ModuleList(nn.Sequential(*([item])) for item in layers)
       return newlayers

   def inputs_pretreament(self, inputs):
       ### Ensure the same data augmentation for heatmap and silhouette
       pose_sils = inputs[0]
       new_data_list = []
       for pose, sil in zip(pose_sils[0], pose_sils[1]):
           sil = sil[:, np.newaxis, ...]  # [T, 1, H, W]
           # print(sil.dtype)
           pose_h, pose_w = pose.shape[-2], pose.shape[-1]
           sil_h, sil_w = sil.shape[-2], sil.shape[-1]
           if sil_h != sil_w and pose_h == pose_w:
               cutting = (sil_h - sil_w) // 2
               pose = pose * 255.0
               pose = pose[..., cutting:-cutting]
               # print(pose.dtype)
           cat_data = np.concatenate([pose, sil], axis=1)  # [T, 3, H, W]
           # print(cat_data.shape)
           new_data_list.append(cat_data)
       new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
       return super().inputs_pretreament(new_inputs)

   def forward(self, inputs):
       ipts, labs, _, _, seqL = inputs  # 0 pose  1 sil

       pose = ipts[0]  # n s c h w
       sils = pose[:, :, -1, ...]  # -1=sil 0=pose
       pose = pose[:, :, 0, ...].unsqueeze(2)
       if len(sils.size()) == 4:
           sils = sils.unsqueeze(1)

       del ipts

       x = self.conv0(sils)
       x = self.bn0(x)
       x = self.relu(x)

       # print(x.size()) #torch.Size([8, 64, 30, 64, 44])

       for i in range(self.blocks_num[0]):
           x_l = []
           x_l.append(x)
           x_l.append(pose)
           x = self.layer1[i](x_l)

       # print(x.size()) #torch.Size([1, 64, 514, 64, 44])

       for i in range(self.blocks_num[1]):
           x_l = []
           x_l.append(x)
           x_l.append(pose)
           x = self.layer2[i](x_l)

       # print(x.size()) #torch.Size([1, 128, 459, 32, 22])
       n, _, s, h, w = x.size()
       if s < 3:
           repeat = 3 if s == 1 else 2
           x = x.repeat(1, 1, repeat, 1, 1)
           pose = pose.repeat(1, repeat, 1, 1, 1)
       x, pose = self.dta(x, pose)

       # print(x.size()) #torch.Size([1, 128, 162, 32, 22])

       for i in range(self.blocks_num[2]):
           x_l = []
           x_l.append(x)
           x_l.append(pose)
           x = self.layer3[i](x_l)

       for i in range(self.blocks_num[3]): # [n, c, s, h, w]
           x_l = []
           x_l.append(x)
           x_l.append(pose)
           x = self.layer4[i](x_l)


       # print(x.size())
       # Temporal Pooling, TP
       seqL = None
       outs = self.TP(x, seqL, options={"dim": 2})[0]  # [n, c, h, w]
       n, c, h, w = outs.size()

       # Horizontal Pooling Matching, HPM
       feat = self.HPP(outs)  # [n, c, p]

       embed_1 = self.FCs(feat)  # [n, c, p]
       embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
       
       if self.inference_use_emb:
            embed = embed_2
       else:
            embed = embed_1

       retval = {
           'training_feat': {
               'triplet': {'embeddings': embed_1, 'labels': labs},
               'softmax': {'logits': logits, 'labels': labs}
           },
           'visual_summary': {
               'image/sils': rearrange(pose, 'n c s h w -> (n s) c h w'),
           },
           'inference_feat': {
               'embeddings': embed
           }
       }
       return retval
