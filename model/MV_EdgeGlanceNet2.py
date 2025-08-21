"""
@Project ：MVANet 
@File    ：MV_EdgeGlanceNet2.py
@IDE     ：PyCharm 
@Author  ：chengxuLiu
@Date    ：2025/5/6 15:25
==============================
SAM Encoder replaced with STT
==============================
"""

import torch
import torch.nn as nn

from einops import rearrange
import torch.nn.functional as F
from typing import Tuple

from .modeling.image_encoder import ImageEncoderViT
from .modeling.mask_decoder_hq import MaskDecoderHQ
from .modeling.prompt_encoder import PromptEncoder
from .modeling import TwoWayTransformer
from functools import partial
from .modeling.stvit import stvit_small
from .MVANet import image2patches, patches2image, MCLM, MCRM, get_activation_fn, PositionEmbeddingSine, make_cbr, \
    resize_as


class Trans_to_loc(nn.Module):
    def __init__(self):
        super(Trans_to_loc, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = image2patches(x)
        x = self.conv1(x)
        return x


class Trans_to_glb(nn.Module):
    def __init__(self, in_channels=512, out_channels=256):
        super(Trans_to_glb, self).__init__()
        # 深度可分离卷积：减少参数量
        self.block = nn.Sequential(
            # 深度卷积（保持通道数）
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            # 逐点卷积（调整通道数）
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MCLM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            # 用于用于全局和局部交互
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            # 下面四个用于局部和全局更新后的交互，与MCLM中的cross attention对应
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])
        # 4 个全连接层 (linear1 到 linear4)，用于对注意力输出后的特征进行进一步处理。
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        # 池化比率
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        # 合并后的局部特征图
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            # tokenize
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            if self.g_pos is None:
                pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
                pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                self.p_poses.append(pos_emb)
        pools = torch.cat(pools, 0)
        if self.g_pos is None:
            self.p_poses = torch.cat(self.p_poses, dim=0)
            pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
            self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](g_hw_b_c + self.g_pos, pools + self.p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)  ## (5,c,h*w)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=256, activation=nn.ReLU(), use_bn=True):
        """
        参数说明:
            in_channels: 输入通道数 (默认 256)
            activation: 激活函数 (默认 ReLU)
            use_bn: 是否使用批量归一化 (默认 True)
        """
        super().__init__()

        # 分支 0: 上采样到 128x128，输出通道 128
        self.branch0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            activation
        )

        # 分支 1: 另一条上采样到 128x128，输出通道 128（独立参数）
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            activation
        )

        # 分支 2: 上采样到 64x64，输出通道 256
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            activation
        )

        # 分支 3: 保持 32x32，输出通道 512
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            activation
        )

        # 分支 4: 下采样到 16x16，输出通道 1024
        self.branch4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.BatchNorm2d(1024)
        )

    def forward(self, x):
        # 输入形状: (4, 256, 32, 32)
        features = [
            self.branch0(x),  # (4, 128, 128, 128)
            self.branch1(x),  # (4, 128, 128, 128)
            self.branch2(x),  # (4, 256, 64, 64)
            self.branch3(x),  # (4, 512, 32, 32)
            self.branch4(x)  # (4, 1024, 16, 16)
        ]
        return features


class MV_EdgeGlanceNet2(nn.Module):
    def __init__(self,
                 image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder,
                 mask_decode: MaskDecoderHQ,
                 ):
        super(MV_EdgeGlanceNet2, self).__init__()
        emb_dim = 128
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decode
        self.Trans2MultiFeature = MultiScaleFeatureExtractor()
        # self.trans2loc1 = Trans_to_loc()
        self.trans2glb = Trans_to_glb(512, 256)
        self.STVit = stvit_small()

        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])

        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])
        self.conv_curr = nn.Conv2d(128, 128, 3, 2, 1)
        self.e12embedding = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.temp_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.Conv2d(16, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, 2, 1),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU()
        )
        self.conv_interm_embeddings = nn.Conv2d(256, 768, 1)

    def forward(self, x):
        # loc, interm_embeddings = self.image_encoder(x)
        # interm_embeddings(1,64,64,768)
        # loc (1,256,64,64)
        loc = self.temp_conv(x)
        interm_embeddings = self.conv_interm_embeddings(loc)
        interm_embeddings = interm_embeddings.permute(0, 2, 3, 1)
        # loc (1,256,64,64)
        loc = image2patches(loc)
        # loc (4,256,32,32)
        glb = self.STVit(x)
        # glb (1,512,32,32)
        glb = self.trans2glb(glb)
        # glb (1,256,32,32)
        feature = torch.cat((loc, glb), dim=0)
        # loc = self.trans2loc1(loc)

        feature = self.Trans2MultiFeature(feature)
        # input[0] (5,128,128,128)
        # input[1] (5,128,128,128)
        # input[2] (5,256,64,64)
        # input[3] (5,512,32,32)
        # input[4] (5,1024,16,16)

        e5 = self.output5(feature[4])  # (5,128,16,16)
        e4 = self.output4(feature[3])  # (5,128,32,32)
        e3 = self.output3(feature[2])  # (5,128,64,64)
        e2 = self.output2(feature[1])  # (5,128,128,128)
        e1 = self.output1(feature[0])  # (5,128,128,128)

        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        # MCLM
        e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)

        e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4))
        e4 = self.conv4(e4)
        e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
        e3 = self.conv3(e3)
        e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
        e2 = self.conv2(e2)
        e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
        e1 = self.conv1(e1)
        temp_e1 = e1
        curr_embedding1, curr_embedding2 = e1.split([4, 1], dim=0)
        curr_embedding1 = patches2image(curr_embedding1)
        curr_embedding1 = self.conv_curr(curr_embedding1)
        e1 = torch.cat((curr_embedding1, curr_embedding2), dim=1)
        e1 = self.e12embedding(e1)
        # e1(1,256,128,128)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=tokenattmap1,
        )
        # curr_embedding(256, 64, 64)
        # curr_interm(64,64,768)
        # interm_embeddings = interm_embeddings[0]
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=e1,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=interm_embeddings.unsqueeze(0),
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(1024, 1024),
            original_size=(1024, 1024),
        )
        ###
        sideout5 = self.sideout5(e5).cuda()
        sideout4 = self.sideout4(e4)
        sideout3 = self.sideout3(e3)
        sideout2 = self.sideout2(e2)
        sideout1 = self.sideout1(temp_e1)

        glb5 = self.sideout5(glb_e5)
        glb4 = sideout4[-1, :, :, :].unsqueeze(0)
        glb3 = sideout3[-1, :, :, :].unsqueeze(0)
        glb2 = sideout2[-1, :, :, :].unsqueeze(0)
        glb1 = sideout1[-1, :, :, :].unsqueeze(0)

        sideout1 = patches2image(sideout1[:-1]).cuda()
        sideout2 = patches2image(sideout2[:-1]).cuda()  ####(5,c,h,w) -> (1 c 2h,2w)
        sideout3 = patches2image(sideout3[:-1]).cuda()
        sideout4 = patches2image(sideout4[:-1]).cuda()
        sideout5 = patches2image(sideout5[:-1]).cuda()
        if self.training:
            return (sideout5, sideout4, sideout3, sideout2, sideout1,
                    masks, glb5, glb4, glb3, glb2, glb1, tokenattmap4,
                    tokenattmap3, tokenattmap2, tokenattmap1)
        else:
            return masks

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


class Testmodel(nn.Module):
    def __init__(self, imageencoder):
        super().__init__()
        self.sam_encoder = imageencoder


def build_MVEGN2():
    checkpoint_sam = '/22liuchengxu/MVANet-main/MVANet-main/sam_vit_b_01ec64.pth'
    with open(checkpoint_sam, "rb") as f:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(f, map_location=device)
    # print(state_dict)

    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256
    )

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16
    )
    mask_decode = MaskDecoderHQ(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        vit_dim=768,
    )
    net = MV_EdgeGlanceNet2(image_encoder, prompt_encoder, mask_decode)
    info = net.load_state_dict(state_dict, strict=False)
    # loaded_keys = state_dict.keys()
    # for name, param in net.named_parameters():
    #     if name in loaded_keys:
    #         param.requires_grad = False  # 冻结预训练参数
    #     else:
    #         param.requires_grad = True
    return net


if __name__ == '__main__':
    # state_dict = torch.load('/22liuchengxu/MVANet-main/MVANet-main/sam_vit_b_01ec64.pth')
    checkpoint = '/22liuchengxu/MVANet-main/MVANet-main/sam_vit_b_01ec64.pth'
    with open(checkpoint, "rb") as f:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(f, map_location=device)
    # print(state_dict)
    image_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256
    )

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16
    )
    mask_decode = MaskDecoderHQ(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        vit_dim=768,
    )
    net = MV_EdgeGlanceNet2(image_encoder, prompt_encoder, mask_decode)
    # info = net.load_state_dict(state_dict, strict=False)
    # loaded_keys = state_dict.keysf f
    # for name, param in net.named_parameters():
    #   f fs:
    #         param.frequires_grad = False  # 冻结预训练参数
    #         param.requires_grad = True
    # summary(net, input_size=(1, 3, 1024, 1024))f
    device = torch.device("cuda:0")
    net = net.to(device)
    tensor = torch.randn(1, 3, 1024, 1024).cuda()
    out = net(tensor)
    for name, param in net.named_parameters():
        print(f"{name}: {'可训练' if param.requires_grad else '冻结'}")
