import torch
from torch import nn

from einops import rearrange
from functools import partial

from configs import configs as cfg
from lib.utils import transforms
from lib.models.layers import (MultiHeadSelfAttentionBlock, 
                               MultiHeadAttentionBlock)


class Network(nn.Module):
    def __init__(self, args, mods=[]):
        super().__init__()

        """ Test network for Transformer of previous architecture
        """

        self.model_type = args.model_type
        self.joint_type = args.joint_type

        self._get_attention_attributes(args)
        self.residual = args.residual
        
        if args.model_type in ['video', 'fusion']: mods.append('video')
        if args.model_type in ['imu', 'fusion']: mods.append('imu')
        l_mod = args.seq_length
        d_emb = args.embed_dim
        n_smpl = len(cfg.SMPL.MAIN_JOINTS)

        # Unimodal network
        for mod in mods:
            d_mod = 3 if mod == 'video' else 6
            n_mod = int(args.joint_type[-2:]) if mod == 'video' else len(cfg.IMU.LIST)
            self._make_init_conv_embedding(mod, d_mod, d_emb)
            self._make_attn_modules(mod, 'spatial', n_mod, d_emb, 'self')
            self._make_attn_modules(mod, 'temporal', l_mod, d_emb, 'self')
            self._make_smpl_mapper(mod, n_mod, d_emb, n_smpl)
            self._make_decoder_layers(mod, d_emb, n_smpl)


        if args.model_type != 'fusion': return

        # Fusion network
        self._make_attn_modules('VtoI', 'spatial', n_smpl, d_emb, 'cross')
        self._make_attn_modules('ItoV', 'spatial', n_smpl, d_emb, 'cross')
        self._make_attn_modules('VtoI', 'temporal', l_mod, d_emb, 'cross')
        self._make_attn_modules('ItoV', 'temporal', l_mod, d_emb, 'cross')

        # Final self attention
        self._make_attn_modules('fusion', 'temporal', l_mod, d_emb, 'self')
        self._make_decoder_layers('fusion', d_emb, n_smpl)

        # self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


    def _get_attention_attributes(self, args):
        net_attrs = {'n_head': 'attn_num_heads',
                     'n_block': 'attn_depth',
                     'r_drop': 'drop_rate',
                     'r_attn_drop': 'attn_drop_rate',
                     'r_drop_path': 'attn_drop_path_rate',
                     'r_mlp': 'attn_mlp_ratio',
                     }
        for k, v in net_attrs.items():
            setattr(self, k, getattr(args, f'{v}'))


    def _make_init_conv_embedding(self, modality, d_mod, d_emb):
        d_hid = d_emb * 2
        setattr(self, f'{modality}_to_embedding_space',
                nn.Sequential(
                    nn.Conv1d(d_mod, d_hid, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU(), nn.BatchNorm1d(d_hid),
                    nn.Conv1d(d_hid, d_hid, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(), nn.BatchNorm1d(d_hid),
                    nn.Conv1d(d_hid, d_emb, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.ReLU(), nn.BatchNorm1d(d_emb)))


    def _make_decoder_layers(self, modality, d_emb, n_smpl):
        d_pose = 6
        d_shape = 10
        d_transl = 3
        expansion = 2
        expansion_pose = 4

        add = d_pose if modality == 'fusion' else 0
        setattr(self, modality + '_pose_decoder',
                nn.Sequential(nn.Linear(d_emb + add, d_emb * expansion_pose),
                              nn.GELU(), nn.Dropout(self.r_drop),
                              nn.Linear(d_emb * expansion_pose, d_emb * expansion_pose),
                              nn.GELU(), nn.Dropout(self.r_drop),
                              nn.Linear(d_emb * expansion_pose, d_pose)))

        if modality == self.model_type:
            self.final_fc_layers = nn.Sequential(
                nn.Linear(n_smpl * d_emb, n_smpl * d_emb * expansion, bias=False),
                nn.GELU(), nn.Dropout(self.r_drop),
                nn.Linear(n_smpl * d_emb * expansion, n_smpl * d_emb, bias=False),
                nn.GELU(), nn.Dropout(self.r_drop))
            self.shape_decoder = nn.Linear(n_smpl * d_emb, d_shape)
            self.transl_decoder = nn.Linear(n_smpl * d_emb, d_transl)


    def _make_smpl_mapper(self, modality, n_mod, d_emb, n_smpl):
        d_in = d_emb * n_mod
        d_hid = d_in * 2
        d_out = d_emb * n_smpl
        setattr(self, modality + '_to_smpl_space',
                nn.Sequential(nn.Linear(d_in, d_hid, bias=False),
                              nn.GELU(), nn.Dropout(self.r_drop),
                              nn.Linear(d_hid, d_out)))

    def _make_attn_modules(self, modality, axis, axis_len, embed_dim, attn_type):

        prefix = '_'.join((modality, axis))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, self.r_drop_path, self.n_block)]
        block = {'self': MultiHeadSelfAttentionBlock, 'cross': MultiHeadAttentionBlock}[attn_type]

        # if not (mode == 'Encoder' and axis == 'Temporal'):
        setattr(self, prefix + '_pre_layer',
                nn.Linear(embed_dim, embed_dim, bias=False))
        setattr(self, prefix + '_positional_embedding',
                nn.Parameter(torch.zeros(1, axis_len, embed_dim)))
        setattr(self, prefix + '_norm_layer', norm_layer(embed_dim))
        setattr(self, prefix + '_dropout', nn.Dropout(self.r_attn_drop))
        setattr(self, prefix + '_attention_blocks',
                nn.ModuleList([block(dim=embed_dim, num_heads=self.n_head,
                attn_drop=self.r_attn_drop, drop_path=dpr[i], norm_layer=norm_layer,
                mlp_ratio=self.r_mlp) for i in range(self.n_block)]))


    def forward_attn(self, x, modality, axis, y=None):
        """Forward pass of Attention algorithm"""

        prefix = '_'.join((modality, axis))
        x = getattr(self, prefix + '_pre_layer')(x)
        x += getattr(self, prefix + '_positional_embedding')
        x = getattr(self, prefix + '_dropout')(x)

        for blk in getattr(self, prefix + '_attention_blocks'):
            x, attn = blk(x, y=y)
        x = getattr(self, prefix + '_norm_layer')(x)

        return x


    def _get_unimodal_regression(self, x, modality):
        # Dimensions
        b, f, m = x.shape[:3]; s = len(cfg.SMPL.MAIN_JOINTS)

        # Map To Embedding Space
        org_kp = x.clone().detach()
        x = rearrange(x, 'b f m d -> (b m) d f')
        x = getattr(self, modality + '_to_embedding_space')(x)

        # Spatial self attention
        x = rearrange(x, '(b m) d f -> (b f) m d', b=b)
        x = self.forward_attn(x, modality, 'spatial')

        # Temporal self attention
        x = rearrange(x, '(b f) m d -> (b m) f d', b=b, f=f)
        x = self.forward_attn(x, modality, 'temporal')

        # Map To SMPL space
        x = rearrange(x, '(b m) f d -> b f (m d)', b=b)
        x = getattr(self, modality + '_to_smpl_space')(x)
        x = rearrange(x, 'b f (s d) -> (b f) s d', s=s)

        pose = getattr(self, modality + '_pose_decoder')(x)
        pose = transforms.rot6d_to_rotmat(pose).contiguous().view(-1, s, 3, 3)
        pose = rearrange(pose, '(b f) s d1 d2 -> b f s d1 d2', b=b)
        emb_x = rearrange(x, '(b f) s d -> b f s d', b=b)

        if self.model_type == 'video':
            x = rearrange(x, '(b f) s d -> b f (s d)', b=b)
            x = self.final_fc_layers(x)
            shape = self.shape_decoder(x)
            transl = self.transl_decoder(x)
        else: shape, transl = None, None

        return (pose, shape, transl), emb_x


    def _get_multimodal_regression(self, x_video, x_imu, pose_b):
        b, f = x_video.shape[:2]; s = len(cfg.SMPL.MAIN_JOINTS)

        # # Spatial Fusion
        # Transform Video and IMU representation in space domain
        x_video = rearrange(x_video, 'b f s d -> (b f) s d')
        x_imu = rearrange(x_imu, 'b f s d -> (b f) s d')
        x_VtoI = self.forward_attn(x_imu, 'VtoI', 'spatial', x_video)
        x_ItoV = self.forward_attn(x_video, 'ItoV', 'spatial', x_imu)

        # # Temporal Fusion
        # Transform Video and IMU representation in time domain
        x_imu = rearrange(x_VtoI, '(b f) s d -> (b s) f d', b=b)
        x_video = rearrange(x_ItoV, '(b f) s d -> (b s) f d', b=b)
        x_VtoI = self.forward_attn(x_imu, 'VtoI', 'temporal', x_video)
        x_ItoV = self.forward_attn(x_video, 'ItoV', 'temporal', x_imu)

        # Aggregate modalities
        x = torch.stack((x_ItoV, x_VtoI)).max(0)[0]

        # Temporal Self Attention
        x = self.forward_attn(x, 'fusion', 'temporal')

        # Decode SMPL Pose
        x = rearrange(x, '(b s) f d -> (b f) s d', b=b)
        pose = rearrange(pose_b, 'b f s d -> (b f) s d')
        for _ in range(4):
            x_cat = torch.cat((pose, x), dim=-1)
            pose = self.fusion_pose_decoder(x_cat) + pose

        pose = pose.contiguous().view(b, f, -1)
        pose = transforms.rot6d_to_rotmat(pose).contiguous().view(-1, s, 3, 3)
        pose = rearrange(pose, '(b f) s d1 d2 -> b f s d1 d2', b=b)

        # Decode SMPL Shape and Translation
        x = rearrange(x, '(b f) s d -> b f (s d)', b=b)
        x = self.final_fc_layers(x)
        shape = self.shape_decoder(x)
        transl = self.transl_decoder(x)

        return (pose, shape, transl), None


    def forward_video(self, inputs):
        init_transl = inputs['kp3d'][:, :, cfg.KEYPOINTS.PELVIS_IDX[self.joint_type]]
        if init_transl.shape[-2] == 2: init_transl = init_transl.mean(-2)
        else: init_transl = init_transl.unsqueeze(-2)
        
        input_video = inputs['kp3d'] - init_transl.unsqueeze(-2)
        (pose, shape, transl), _ = self._get_unimodal_regression(input_video, 'video')
        if transl is not None:
            transl = transl + init_transl

        return (pose, shape, transl), None


    def forward_imu(self, inputs):
        input_imu = inputs['imu']
        (pose, _, _), _ = self._get_unimodal_regression(input_imu, 'imu')

        return (pose, None, None), None


    def forward_fusion(self, inputs):
        init_transl = inputs['kp3d'][:, :, cfg.KEYPOINTS.PELVIS_IDX[self.joint_type]]
        if init_transl.shape[-2] == 2: init_transl = init_transl.mean(-2)
        
        input_video = inputs['kp3d'] - init_transl.unsqueeze(-2)
        input_imu = inputs['imu']

        (pose_v, _, _), x_video = self._get_unimodal_regression(input_video, 'video')
        (pose_i, _, _), x_imu = self._get_unimodal_regression(input_imu, 'imu')

        if self.residual == 'video':
            pose_b = pose_v[..., :-1].contiguous().reshape(*pose_v.shape[:3], 6)
        else:
            pose_b = pose_i[..., :-1].contiguous().reshape(*pose_v.shape[:3], 6)
        
        (pose, shape, transl), _ = self._get_multimodal_regression(x_video, x_imu, pose_b)
        transl = transl + init_transl

        return (pose, shape, transl), (pose_v, pose_i)


    def forward(self, inputs):
        if self.model_type == 'video':
            return self.forward_video(inputs)

        elif self.model_type == 'imu':
            return self.forward_imu(inputs)

        elif self.model_type == 'fusion':
            return self.forward_fusion(inputs)
