import torch
import argparse
import configargparse
import os
import json

class TrainOptions():
    def __init__(self):

        arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
        cfg_parser = configargparse.YAMLConfigFileParser
        description = 'Training Options for Video Capture'
        self.parser = configargparse.ArgParser(
            formatter_class=arg_formatter,
            config_file_parser_class=cfg_parser,
            description=description,
            prog='Training Options')

        self.parser.add_argument('-c', '--config', required=True, is_config_file=True,
                                 help='Configuration File')

        gen_parser = self.parser.add_argument_group('General')
        gen_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                                help='General | Device')
        gen_parser.add_argument('--name', default='tmp',
                                help='General | Name of Training Experiment')
        gen_parser.add_argument('--model-type', default='fusion',
                                choices=['fusion', 'imu', 'video', 'transformer'],
                                help='General | Type of Training Model')
        gen_parser.add_argument('--batch-size', type=int, default=64,
                                help='General | Batch Size')
        gen_parser.add_argument('--lr', type=float, default=3e-5,
                                help='General | Learning Rate')
        gen_parser.add_argument('--lr-finetune', type=float, default=1e-6,
                                help='General | Learning Rate for finetuning')
        gen_parser.add_argument('--betas', type=float, default=0.9,
                                help='Beta1 of the Optimizer')
        gen_parser.add_argument('--optim-type', default='Adam',
                                help='General | Optimizer Type')
        gen_parser.add_argument('--epoch', type=int, default=9999,
                                help='General | Number of Epoch')

        log_parser = self.parser.add_argument_group('Logger')
        log_parser.add_argument('--logdir', default='logger/',
                                help='Logger | Log Output Directory')
        log_parser.add_argument('--write-freq', type=int, default=20,
                                help='Logger | Writing Frequency')
        log_parser.add_argument('--checkpoint-freq', default=25,
                                help='Checkpoint Save Frequency')

        data_parser = self.parser.add_argument_group('Dataset and Dataloader')
        data_parser.add_argument('--calib-noise', type=float, default=1.5e-1,
                                 help='Data Parser | Sensor Calibration Noise STD')
        data_parser.add_argument('--keypoints-noise', type=float, default=5e-3,
                                 help='Data Parser | Keypoints Quality Noise STD')
        data_parser.add_argument('--sex', type=str, default='neutral',
                                 help='Data Parser | Sex of SMPL Body Model')
        data_parser.add_argument('--seq-length', type=int, default=25,
                                 help='Data Parser | AMASS sequence length')
        data_parser.add_argument('--joint-type', default='COCO17', choices=['TC16', 'COCO17'],
                                 help='Data Parser | Keypoints type')
        data_parser.add_argument('--num-workers', type=int, default=0,
                                 help='Data Parser | Number of Data Loading Workers')
        data_parser.add_argument('--pin-memory', default=True,
                                 action=argparse.BooleanOptionalAction,
                                 help='Data Parser | Using PIN Memory')

        network_parser = self.parser.add_argument_group('Network Arguments')
        network_parser.add_argument('--checkpoint', type=str, default='',
                                    help='Network | Pretrained Model Path')
        network_parser.add_argument('--embed-dim', type=int, default=32,
                                    help='Network | Embedding Dimension')
        network_parser.add_argument('--attn-num-heads', type=int, default=8,
                                    help='Network | Number of Attention Heads')
        network_parser.add_argument('--attn-depth', type=int, default=4,
                                    help='Network | Attention Layer Depth')
        network_parser.add_argument('--attn-mlp-ratio', type=int, default=2,
                                    help='Network | MLP Layer Ratio')
        network_parser.add_argument('--drop-rate', type=float, default=0.,
                                    help='Network | Regressor Drop Rate')
        network_parser.add_argument('--attn-drop-rate', type=float, default=0.,
                                    help='Network | Attention Drop Rate')
        network_parser.add_argument('--attn-drop-path-rate', type=float, default=0.,
                                    help='Network | Drop Path Rate')
        network_parser.add_argument('--residual', default='video', type=str,
                                    help='Network | Use unimodal prediction as initial prediction')
        network_parser.add_argument('--supervise-unimodal-until-n-epoch', default=9999, type=int,
                                    help='Network | Supervise unimodal unitl N epochs')

        lw_parser = self.parser.add_argument_group('LW Loss Weight')
        lw_parser.add_argument('--lw-pose', type=float, default=0,
                               help='Loss Weight | Pose Parameter LW')
        lw_parser.add_argument('--lw-betas', type=float, default=0,
                               help='Loss Weight | Shape Parameter LW')
        lw_parser.add_argument('--lw-transl', type=float, default=0,
                               help='Loss Weight | Translation LW')
        lw_parser.add_argument('--lw-kp3d', type=float, default=0,
                               help='Loss Weight | Keypoints LW')

    def parse_args(self, attr_dict=None, print_args=True):
        self.args = self.parser.parse_args()

        if attr_dict is not None:
            for key, value in attr_dict.items():
                setattr(self.args, key, value)

        if print:
            if self.args.device == 'cuda' and not torch.cuda.is_available():
                setattr(self.args, 'device', 'cpu')
                print('CUDA is not available at your machine. Switch to CPU')

            print('\n')
            for idx, (key, value) in enumerate(vars(self.args).items()):
                print(f'{self.parser._actions[idx+1].help:<60} |   {value}')
            print('\n')

        return self.args
