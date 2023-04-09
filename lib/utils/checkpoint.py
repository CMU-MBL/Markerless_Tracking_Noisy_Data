import torch
from torch.utils.tensorboard import SummaryWriter

import json
import time
import os
import os.path as osp

class Timer():
    def __init__(self):
        self.init_time = time.time()
        self.curr_time = time.time()

    def __call__(self):
        delta_T = time.time() - self.init_time
        delta_t = time.time() - self.curr_time
        self._update_time()

        return delta_t, delta_T

    def _update_time(self):
        self.curr_time = time.time()


class Logger():
    def __init__(self, model_dir, write_freq, checkpoint_freq, total_iteration):
        self.model_dir = model_dir
        self.log_dir = osp.join(model_dir, 'log')
        self.write_freq = write_freq
        self.checkpoint_freq = checkpoint_freq
        self.total_iteration = total_iteration
        self.results_fname = osp.join(self.model_dir, 'eval.txt')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            comment='Training Log'
            )

        self.timer = Timer()
        self._read_eval_results()

    def __call__(self, losses, iteration, epoch, is_val=False):
        if not is_val:
            iter_in_epoch = iteration % self.total_iteration
            if iter_in_epoch % self.write_freq == 0 and iter_in_epoch > 0:
                self._write_tensorboard(losses, iteration, epoch)
                self._print_loss_status(losses, iteration, epoch)

        else:
            self._write_tensorboard(losses, epoch)

    def _save_configs(self, args):
        with open(osp.join(self.model_dir, 'configs.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def _get_total_loss(self, losses):
        total_loss = 0
        for l in losses:
            total_loss += losses[l]
        losses['total_loss'] = total_loss

        return losses

    def _print_loss_status(self, losses, iteration, epoch):
        _iteration = iteration % self.total_iteration

        delta_t, delta_T = self.timer()
        _hours = int(delta_T) // 3600
        _minutes = (int(delta_T) % 3600 ) // 60
        _seconds = int(delta_T) % 60

        output_msg = "Epoch: %d | iteration: %d/%d | "%(epoch, _iteration, self.total_iteration)
        output_msg = output_msg + 'time: %.2f sec (overall: %02d:%02d:%02d sec) | Losses: '%(
            delta_t, _hours, _minutes, _seconds)
        for i, l in enumerate(sorted(losses.keys())):
            output_msg = output_msg + '{}. '.format(str(i+1)) + l + \
                ' %.2f,   '%losses[l]

        print(output_msg, flush=True)

    def _add_scalars(self, tag, losses, iteration, epoch):
        total_iteration = iteration + (epoch - 1) * self.total_iteration
        self.writer.add_scalars(tag, losses, total_iteration)

    def _write_tensorboard(self, losses, iteration, epoch):
        total_iteration = iteration + (epoch - 1) * self.total_iteration
        for l in losses:
            self.writer.add_scalar(l, losses[l], total_iteration)

    def _read_eval_results(self):
        self.results = []
        if osp.exists(self.results_fname):
            f = open(self.results_fname, 'r')
            results = f.readlines()[1:]
            for result in results:
                self.results.append(float(result.split()[-1]))

    def _save_eval_results(self, eval_dict):
        with open(self.results_fname, 'a') as write_file:
            if len(self.results) == 0:
                columns = ''.join(
                    ['{:20}'.format(key) for key in eval_dict.keys()])
                write_file.writelines(columns)
            eval_results_str = ''.join(
                ['{:20}'.format(str(eval_dict['epoch']))] + \
                ['{:20}'.format(
                    '%.2f'%val) for val in eval_dict.values()][1:])
            write_file.writelines('\n' + eval_results_str)
        self.results.append(eval_dict[list(eval_dict.keys())[-1]])

    def _save_checkpoint(self, model, optimizer_dict, eval_dict, epoch):
        if eval_dict is not None:
            self._save_eval_results(eval_dict)
            if eval_dict[list(eval_dict.keys())[-1]] == min(self.results):
                suffix = 'best_'
            elif epoch % self.checkpoint_freq == 0:
                suffix = 'epoch_%03d_'%epoch
            else: suffix = 'last_'
        else: suffix = 'last_'
        fname = suffix + 'checkpoint.pt'
        checkpoint_path = osp.join(self.model_dir, fname)

        checkpoint = {'model': model.state_dict()}
        for key, value in optimizer_dict.items():
            checkpoint[key] = value.state_dict()

        checkpoint['epoch'] = epoch + 1
        torch.save(checkpoint, checkpoint_path)


def build_logger(logdir, name, write_freq, total_iteration):
    logger = Logger(model_dir=osp.join(logdir, name),
                    log_dir=osp.join(logdir, name, 'log'),
                    write_freq=write_freq,
                    total_iteration=total_iteration,
                    )

    return logger
