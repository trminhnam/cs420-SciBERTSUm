from __future__ import division

import argparse
import os

from others.log import init_logger
from train_extractive import train_ext, validate_ext, test_ext

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../bert_data')
    # parser.add_argument("-model_path", default='../models/') #fix
    # parser.add_argument("-tensorboard_log_path", default='../tensorboard_log/')# fix
    parser.add_argument("-model_path", default='../alakimodels/')  # fix
    parser.add_argument("-tensorboard_log_path", default='../alakitensorboard_log/')  # fix
    parser.add_argument("-result_path", default='../results/')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-test_batch_size", default=1, type=int)

    # parser.add_argument("-max_pos", default=20480, type=int) #fix
    # parser.add_argument("-chunk_size", default=3072, type=int) # fix
    parser.add_argument("-max_pos", default=10240, type=int) #fix
    parser.add_argument("-chunk_size", default=512, type=int) # fix
    parser.add_argument("-use_interval", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # global attention params
    parser.add_argument('-global_attention', default=1, type=int, choices=[0,1,2], help=" global attention types:0,1,2. 0: no global attention, 1: global attention at random indices, 2: global attention at the beginning and end of the sections  ")
    parser.add_argument('-global_attention_ratio', default=0.2, type=float, help="ratio of global attention indices chosen at random")

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int, help="number of extractive encoder layers")
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=4, type=int, help="number of attention head in each encoder layer")
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha", default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-max_tgt_len", default=5000, type=int) # max size for the slide

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-visible_gpus', default='2', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/slide_gen.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    if args.mode == 'train':
        train_ext(args, device_id)
    elif args.mode == 'validate':
        validate_ext(args, device_id)
    if args.mode == 'test':
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        test_ext(args, device_id, cp, step)

