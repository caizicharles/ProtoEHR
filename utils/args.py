import argparse
import yaml
from .misc import get_time_str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mimiciv')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--triplet_ratio', type=float, default=1.0)
    
    parser.add_argument('--processed_data_path', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)

    optimizer_hyp_search_args = {
        'lr': 0.0001,
        'weight_decay': 0.,
    }

    scheduler_hyp_search_args = {'gamma': 0.995}

    model_hyp_search_args = {
                       'start_embed_dim': 64,
                       'gnn_type': 'CompGCN',
                       'gnn_layer': 2,
                       'hidden_dim': 64,
                       'ff_dim': 512,
                       'head_num': 4,
                       'depth': 2,
                       'max_weeks_between': 630,
                       'join_method': 'add',
                       'dropout': 0.1,
                       'patient_proto_num': 32,
                       'visit_proto_num': 32,
                       'code_proto_num': 32,
                       'attn_iter': 1,
                       'prototype_learning_mode': 'QKV',
                       'prototype_prediction_mode': 'simC',
                       'commitment_cost': 0.25,
                       'softmax_temp': 1.}
    
    for hyp, default_val in optimizer_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)
    for hyp, default_val in scheduler_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)
    for hyp, default_val in model_hyp_search_args.items():
        parser.add_argument(f'--{hyp}', type=type(default_val), default=default_val)

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Config', add_help=False)
    config_parser.add_argument('-c', '--config', default=None, type=str, help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    
    if 'data_preparation' not in args.task and args.pretrained == None:
        if args.optimizer is not None:
            for hyp in optimizer_hyp_search_args.keys():
                value = getattr(args, hyp, None)
                if value is not None:
                    optimizer_hyp_search_args[hyp] = value

            args.optimizer['args'] = optimizer_hyp_search_args

        if args.scheduler is not None:
            for hyp in scheduler_hyp_search_args.keys():
                value = getattr(args, hyp, None)
                if value is not None:
                    scheduler_hyp_search_args[hyp] = value

            args.scheduler['args'] = scheduler_hyp_search_args

        if args.model['model_type'] == 'base':
            for hyp in model_hyp_search_args.keys():
                value = getattr(args, hyp, None)
                if value is not None:
                    model_hyp_search_args[hyp] = value

            args.model['args'] = model_hyp_search_args
        
        
    elif 'data_preparation' in args.task and args.pretrained == '':
        task_name = args.task.split('-')[1]
        args.task = task_name

    args.start_time = get_time_str()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
