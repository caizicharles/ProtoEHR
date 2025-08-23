import random
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
import tensorboardX
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *
from dataset.dataset import MIMICIVBaseDataset, split_by_patient
from dataloader.dataloader import DataLoader
from model.model import MODELS
from metrics.metrics import METRICS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


class Args:

    def __init__(self, **entries):
        self.__dict__.update(entries)


def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def load_all(processed_data_path: str, dataset_name: str, triplet_method: str):

    node_id_to_name = read_pickle_file(processed_data_path, f'{dataset_name}_node_id_to_name.pickle')
    triplet_id_to_info = read_pickle_file(processed_data_path,
                                          f'{dataset_name}_{triplet_method}_triplet_id_to_info.pickle')
    diagnoses_map = read_csv_file(processed_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(processed_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(processed_data_path, 'prescriptions_code.csv')

    filtered_patients = read_pickle_file(processed_data_path, f'{dataset_name}_filtered.pickle')
    full_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_padded.pickle')
    full_graph = read_pickle_file(processed_data_path, f'{dataset_name}_{triplet_method}_graph.pickle')

    return {
        'node_id_to_name': node_id_to_name,
        'triplet_id_to_info': triplet_id_to_info,
        'diagnoses_map': diagnoses_map,
        'procedures_map': procedures_map,
        'prescriptions_map': prescriptions_map,
        'filtered_patients': filtered_patients,
        'full_dataset': full_dataset,
        'full_graph': full_graph
    }


def task_configuring_model(args, node_id_to_name, prescriptions):

    task = args.task

    if task == 'mortality_prediction' or task == 'readmission_prediction':
        out_dim = 1
        class_num = 2

    elif task == 'drug_recommendation':
        out_dim = len(prescriptions)
        class_num = len(prescriptions)

    elif task == 'los_prediction':
        out_dim = 10
        class_num = 10

    elif task == 'pretrain':
        out_dim = len(node_id_to_name) + 1
        class_num = len(node_id_to_name) + 1

    return out_dim, class_num


def t_SNE_visualization(embed, label=None):

    scaler = StandardScaler()
    tsne = TSNE(n_components=2, random_state=0, perplexity=3, n_iter=500)
    embed_normalized = scaler.fit_transform(embed)
    tensor_tsne = tsne.fit_transform(embed_normalized)

    if label is not None:
        classes = torch.unique(label.flatten()).tolist()
    else:
        label = [i for i in range(embed.size(0))]
        classes = label

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], c=label, cmap='viridis', s=50)
    plt.colorbar(scatter, ticks=classes, label='Class')
    plt.title('t-SNE Visualization of 2D Tensor with Class Coloring')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()


def visualize_embed(embed: list, label: list, mask: list, prototype=None):

    embed = torch.cat(embed, dim=0)

    B, V, D = embed[0].shape

    label_all = []
    embed_all = []

    for batch_embed, batch_label, batch_mask in zip(embed, label, mask):
        visit_mask = batch_mask.all(dim=-1)
        visit_mask = ~visit_mask
        visit_mask_for_label = visit_mask.reshape(B * V)
        visit_mask_for_embed = visit_mask.unsqueeze(-1).expand(-1, -1, D)
        visit_mask_for_embed = visit_mask_for_embed.reshape(B * V, D)

        visit_label = batch_label.expand(-1, V).reshape(B * V)
        visit_embed = batch_embed.reshape(B * V, D)

        filtered_label = visit_label[visit_mask_for_label].to(int)
        filtered_embed = visit_embed[visit_mask_for_embed].view(-1, D)
        label_all.append(filtered_label)
        embed_all.append(filtered_embed)

    label_all = torch.cat(label_all, dim=0)
    embed_all = torch.cat(embed_all, dim=0)

    if prototype is not None:
        prototype_labels = torch.arange(prototype.size(0)) + 2
        label_all = torch.cat((label_all, prototype_labels), dim=0)
        embed_all = torch.cat((embed_all, prototype), dim=0)

    classes = torch.unique(label_all.flatten()).tolist()

    scaler = StandardScaler()
    # pca = PCA(n_components=D // 8)
    tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=500)
    embed_all_normalized = scaler.fit_transform(embed_all)
    # embed_all_pca = pca.fit_transform(embed_all_normalized)
    tensor_tsne = tsne.fit_transform(embed_all_normalized)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], c=label_all, cmap='viridis', s=50)

    plt.colorbar(scatter, ticks=classes, label='Class')
    plt.title('t-SNE Visualization of 2D Tensor with Class Coloring')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()


def run(args):

    writer = init_logger(args, use_tensorboard=False)

    logger.info(f'Explanation begins...')
    logger.info(f"Load pretrained model from {args.pretrained}")

    ckpt = torch.load(args.pretrained, map_location=device)
    args = ckpt['args']
    args = yaml.safe_load(args)
    args = Args(**args)

    logger.info(f'Task: {args.task}')
    logger.info(f'Triplet method: {args.triplet_method}')
    logger.info(f'Model: {args.model["name"]}')

    seed_everything(args.seed)

    file_lib = load_all(args.processed_data_path, args.dataset, args.triplet_method)
    logger.info('Completed file loading')

    diagnoses_maps = format_code_map(file_lib['diagnoses_map'])
    procedures_maps = format_code_map(file_lib['procedures_map'])
    prescriptions_maps = format_code_map(file_lib['prescriptions_map'])

    ratio = [args.train_proportion, args.val_proportion, args.test_proportion]
    train_patients, val_patients, test_patients = split_by_patient(file_lib['full_dataset'], ratio)

    train_dataset = MIMICIVBaseDataset(patients=train_patients,
                                       prescriptions_code_to_name=prescriptions_maps[0],
                                       filtered_patients=file_lib['filtered_patients'],
                                       task=args.task,
                                       graph=file_lib['full_graph'])
    val_dataset = MIMICIVBaseDataset(patients=val_patients,
                                     prescriptions_code_to_name=prescriptions_maps[0],
                                     filtered_patients=file_lib['filtered_patients'],
                                     task=args.task,
                                     graph=file_lib['full_graph'])
    test_dataset = MIMICIVBaseDataset(patients=test_patients,
                                      prescriptions_code_to_name=prescriptions_maps[0],
                                      filtered_patients=file_lib['filtered_patients'],
                                      task=args.task,
                                      graph=file_lib['full_graph'])
    logger.info('Dataset ready')

    train_loader = DataLoader(dataset=train_dataset,
                              graph=file_lib['full_graph'],
                              model_name=args.model['name'],
                              task=args.task,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            graph=file_lib['full_graph'],
                            model_name=args.model['name'],
                            task=args.task,
                            batch_size=args.val_batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             graph=file_lib['full_graph'],
                             model_name=args.model['name'],
                             task=args.task,
                             batch_size=args.test_batch_size,
                             shuffle=False)
    logger.info('DataLoader ready')

    out_dim, class_num = task_configuring_model(args, file_lib['node_id_to_name'], prescriptions_maps[0])

    multi_level_embed = None
    if args.model['load_embed']:
        multi_level_embed = torch.load(args.model['load_embed'])['embed']

    model_configs = args.model['args'] | {
        'device': device,
        'num_nodes': len(file_lib['node_id_to_name']),
        'num_edges': len(file_lib['triplet_id_to_info']),
        'visit_thresh': args.visit_thresh,
        'visit_code_num': args.code_thresh,
        'out_dim': out_dim,
        'class_num': class_num,
        'global_node_attr': file_lib['full_graph'].x,
        'global_edge_attr': file_lib['full_graph'].edge_attr,
        'global_edge_index': file_lib['full_graph'].edge_index,
        'global_edge_ids': file_lib['full_graph'].edge_ids,
        'multi_level_embed': multi_level_embed
    }
    model = MODELS[args.model['name']](model_configs)
    model.to(device)

    model.load_state_dict(ckpt['model'], strict=False)

    global_iter_idx = [0]

    if ckpt['iter'] is not None:
        global_iter_idx[0] = ckpt['iter']

    logger.info('Model ready')

    embed_all, label_all, attn_mask_all, scores_all, pred_all, prototypes = single_validate(
        model,
        args.task,
        val_loader,
        global_iter_idx,
        metrics=[METRICS[metric](args.task) for metric in args.metrics],
        writer=writer)

    label_all = torch.cat(label_all, dim=0).squeeze(-1)

    patient_prototypes = prototypes['patient_prototypes']
    visit_prototypes = prototypes['visit_prototypes']
    code_prototypes = prototypes['code_prototypes']

    patient_embed = embed_all['patient_embed']
    visit_embed = embed_all['visit_embed']
    code_embed = embed_all['code_embed']

    patient_embed = torch.cat(patient_embed, dim=0)
    visit_embed = torch.cat(visit_embed, dim=0)

    # t_SNE_visualization(patient_embed, label=label_all)
    # t_SNE_visualization(visit_embed, label=label_all)
    # t_SNE_visualization(code_embed, label=label_all)

    # visualize_embed(embed_all, label_all, attn_mask_all, prototypes)


def single_validate(model, task, dataloader, global_iter_idx, metrics=[], writer=None):

    model.eval()
    prob_all = []
    target_all = []
    embed_all = {'patient_embed': [], 'visit_embed': [], 'code_embed': []}
    attn_mask_all = []
    scores_all = {'patient_prototype_scores': [], 'visit_prototype_scores': [], 'code_prototype_scores': []}

    for _, data in enumerate(tqdm(dataloader)):
        data = data.to(device)

        with torch.no_grad():
            additional_data = {
                'cat_node_ids': data.cat_node_ids,
                'cat_edge_ids': data.cat_edge_ids,
                'cat_edge_index': data.cat_edge_index,
                'cat_edge_attr': data.cat_edge_attr,
                'visit_nodes': data.visit_nodes,
                'visit_node_type': data.visit_node_type,
                'ehr_nodes': data.ehr_nodes,
                'batch': data.batch,
                'batch_patient': data.batch_patient
            }

            output = model(node_ids=data.visit_node_ids,
                           edge_idx=data.global_edge_index,
                           edge_attr=data.global_edge_attr,
                           visit_times=data.visit_rel_times,
                           visit_order=data.visit_order,
                           attn_mask=data.attn_mask,
                           **additional_data)

            out = output['logits']
            embeddings = output['embeddings']
            scores = output['scores']

            labels = data.labels
            if task == 'pretrain':
                labels_2 = data.labels_additional

                visit_mask = data.attn_mask.all(dim=-1)
                visit_mask = ~visit_mask
                visit_mask = visit_mask.unsqueeze(-1).expand(-1, -1, labels.size(-1))

                labels = labels[visit_mask].view(-1, labels.size(-1))
                labels_2 = labels_2[visit_mask].view(-1, labels.size(-1))

            attn_mask_all.append(data.attn_mask.cpu())

            if task == 'los_prediction':
                probability = F.softmax(out[0], dim=-1)
            else:
                probability = torch.sigmoid(out[0])

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

            if embeddings is not None:
                for k, v in embeddings.items():
                    if v is not None:
                        embed_all[k].append(v.cpu())

            if scores is not None:
                for k, v in scores.items():
                    if v is not None:
                        scores_all[k].append(v.cpu())

    pred_all = prob_all
    label_all = target_all

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        metric.log(score, logger, writer=writer, global_iter=global_iter_idx[0], name_prefix='val/')

    if task == 'mortality_prediction' or task == 'readmission_prediction':
        for idx, pred in enumerate(pred_all):
            pred_all[idx] = (pred >= 0.5).to(int)
    elif task == 'los_prediction':
        for idx, pred in enumerate(pred_all):
            pred_all[idx] = torch.argmax(pred, dim=-1)

    prototypes = output['prototypes']
    if prototypes is not None:
        for k, v in prototypes.items():
            if v is not None:
                prototypes[k] = v.cpu()

    return embed_all, label_all, attn_mask_all, scores_all, pred_all, prototypes


if __name__ == '__main__':
    args = get_args()
    run(args=args)
