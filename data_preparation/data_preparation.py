import random
import logging
import time
import os
import os.path as osp
import sys
import math
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch_geometric.utils import subgraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

import dgl

from data_preparation.dataset import TripletDataset
from data_preparation.classifier_model import Triplet_Classifier
from trainer.optimizer import OPTIMIZERS
from trainer.scheduler import SCHEDULERS
from trainer.criterion import CRITERIONS
from metrics.metrics import METRICS
import mlflow
from mlflow.tracking import MlflowClient

from utils.args import get_args
from utils.utils import *
from utils.misc import save_params, init_logger
from data_preparation.data_preparation_utils import get_BERT_embeddings, verify_embeddings, encode_for_BERT, verify_clusters, merge_edge_type

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dependent_files(raw_data_path: str, processed_data_path: str, dataset_name: str):
    if dataset_name == 'mimiciv':
        patient_info = read_csv_file(processed_data_path, 'patients.csv')
    else:
        patient_info = read_csv_file(processed_data_path, 'PATIENTS.csv')

    diagnoses_map = read_csv_file(processed_data_path, 'diagnoses_code.csv')
    procedures_map = read_csv_file(processed_data_path, 'procedures_code.csv')
    prescriptions_map = read_csv_file(processed_data_path, 'prescriptions_code.csv')
    node_id_to_name = read_pickle_file(processed_data_path, f'{dataset_name}_node_id_to_name.pickle')
    node_name_to_id = read_pickle_file(processed_data_path, f'{dataset_name}_node_name_to_id.pickle')
    visit_id_to_nodes = read_pickle_file(processed_data_path, f'{dataset_name}_visit_id_to_nodes.pickle')
    phenotype_names = read_pickle_file(processed_data_path, 'ccs_phenotypes.pickle')

    return {
        'node_id_to_name': node_id_to_name,
        'node_name_to_id': node_name_to_id,
        'patient_info': patient_info,
        'diagnoses_map': diagnoses_map,
        'procedures_map': procedures_map,
        'prescriptions_map': prescriptions_map,
        'visit_id_to_nodes': visit_id_to_nodes,
        'phenotype_names': phenotype_names
    }


def get_dataset(raw_data_path: str, processed_data_path: str, dataset_name: str, save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}.pickle')):
        dataset = read_pickle_file(processed_data_path, dataset_name)

    else:
        if dataset_name == "mimiciv":
            raw_data_path = osp.join(raw_data_path, '2.2/hosp/')
            dataset = MIMIC4Dataset(root=raw_data_path,
                                    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                                    code_mapping={
                                        "NDC": ("ATC", {
                                            "target_kwargs": {
                                                "level": 3
                                            }
                                        }),
                                        "ICD9CM": "CCSCM",
                                        "ICD9PROC": "CCSPROC",
                                        "ICD10CM": "CCSCM",
                                        "ICD10PROC": "CCSPROC"
                                    },
                                    dev=False,
                                    refresh_cache=True)

        elif dataset_name == 'mimiciii':
            dataset = MIMIC3Dataset(root=raw_data_path,
                                    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
                                    code_mapping={
                                        "NDC": ("ATC", {
                                            "target_kwargs": {
                                                "level": 3
                                            }
                                        }),
                                        "ICD9CM": "CCSCM",
                                        "ICD9PROC": "CCSPROC"
                                    },
                                    dev=False,
                                    refresh_cache=True)

    if save_dataset:
        save_with_pickle(dataset, processed_data_path, f'{dataset_name}.pickle')

    return dataset


def filter_dataset(patients: dict,
                   patient_info: dict,
                   filter_args: dict,
                   processed_data_path: str,
                   dataset_name: str,
                   save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_filtered.pickle')):
        filtered_patients = read_pickle_file(processed_data_path, f'{dataset_name}_filtered.pickle')

    else:
        age_thresh_low = filter_args['age_thresh_low']
        age_thresh_high = filter_args['age_thresh_high']
        code_thresh = filter_args['code_thresh']
        visit_thresh = filter_args['visit_thresh']

        filtered_patients = patients.copy()

        for (id, patient) in tqdm(patients.items(), desc='Filtering dataset'):
            # Filter for Age
            if dataset_name == 'mimiciv':
                pos = patient_info['subject_id'].index(id)
                age = int(patient_info['anchor_age'][pos])

                if age < age_thresh_low or age > age_thresh_high:
                    del filtered_patients[id]
                    continue

            elif dataset_name == 'mimiciii':
                dob = patient.birth_datetime.year

                for visit in patient:
                    dov = visit.encounter_time.year
                    age = dov - dob

                    if age < age_thresh_low or age > age_thresh_high:
                        break

                if age < age_thresh_low or age > age_thresh_high:
                    del filtered_patients[id]
                    continue

            # Filter for Visit Num
            if len(patient) <= 1 or len(patient) > visit_thresh:
                del filtered_patients[id]
                continue

            # Filter for Discharge Record Error
            all_discharge_status = []
            encounter_times = []
            discharge_times = []

            for visit in patient:
                all_discharge_status.append(visit.discharge_status)
                encounter_times.append(visit.encounter_time)
                discharge_times.append(visit.discharge_time)

            if sum(all_discharge_status) > 1:
                del filtered_patients[id]
                continue

            if len(set(encounter_times)) != len(encounter_times) or len(set(discharge_times)) != len(discharge_times):
                del filtered_patients[id]
                continue

            # Filter for Code Num
            for visit in patient:
                if dataset_name == 'mimiciv':
                    num_diagnoses = len(visit.get_code_list('diagnoses_icd'))
                    num_procedures = len(visit.get_code_list('procedures_icd'))
                    num_prescriptions = len(visit.get_code_list('prescriptions'))
                elif dataset_name == 'mimiciii':
                    num_diagnoses = len(visit.get_code_list('DIAGNOSES_ICD'))
                    num_procedures = len(visit.get_code_list('PROCEDURES_ICD'))
                    num_prescriptions = len(visit.get_code_list('PRESCRIPTIONS'))

                num_codes = num_diagnoses + num_procedures + num_prescriptions

                if num_codes > code_thresh or num_codes == 0:
                    del filtered_patients[id]
                    break

    if save_dataset:
        save_with_pickle(filtered_patients, processed_data_path, f'{dataset_name}_filtered.pickle')

    return filtered_patients


def sort_visits(patients: dict, processed_data_path: str, dataset_name: str, save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_sorted.pickle')):
        sorted_patients = read_pickle_file(processed_data_path, f'{dataset_name}_sorted.pickle')

    else:
        sorted_patients = {}

        SORTED_PATIENT_TEMPLATE = {
            'patient_id': None,
            'visits': [],
            'encounter_times': [],
            'discharge_times': [],
            'mortality': False,
            'mortality_visit_discharge_time': None
        }

        for patient_id, patient in tqdm(patients.items(), desc='Sorting Patients'):
            single_patient = deepcopy(SORTED_PATIENT_TEMPLATE)
            single_patient['patient_id'] = patient_id

            visit_ids = []
            encounter_times = []
            discharge_times = []
            all_discharge_status = []

            for visit in patient:
                visit_ids.append(visit.visit_id)
                encounter_times.append(visit.encounter_time)
                discharge_times.append(visit.discharge_time)
                all_discharge_status.append(visit.discharge_status)

            if 1 in all_discharge_status:
                assert sum(all_discharge_status) <= 1, '2 motality in 1 patient?'
                single_patient['mortality'] = True
                single_patient['mortality_visit_discharge_time'] = discharge_times[all_discharge_status.index(1)]

            sorted_indices = sorted(range(len(encounter_times)), key=lambda i: encounter_times[i])

            sorted_visit_ids = [visit_ids[i] for i in sorted_indices]
            for vid in sorted_visit_ids:
                single_patient['visits'].append(patient.get_visit_by_id(vid))

            single_patient['encounter_times'] = [encounter_times[i] for i in sorted_indices]
            single_patient['discharge_times'] = [discharge_times[i] for i in sorted_indices]

            sorted_patients[patient_id] = single_patient

        if save_dataset:
            save_with_pickle(sorted_patients, processed_data_path, f'{dataset_name}_sorted.pickle')

    return sorted_patients


def expand_dataset(organized_patients: dict, processed_data_path: str, dataset_name: str, split_key: str, save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_LLM_{split_key}_expanded.pickle')):
        expanded_dataset = read_pickle_file(processed_data_path, f'{dataset_name}_LLM_{split_key}_expanded.pickle')

    else:
        expanded_dataset = {}

        for patient_id, patient in organized_patients.items():
            if len(patient['visit_ids']) > 2:
                samples = []

                for idx in range(2, len(patient['visit_ids']) + 1):
                    single_sample = {}

                    for data_key, data in patient.items():
                        if isinstance(data, list) or (isinstance(data, torch.Tensor) and data.ndim == 2):
                            single_sample[data_key] = data[:idx]

                        else:
                            single_sample[data_key] = data

                    patient_node_ids = torch.tensor(
                        [id for id_set in single_sample['visit_node_ids'] for id in id_set], dtype=int)
                    patient_node_ids = torch.unique(patient_node_ids)
                    updated_ehr_nodes = torch.zeros_like(single_sample['ehr_nodes'])
                    updated_ehr_nodes[patient_node_ids] = 1
                    single_sample['ehr_nodes'] = updated_ehr_nodes

                    samples.append(single_sample)

                for idx, sample in enumerate(samples):
                    expand_patient_id = patient_id + f'_{idx}'
                    expanded_dataset[expand_patient_id] = sample

            else:
                expanded_dataset[patient_id] = patient

        if save_dataset:
            save_with_pickle(expanded_dataset, processed_data_path, f'{dataset_name}_LLM_{split_key}_expanded.pickle')

    return expanded_dataset



def get_triplets(args,
                 node_name_to_id: dict,
                 processed_data_path: str,
                 save_triplets: bool = False):

    if osp.exists(
            osp.join(processed_data_path, f'LLM_triplet_id_to_info_ratio{args.triplet_ratio}.pickle')) and osp.exists(
                                      osp.join(processed_data_path, f'LLM_node_attr.pickle')) and osp.exists(osp.join(processed_data_path, f'LLM_edge_type_ratio{args.triplet_ratio}.pickle')):
        triplet_id_to_info = read_pickle_file(processed_data_path, f'LLM_triplet_id_to_info_ratio{args.triplet_ratio}.pickle')
        node_attr = read_pickle_file(processed_data_path, f'LLM_node_attr.pickle')
        edge_types = read_pickle_file(processed_data_path, f'LLM_edge_type_ratio{args.triplet_ratio}.pickle')

    else:
        llm_response = read_pickle_file(processed_data_path, f'LLM_response.pickle')

        # Reformat
        for idx, triplet in enumerate(llm_response):
            formatted_triplet = triplet

            for i, item in enumerate(triplet):
                formatted_item = convert_to_uppercase(item)
                formatted_item = formatted_item.replace('"', '')
                formatted_triplet[i] = formatted_item

            llm_response[idx] = formatted_triplet

        # Remove Duplicate Triplets
        llm_response = np.array(llm_response)
        llm_response = np.unique(llm_response, axis=0)

        # Remove Impossible Nodes
        src_dst = llm_response[:, [0, -1]]
        stored_nodes = list(node_name_to_id.keys())
        mask = np.ones(len(llm_response), dtype=bool)
        indices_to_remove = []

        for idx, pair in enumerate(src_dst):
            for node in pair:
                if node not in stored_nodes:
                    indices_to_remove.append(idx)

        mask[indices_to_remove] = False
        llm_response = llm_response[mask]

        # Remove N/A edges
        keys = ['N/A', 'IS NOT RELATED TO', 'NOT RELATED TO']
        for key in keys:
            mask = np.where(llm_response[:, 1] != key)[0]
            llm_response = llm_response[mask]

        # sample triplets for classifier
        selection = np.random.choice(llm_response.shape[0], 30000, replace=False)
        inv_selection = np.setdiff1d(np.arange(len(llm_response)), selection)
        classifier_llm_triplets = llm_response[selection]
        remain_llm_triplets = llm_response[inv_selection]

        if osp.exists(osp.join(
                processed_data_path, f'LLM_positive_triplets_30000.pickle')) and osp.exists(
                    osp.join(processed_data_path, f'LLM_negative_triplets_30000.pickle')):
            positive_triplets = read_pickle_file(processed_data_path,
                                                    f'LLM_positive_triplets_30000.pickle')
            negative_triplets = read_pickle_file(processed_data_path,
                                                    f'LLM_negative_triplets_30000.pickle')

        else:
            positive_triplets, negative_triplets = verify_embeddings(classifier_llm_triplets)

        if args.trained_classifier_path is not None:
            positive_input = np.char.add(positive_triplets[:, 0], ' ')
            positive_input = np.char.add(positive_input, ' ')
            positive_input = np.char.add(positive_input, positive_triplets[:, 1])
            positive_input = np.char.add(positive_input, ' ')
            positive_input = np.char.add(positive_input, positive_triplets[:, 2])

            negative_input = np.char.add(negative_triplets[:, 0], ' ')
            negative_input = np.char.add(negative_input, ' ')
            negative_input = np.char.add(negative_input, negative_triplets[:, 1])
            negative_input = np.char.add(negative_input, ' ')
            negative_input = np.char.add(negative_input, negative_triplets[:, 2])

            full_input = np.concatenate((positive_input, negative_input), axis=0)
            input_ids, attention_mask = encode_for_BERT(full_input)

            labels = torch.cat((torch.ones(len(positive_input)), torch.zeros(len(negative_input))), dim=0)

            experiment_name = 'Experiments_Triplet_Classifier'
            mlflow.set_tracking_uri(osp.join(args.log_path, 'mlflow'))
            client = MlflowClient()
            try:
                EXP_ID = client.create_experiment(experiment_name)
            except:
                experiments = client.get_experiment_by_name(experiment_name)
                EXP_ID = experiments.experiment_id

            with mlflow.start_run(experiment_id=EXP_ID,
                                    run_name=f'LLM_triplet_classifier_{args.start_time}'):
                mlflow.log_params(vars(args))
                trained_classifier_path = classifier_train_loop(args, (input_ids, attention_mask), labels)
                logger.info(f"Load pretrained model from {trained_classifier_path}")
                ckpt = torch.load(trained_classifier_path, map_location=device)

        else:
            logger.info(f"Load pretrained model from {args.trained_classifier_path}")
            ckpt = torch.load(args.trained_classifier_path, map_location=device)

        if osp.exists(osp.join(processed_data_path, 'classifier_output.pickle')):
            classifier_output = read_pickle_file(processed_data_path, 'classifier_output.pickle')
        else:
            remain_input = np.char.add(remain_llm_triplets[:, 0], ' ')
            remain_input = np.char.add(remain_input, ' ')
            remain_input = np.char.add(remain_input, remain_llm_triplets[:, 1])
            remain_input = np.char.add(remain_input, ' ')
            remain_input = np.char.add(remain_input, remain_llm_triplets[:, 2])

            input_ids, attention_mask = encode_for_BERT(remain_input)

            dataset = TripletDataset(data=(input_ids, attention_mask))
            dataloader = DataLoader(dataset=dataset, batch_size=args.triplet_method['classifier_train_batch_size'], shuffle=False, drop_last=False)

            model = Triplet_Classifier(args.model['args'])
            model.to(device)
            model.load_state_dict(ckpt['model'], strict=False)

            model.eval()
            mask = []
            full_prob = []

            for input_ids, attention_mask in tqdm(dataloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    out = model(input_ids=input_ids,
                        attention_mask=attention_mask)

                    probability = F.sigmoid(out)
                    pred = probability >= 0.5

                    full_prob.append(probability.cpu())
                    mask.append(pred.cpu())

            mask = torch.cat(mask, dim=0).flatten().numpy()
            full_prob = torch.cat(full_prob, dim=0).flatten().numpy()

            classifier_output = {'mask': mask, 'full_prob': full_prob}
            save_with_pickle(classifier_output, processed_data_path, 'classifier_output.pickle')
        
        mask = classifier_output['mask']
        full_prob = classifier_output['full_prob']

        remain_positive_triplets = remain_llm_triplets[mask]
        remain_positive_prob = full_prob[mask]

        llm_triplets = np.concatenate((positive_triplets, remain_positive_triplets), axis=0)    # join GPT verified and classifier prediction triplets
        edge_types = llm_triplets[:, 1]
        unique_edge_types, inv_map_1 = np.unique(edge_types, return_inverse=True)

        merge_map = merge_edge_type(unique_edge_types)

        merged_edge_types, inv_map_2 = np.unique(list(merge_map.values()), return_inverse=True)

        if osp.exists(osp.join(processed_data_path, f'LLM_edge_type_merge_map.pickle')):
            llm_merge_map = read_pickle_file(processed_data_path, f'LLM_edge_type_merge_map.pickle')
        else:
            edge_type_embeds = get_BERT_embeddings(merged_edge_types)
            clustering = AgglomerativeClustering(metric='euclidean',
                                                    linkage='ward',
                                                    distance_threshold=8,
                                                    n_clusters=None)
            clustering.fit(edge_type_embeds)
            assignments = clustering.labels_
            llm_merge_map = verify_clusters(merged_edge_types, assignments)
            save_with_pickle(llm_merge_map, processed_data_path, 'LLM_edge_type_merge_map.pickle')

        # manual fix
        llm_merge_map['may be prescribed for symptom associate with condition diagnose by'] = llm_merge_map.pop(
            'may be prescribed for symptom associate with condition diagnose ')

        src_edge_types = np.array(list(llm_merge_map.keys()))
        dst_edge_types = np.array(list(llm_merge_map.values()))

        for idx, edge in enumerate(dst_edge_types):
            dst_edge_types[idx] = edge.upper()

        permutation_2 = [np.where(src_edge_types == edge)[0][0] for edge in merged_edge_types]

        sorted_dst_edge_types = dst_edge_types[permutation_2]
        expanded_dst_edge_types = sorted_dst_edge_types[inv_map_2]

        prelim_merge_keys = np.array(list(merge_map.keys()))
        permutation_1 = [np.where(prelim_merge_keys == edge.lower())[0][0] for edge in unique_edge_types]

        sorted_expanded_dst_edge_types = expanded_dst_edge_types[permutation_1]
        clustered_edge_types = sorted_expanded_dst_edge_types[inv_map_1]

        for idx, edge_type in enumerate(clustered_edge_types):
            clustered_edge_types[idx] = edge_type.upper()

        llm_triplets[:, 1] = clustered_edge_types

        ratio = args.triplet_ratio
        full_triplet_prob = np.concatenate((np.ones(len(positive_triplets), dtype=float), remain_positive_prob), axis=0)
        sort_prob_mask = np.argsort(full_triplet_prob)[::-1]
        llm_triplets = llm_triplets[sort_prob_mask]

        if ratio == 1.0:
            base_triplets = llm_triplets[:len(positive_triplets)]
            print(f'triplet number: {len(positive_triplets)}')
        else:
            base_triplet_num = int(len(positive_triplets) * ratio)
            assert base_triplet_num <= len(llm_triplets), 'ratio too large, not enough triplets'
            base_triplets = llm_triplets[:base_triplet_num]
            print(f'triplet number: {base_triplet_num}')

        node_names = list(node_name_to_id.keys())
        node_names = np.array(node_names)
        edge_names = base_triplets[:, 1]
        unique_edges, edge_types = np.unique(edge_names, return_inverse=True)

        # Get Embedding from BERT
        llm_node_embeddings = get_BERT_embeddings(node_names)

        triplet_id_to_info = {}
        for idx in range(len(base_triplets)):
            triplet_id_to_info[idx + 1] = base_triplets[idx].tolist()

        node_attr = llm_node_embeddings.numpy()
        edge_types += 1

        if save_triplets:
            save_with_pickle(triplet_id_to_info, processed_data_path,
                             f'LLM_triplet_id_to_info_ratio{args.triplet_ratio}.pickle')
            save_with_pickle(node_attr, processed_data_path, f'LLM_node_attr.pickle')
            save_with_pickle(edge_types, processed_data_path, f'LLM_edge_type_ratio{args.triplet_ratio}.pickle')

    return triplet_id_to_info, node_attr, edge_types


def classifier_single_train(model,
                            dataloader,
                            epoch_idx,
                            global_iter_idx,
                            optimizer,
                            criterions=[],
                            metrics=[],
                            scheduler=None,
                            logging_freq=10):

    train_start_time = time.time()
    model.train()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.unsqueeze(-1).to(device)

        optimizer.zero_grad()

        out = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = 0.
        for criterion in criterions:
            loss += criterion(out, labels)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        probability = F.sigmoid(out)

        prob_all.append(probability.cpu().detach())
        target_all.append(labels.cpu().detach())

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(dataloader):4d} [{global_iter_idx[0]:5d}], Loss: {loss.item()}"
            )

        mlflow.log_metric(key='train_batch_loss', value=loss.item(), step=global_iter_idx[0])

        global_iter_idx[0] += 1

    if scheduler is not None:
        scheduler.step()

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metrics({
        'train_time': time.time() - train_start_time,
        'train_epoch_loss': epoch_loss_avg
    },
                       step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        mlflow.log_metric(key=f'train_{metric.NAME}', value=score, step=epoch_idx)


def classifier_single_validate(model, dataloader, epoch_idx, global_iter_idx, criterions=[], metrics=[]):

    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for _, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.unsqueeze(-1).to(device)

        with torch.no_grad():

            out = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = 0.
            for criterion in criterions:
                loss += criterion(out, labels)

            epoch_loss.append(loss.item())

            probability = F.sigmoid(out)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metric(key='val_epoch_loss', value=epoch_loss_avg, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    results = {}

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        mlflow.log_metric(key=f'val_{metric.NAME}', value=score, step=epoch_idx)
        results[metric.NAME] = score

    return results


def classifier_single_test(model, dataloader, epoch_idx, global_iter_idx, criterions=[], metrics=[]):

    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for _, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.unsqueeze(-1).to(device)

        with torch.no_grad():

            out = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = 0.
            for criterion in criterions:
                loss += criterion(out, labels)

            epoch_loss.append(loss.item())

            probability = F.sigmoid(out)

            prob_all.append(probability.cpu())
            target_all.append(labels.cpu())

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metric(key='test_epoch_loss', value=epoch_loss_avg, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    results = {}

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        mlflow.log_metric(key=f'test_{metric.NAME}', value=score, step=epoch_idx)
        results[metric.NAME] = score

    return results


def classifier_train_loop(args, data, labels):
    dataset = TripletDataset(data=data, labels=labels)

    ratio = [args.triplet_method['classifier_train_proportion'], args.triplet_method['classifier_val_proportion'], args.triplet_method['classifier_test_proportion']]
    train_dataset, val_dataset, test_dataset = random_split(dataset, ratio)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.triplet_method['classifier_train_batch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.triplet_method['classifier_val_batch_size'], shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.triplet_method['classifier_test_batch_size'], shuffle=False)

    model = Triplet_Classifier(args.model['args'])
    model.to(device)

    optimizer = OPTIMIZERS[args.optimizer['name']](filter(lambda p: p.requires_grad, model.parameters()),
                                                   **args.optimizer['args'])
    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])

    global_iter_idx = [0]
    early_stopping_counter = 0
    best_score = 0.
    path = None

    for epoch_idx in range(args.max_epoch):
        # Train
        classifier_single_train(
            model,
            train_dataloader,
            epoch_idx,
            global_iter_idx,
            optimizer,
            criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
            metrics=[METRICS[metric](args.task) for metric in args.val_metrics],
            scheduler=scheduler,
            logging_freq=args.logging_freq)

        # Validate
        if epoch_idx % args.val_freq == 0 or epoch_idx == args.max_epoch - 1:
            results = classifier_single_validate(
                model,
                val_dataloader,
                epoch_idx,
                global_iter_idx,
                criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                metrics=[METRICS[metric](args.task) for metric in args.val_metrics])

            test_results = classifier_single_test(
                model,
                test_dataloader,
                epoch_idx,
                global_iter_idx,
                criterions=[CRITERIONS[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                metrics=[METRICS[metric](args.task) for metric in args.test_metrics])

        score = results[args.triplet_method['classifier_early_stopping_indicator']]

        if score >= best_score:
            best_model = deepcopy(model)
            best_optimizer = deepcopy(optimizer)
            best_scheduler = deepcopy(scheduler)
            best_score = score
            best_results = results
            best_test_results = test_results
            best_epoch = epoch_idx
            best_iter = global_iter_idx[0]
            early_stopping_counter = 0

        else:
            early_stopping_counter += 1

        if early_stopping_counter >= args.triplet_method['classifier_early_stopping_threshold']:
            logger.info(f'Early stopping triggered, best epoch: {best_epoch}')
            for k, v in best_results.items():
                if isinstance(v, list):
                    for element in v:
                        logger.info(f'Best {k}: {element:.4f}')
                else:
                    logger.info(f'Best {k}: {v:.4f}')

            for k, v in best_test_results.items():
                if isinstance(v, list):
                    for element in v:
                        logger.info(f'Best test {k}: {element:.4f}')
                else:
                    logger.info(f'Best test {k}: {v:.4f}')

            mlflow.log_params({'best_results': best_results})
            mlflow.log_params({'best_test_results': best_test_results})

            if args.save_params:
                path = osp.join(args.log_path, 'checkpoints', args.dataset, 'triplet_classifier',
                                f'LLM_triplet_classifier_{args.start_time}.pth')
                save_params(model=best_model,
                            args=args,
                            epoch_idx=best_epoch,
                            iter_idx=best_iter,
                            optimizer=best_optimizer,
                            scheduler=best_scheduler,
                            path=path)

            logger.info('Process completed')
            return path

    logger.info(f'Max epoch reached')
    for k, v in best_results.items():
        if isinstance(v, list):
            for element in v:
                logger.info(f'Best {k}: {element:.4f}')
        else:
            logger.info(f'Best {k}: {v:.4f}')

    for k, v in best_test_results.items():
        if isinstance(v, list):
            for element in v:
                logger.info(f'Best test {k}: {element:.4f}')
        else:
            logger.info(f'Best test {k}: {v:.4f}')

    mlflow.log_params({'best_results': best_results})
    mlflow.log_params({'best_test_results': best_test_results})

    if args.save_params:
        path = osp.join(args.log_path, 'checkpoints', args.dataset, 'triplet_classifier',
                        f'LLM_triplet_classifier_{args.start_time}.pth')
        save_params(model=best_model,
                    args=args,
                    epoch_idx=best_epoch,
                    iter_idx=best_iter,
                    optimizer=best_optimizer,
                    scheduler=best_scheduler,
                    path=path)

    return path


def construct_graph(node_name_to_id: dict,
                    triplet_id_to_info: dict,
                    node_attr: np.ndarray,
                    edge_type: np.ndarray,
                    processed_data_path: str,
                    triplet_ratio: float,
                    PAD_ID: int = 0,
                    save_graph: bool = False,
                    save_triplet_maps: bool = False):

    if osp.exists(osp.join(
            processed_data_path, f'LLM_triplet_id_to_edge_index_ratio{triplet_ratio}')) and osp.exists(
                osp.join(processed_data_path, f'LLM_triplet_edge_index_to_id_ratio{triplet_ratio}')):
        triplet_id_to_edge_index = read_pickle_file(processed_data_path,
                                                    f'LLM_triplet_id_to_edge_index_ratio{triplet_ratio}')
        triplet_edge_index_to_id = read_pickle_file(processed_data_path,
                                                    f'LLM_triplet_edge_index_to_id_ratio{triplet_ratio}')

    else:
        triplet_info = np.array(list(triplet_id_to_info.values()))
        src_dst = triplet_info[:, [0, -1]]

        edge_src = []
        edge_dst = []
        for src, dst in src_dst:
            edge_src.append(node_name_to_id[src])
            edge_dst.append(node_name_to_id[dst])

        edge_index = [edge_src, edge_dst]
        edge_index = torch.tensor(edge_index)
        rev_edge_index = edge_index.flip(dims=[0])
        edge_index = torch.cat((edge_index, rev_edge_index), dim=-1)

        edge_type = torch.tensor(edge_type, dtype=torch.int64)
        rev_edge_type = edge_type + edge_type.max()
        edge_type = torch.cat((edge_type, rev_edge_type), dim=0)

        triplet_id_to_edge_index = {}
        triplet_edge_index_to_id = {}

        for idx, pair in enumerate(edge_index.t()):
            triplet_id_to_edge_index[idx + 1] = pair.tolist()
            triplet_edge_index_to_id[str(pair.tolist())] = idx + 1

        if save_triplet_maps:
            save_with_pickle(triplet_id_to_edge_index, processed_data_path,
                             f'LLM_triplet_id_to_edge_index_ratio{triplet_ratio}.pickle')
            save_with_pickle(triplet_edge_index_to_id, processed_data_path,
                             f'LLM_triplet_edge_index_to_id_ratio{triplet_ratio}.pickle')

    if osp.exists(osp.join(processed_data_path, f'LLM_graph_ratio{triplet_ratio}.pickle')):
        GRAPH = read_pickle_file(processed_data_path, f'LLM_graph_ratio{triplet_ratio}.pickle')

    else:
        node_attr = torch.from_numpy(node_attr)

        edge_index = list(triplet_id_to_edge_index.values())
        edge_index = torch.tensor(edge_index, dtype=int).t()

        edge_pairs = edge_index.t()
        edge_ids = []
        for pair in edge_pairs:
            edge_ids.append(triplet_edge_index_to_id[str(pair.tolist())])
        edge_ids = torch.tensor(edge_ids, dtype=int)

        pad_node_attr = torch.randn((1, node_attr.size(-1)))
        node_attr = torch.cat((pad_node_attr, node_attr), dim=0)
        pad_edge_index = torch.tensor([[PAD_ID], [PAD_ID]], dtype=int)
        edge_index = torch.cat((pad_edge_index, edge_index), dim=1)
        pad_edge_ids = torch.tensor([0], dtype=int)
        edge_ids = torch.cat((pad_edge_ids, edge_ids), dim=0)
        pad_edge_type = torch.tensor([0], dtype=int)
        edge_type = torch.cat((pad_edge_type, edge_type), dim=0)

        device = torch.device('cpu')
        graph = dgl.DGLGraph()
        graph.add_nodes(len(node_attr))
        graph.add_edges(edge_index[0], edge_index[1])
        graph = graph.to(device)

        in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        norm = in_deg**-0.5
        norm[np.isinf(norm)] = 0
        graph.ndata['xxx'] = torch.from_numpy(norm).to(device)
        graph.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        edge_norm = graph.edata.pop('xxx').squeeze()

        GRAPH = {
            'graph': graph,
            'edge_index': edge_index,
            'edge_ids': edge_ids,
            'h': node_attr,
            'type': edge_type,
            'norm': edge_norm
        }

        if save_graph:
            save_with_pickle(GRAPH, processed_data_path, f'LLM_graph_ratio{triplet_ratio}.pickle')

    return GRAPH, triplet_id_to_edge_index, triplet_edge_index_to_id


def construct_patient(patient,
                      patient_data: dict,
                      graph: dict,
                      visit_id_to_nodes: dict,
                      node_name_to_id: dict,
                      triplet_edge_index_to_id: dict):

    patient_data['patient_id'] = patient['patient_id']
    patient_data['mortality'] = patient['mortality']
    patient_data['mortality_visit_discharge_time'] = patient['mortality_visit_discharge_time']
                          
    subset_threshold = torch.max(graph['edge_index'])

    visit_encounters = patient['encounter_times']
    visit_discharges = patient['discharge_times']
    visit_rel_times = []
    hist_time = 0

    for enc, dis in zip(visit_encounters, visit_discharges):
        if hist_time == 0:
            visit_rel_times.append(0)

        else:
            time_dif = (enc - hist_time).total_seconds()
            day_dif = (enc - hist_time).days
            week_dif = day_dif // 7
            if time_dif < 0:
                return
            visit_rel_times.append(week_dif + 1)

        hist_time = dis
    patient_data['visit_rel_times'] = visit_rel_times

    # Construct Patient
    for visit in patient['visits']:
        visit_id = visit.visit_id
        codes = visit_id_to_nodes[visit_id]
        patient_data['visit_ids'].append(visit_id)

        single_visit_node_ids = []
        single_visit_edge_ids = []
        single_visit_mask = []

        for code in codes:
            single_visit_node_ids.append(node_name_to_id[code])

        subset = torch.tensor(single_visit_node_ids, dtype=int)
        subset = subset[subset <= subset_threshold]

        single_visit_edge_index, _ = subgraph(subset=subset,
                                              edge_index=graph['edge_index'],
                                              relabel_nodes=False)

        if len(single_visit_edge_index.flatten().tolist()) == 0:
            single_visit_edge_index = None
            single_visit_edge_ids = None

        single_visit_mask = [False] * len(single_visit_node_ids)

        patient_data['visit_node_ids'].append(single_visit_node_ids)
        patient_data['node_padding_mask'].append(single_visit_mask)
        patient_data['visit_edge_index'].append(single_visit_edge_index)

        if single_visit_edge_ids is not None:
            edge_pairs = single_visit_edge_index.t()
            for pair in edge_pairs:
                id = triplet_edge_index_to_id[str(pair.tolist())]
                single_visit_edge_ids.append(id)

        patient_data['visit_edge_ids'].append(single_visit_edge_ids)

    visit_node_ids = patient_data['visit_node_ids']
    visit_nodes = torch.zeros(len(visit_node_ids), len(node_name_to_id) + 1)
    for i in range(len(visit_node_ids)):
        visit_nodes[i, visit_node_ids[i]] = 1
    patient_data['visit_nodes'] = visit_nodes

    patient_node_ids = torch.tensor([id for id_set in visit_node_ids for id in id_set], dtype=int)
    patient_node_ids = torch.unique(patient_node_ids)
    ehr_nodes = torch.zeros(len(node_name_to_id) + 1)
    ehr_nodes[patient_node_ids] = 1
    patient_data['ehr_nodes'] = ehr_nodes

    return patient_data


def organize_dataset(patients,
                     graph: dict,
                     visit_id_to_nodes: list,
                     node_name_to_id: dict,
                     triplet_edge_index_to_id: dict,
                     processed_data_path: str,
                     dataset_name: str,
                     save_dataset: bool = False):

    path = osp.join(processed_data_path, f'{dataset_name}_LLM_organized.pickle')
    name = f'{dataset_name}_LLM_organized.pickle'

    if osp.exists(path):
        dataset = read_pickle_file(processed_data_path, name)

    else:
        dataset = {}

        PATIENT_TEMPLATE = {
            'patient_id': None,
            'visit_ids': [],
            'visit_node_ids': [],
            'visit_edge_ids': [],
            'visit_edge_index': [],
            'visit_rel_times': [],
            'visit_nodes': None,
            'ehr_nodes': None,
            'node_padding_mask': [],
            'last_visit_drug_names': None,
            'last_visit_diagnosis_names': None,
            'mortality': False,
            'mortality_visit_discharge_time': None,
            'labels': {}
        }

        for id, patient in tqdm(patients.items(), desc='Organizing patients'):
            patient_data = deepcopy(PATIENT_TEMPLATE)

            patient_data = construct_patient(patient=patient,
                                             patient_data=patient_data,
                                             graph=graph,
                                             visit_id_to_nodes=visit_id_to_nodes,
                                             node_name_to_id=node_name_to_id,
                                             triplet_edge_index_to_id=triplet_edge_index_to_id)

            if patient_data is not None:
                dataset[id] = patient_data

        if save_dataset:
            save_with_pickle(dataset, processed_data_path, name)

    return dataset


def split_dataset(patients: dict,
                  split_ratio: list,
                  processed_data_path: str,
                  dataset_name: str,
                  save_dataset: bool = False):

    if osp.exists(osp.join(
            processed_data_path, f'{dataset_name}_LLM_train.pickle')) and osp.exists(
                osp.join(processed_data_path, f'{dataset_name}_LLM_val.pickle')) and osp.exists(
                    osp.join(processed_data_path, f'{dataset_name}_LLM_test.pickle')):
        train_patients = read_pickle_file(processed_data_path,
                                          f'{dataset_name}_LLM_train.pickle')
        val_patients = read_pickle_file(processed_data_path, f'{dataset_name}_LLM_val.pickle')
        test_patients = read_pickle_file(processed_data_path, f'{dataset_name}_LLM_test.pickle')

    else:
        patient_num = len(patients)
        patients_k = np.array(list(patients.keys()))

        np.random.shuffle(patients_k)

        train_num = int(split_ratio[0] * patient_num)
        val_num = int(split_ratio[1] * patient_num)

        train_k = patients_k[:train_num]
        val_k = patients_k[train_num:train_num + val_num]
        test_k = patients_k[train_num + val_num:]

        train_patients = {key: patients[key] for key in train_k}
        val_patients = {key: patients[key] for key in val_k}
        test_patients = {key: patients[key] for key in test_k}

        print(f'# of train patients: {len(train_patients)}; # of val patients: {len(val_patients)}; # of test patients: {len(test_patients)}')

        if save_dataset:
            save_with_pickle(train_patients, processed_data_path,
                             f'{dataset_name}_LLM_train.pickle')
            save_with_pickle(val_patients, processed_data_path, f'{dataset_name}_LLM_val.pickle')
            save_with_pickle(test_patients, processed_data_path,
                             f'{dataset_name}_LLM_test.pickle')

    return {'train': train_patients, 'val': val_patients, 'test': test_patients}


def configure_for_task(dataset: dict,
                       filtered_patients: dict,
                       sorted_patients: dict,
                       task: str,
                       node_id_to_name: dict,
                       processed_data_path: str,
                       dataset_name: str,
                       triplet_ratio: float,
                       split: str,
                       suffix: str,
                       save_dataset: bool = False,
                       prescriptions_name_to_code: dict = None,
                       diagnoses_name_to_code: dict = None,
                       phenotype_names: list = None):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_LLM_configured_{split}_ratio{triplet_ratio}_{suffix}.pickle')):
        configured_dataset = read_pickle_file(processed_data_path,
                                              f'{dataset_name}_LLM_configured_{split}_ratio{triplet_ratio}_{suffix}.pickle')

    else:
        configured_dataset = {}

        if task == 'mortality_prediction' or task == 'readmission_prediction':

            for patient_id, patient in tqdm(dataset.items(), desc='Configuring dataset'):
                configured_patient = deepcopy(patient)
                patient_id = patient['patient_id']
                filtered_patient = filtered_patients[patient_id]
                sorted_patient = sorted_patients[patient_id]

                if len(patient['visit_ids']) == len(filtered_patient):
                    # Remove Last Visit
                    configured_patient['visit_ids'] = patient['visit_ids'][:-1]
                    configured_patient['visit_node_ids'] = patient['visit_node_ids'][:-1]
                    configured_patient['node_padding_mask'] = patient['node_padding_mask'][:-1]
                    configured_patient['visit_edge_index'] = patient['visit_edge_index'][:-1]
                    configured_patient['visit_edge_ids'] = patient['visit_edge_ids'][:-1]
                    configured_patient['visit_nodes'] = patient['visit_nodes'][:-1]
                    configured_patient['visit_rel_times'] = patient['visit_rel_times'][:-1]

                    patient_node_ids = torch.tensor(
                        [id for id_set in configured_patient['visit_node_ids'] for id in id_set], dtype=int)
                    patient_node_ids = torch.unique(patient_node_ids)
                    updated_ehr_nodes = torch.zeros_like(configured_patient['ehr_nodes'])
                    updated_ehr_nodes[patient_node_ids] = 1
                    configured_patient['ehr_nodes'] = updated_ehr_nodes

                this_visit = sorted_patient['visits'][len(configured_patient['visit_ids']) - 1]
                next_visit = sorted_patient['visits'][len(configured_patient['visit_ids'])]

                if patient['mortality'] == 1:
                    days_dif = (patient['mortality_visit_discharge_time'] - this_visit.discharge_time).days

                    if days_dif <= 30:
                        configured_patient['labels']['mortality_prediction'] = torch.tensor([[1.]])
                    else:
                        configured_patient['labels']['mortality_prediction'] = torch.tensor([[0.]])

                else:
                    configured_patient['labels']['mortality_prediction'] = torch.tensor([[0.]])

                time_diff = (next_visit.encounter_time - this_visit.discharge_time).days
                assert time_diff >= 0
                readmission_label = 1. if time_diff <= 30 else 0.
                configured_patient['labels']['readmission_prediction'] = torch.tensor([[readmission_label]])

                configured_dataset[patient_id] = configured_patient

        elif task == 'los_prediction':

            for patient_id, patient in tqdm(dataset.items(), desc='Configuring dataset'):
                configured_patient = deepcopy(patient)
                patient_id = patient['patient_id']
                filtered_patient = filtered_patients[patient_id]

                this_visit = filtered_patient.get_visit_by_id(configured_patient['visit_ids'][-1])
                los_days = (this_visit.discharge_time - this_visit.encounter_time).days

                if los_days < 1:
                    configured_patient['labels']['los_prediction'] = torch.tensor([[0]])
                elif 1 <= los_days <= 7:
                    configured_patient['labels']['los_prediction'] = torch.tensor([[los_days]])
                elif 7 < los_days <= 14:
                    configured_patient['labels']['los_prediction'] = torch.tensor([[8]])
                else:
                    configured_patient['labels']['los_prediction'] = torch.tensor([[9]])

                configured_dataset[patient_id] = configured_patient

        elif task == 'drug_recommendation':
            assert prescriptions_name_to_code is not None, f'Require prescriptions_code_to_name for drug_recommendation'

            for patient_id, patient in tqdm(dataset.items(), desc='Configuring dataset'):
                configured_patient = deepcopy(patient)
                patient_id = patient['patient_id']

                updated_visit_node_ids = configured_patient['visit_node_ids']
                last_visit_nodes = updated_visit_node_ids[-1]
                updated_last_visit_nodes = []
                last_visit_drug_names = []

                for id in last_visit_nodes:
                    node_name = node_id_to_name[id]

                    # Check if Code is Drug
                    try:
                        _ = prescriptions_name_to_code[node_name]
                        last_visit_drug_names.append(node_name)
                    except:
                        updated_last_visit_nodes.append(id)

                updated_visit_node_ids[-1] = updated_last_visit_nodes
                updated_last_node_padding_mask = [False] * len(updated_last_visit_nodes)

                if configured_patient['visit_edge_index'][-1] is not None:
                    edge_mask = torch.isin(configured_patient['visit_edge_index'][-1],
                                           torch.tensor(updated_last_visit_nodes))
                    edge_mask = edge_mask.all(dim=0)
                    updated_last_edge_index = configured_patient['visit_edge_index'][-1][:, edge_mask]
                    updated_last_edge_ids = torch.tensor(configured_patient['visit_edge_ids'][-1])[edge_mask]
                    updated_last_edge_ids = updated_last_edge_ids.tolist()

                    if updated_last_edge_index.size(-1) == 0:
                        updated_last_edge_index = None
                        updated_last_edge_ids = None
                else:
                    updated_last_edge_index = None
                    updated_last_edge_ids = None

                updated_final_visit_nodes = configured_patient['visit_nodes']
                updated_final_visit_nodes = torch.zeros_like(updated_final_visit_nodes[-1])
                updated_final_visit_nodes[updated_last_visit_nodes] = 1

                patient_node_ids = torch.tensor([id for id_set in updated_visit_node_ids for id in id_set], dtype=int)
                patient_node_ids = torch.unique(patient_node_ids)
                updated_ehr_nodes = torch.zeros_like(configured_patient['ehr_nodes'])
                updated_ehr_nodes[patient_node_ids] = 1

                configured_patient['visit_node_ids'] = updated_visit_node_ids
                configured_patient['node_padding_mask'][-1] = updated_last_node_padding_mask
                configured_patient['visit_edge_index'][-1] = updated_last_edge_index
                configured_patient['visit_edge_ids'][-1] = updated_last_edge_ids
                configured_patient['visit_nodes'][-1] = updated_final_visit_nodes
                configured_patient['ehr_nodes'] = updated_ehr_nodes
                configured_patient['last_visit_drug_names'] = last_visit_drug_names

                label = torch.zeros(len(prescriptions_name_to_code) + 1)
                full_drug_names = list(prescriptions_name_to_code.keys())

                if len(last_visit_drug_names) == 0:
                    label[0] = 1
                else:
                    for name in last_visit_drug_names:
                        label[full_drug_names.index(name) + 1] = 1

                configured_patient['labels']['drug_recommendation'] = label.unsqueeze(0)

                configured_dataset[patient_id] = configured_patient

        elif task == 'phenotype_prediction':
            assert diagnoses_name_to_code is not None, f'Require diagnoses_name_to_code for phenotype_prediction'
            assert phenotype_names is not None, f'Require phenotype_names for drug_recommendation'

            for patient_id, patient in tqdm(dataset.items(), desc='Configuring dataset'):
                configured_patient = deepcopy(patient)
                patient_id = patient['patient_id']

                updated_visit_node_ids = configured_patient['visit_node_ids']
                last_visit_nodes = updated_visit_node_ids[-1]
                updated_last_visit_nodes = []
                last_visit_diagnosis_names = []

                for id in last_visit_nodes:
                    node_name = node_id_to_name[id]

                    # Check if Node is Diagnosis
                    try:
                        _ = diagnoses_name_to_code[node_name]
                        last_visit_diagnosis_names.append(node_name)
                    except:
                        updated_last_visit_nodes.append(id)

                updated_visit_node_ids[-1] = updated_last_visit_nodes
                updated_last_node_padding_mask = [False] * len(updated_last_visit_nodes)

                if configured_patient['visit_edge_index'][-1] is not None:
                    edge_mask = torch.isin(configured_patient['visit_edge_index'][-1],
                                           torch.tensor(updated_last_visit_nodes))
                    edge_mask = edge_mask.all(dim=0)
                    updated_last_edge_index = configured_patient['visit_edge_index'][-1][:, edge_mask]
                    updated_last_edge_ids = torch.tensor(configured_patient['visit_edge_ids'][-1])[edge_mask]
                    updated_last_edge_ids = updated_last_edge_ids.tolist()

                    if updated_last_edge_index.size(-1) == 0:
                        updated_last_edge_index = None
                        updated_last_edge_ids = None
                else:
                    updated_last_edge_index = None
                    updated_last_edge_ids = None

                updated_final_visit_nodes = configured_patient['visit_nodes']
                updated_final_visit_nodes = torch.zeros_like(updated_final_visit_nodes[-1])
                updated_final_visit_nodes[updated_last_visit_nodes] = 1

                patient_node_ids = torch.tensor([id for id_set in updated_visit_node_ids for id in id_set], dtype=int)
                patient_node_ids = torch.unique(patient_node_ids)
                updated_ehr_nodes = torch.zeros_like(configured_patient['ehr_nodes'])
                updated_ehr_nodes[patient_node_ids] = 1

                configured_patient['visit_node_ids'] = updated_visit_node_ids
                configured_patient['node_padding_mask'][-1] = updated_last_node_padding_mask
                configured_patient['visit_edge_index'][-1] = updated_last_edge_index
                configured_patient['visit_edge_ids'][-1] = updated_last_edge_ids
                configured_patient['visit_nodes'][-1] = updated_final_visit_nodes
                configured_patient['ehr_nodes'] = updated_ehr_nodes
                configured_patient['last_visit_diagnosis_names'] = last_visit_diagnosis_names

                label = torch.zeros(25 + 1)

                if len(last_visit_diagnosis_names) == 0:
                    label[0] = 1
                else:
                    for name in last_visit_diagnosis_names:
                        if name in phenotype_names:
                            label[phenotype_names.index(name) + 1] = 1
                    if label.sum() == 0:
                        label[0] = 1

                configured_patient['labels']['phenotype_prediction'] = label.unsqueeze(0)

                configured_dataset[patient_id] = configured_patient

        if save_dataset:
            save_with_pickle(configured_dataset, processed_data_path,
                             f'{dataset_name}_LLM_configured_{split}_ratio{triplet_ratio}_{suffix}.pickle')

    print(len(configured_dataset))

    return configured_dataset


def pad_dataset(dataset: dict,
                code_pad_dim: int,
                visit_pad_dim: int,
                processed_data_path: str,
                dataset_name: str,
                split: str,
                suffix: str,
                triplet_ratio: float,
                PAD_ID: int = 0,
                save_dataset: bool = False):

    if osp.exists(osp.join(processed_data_path, f'{dataset_name}_LLM_padded_{split}_ratio{triplet_ratio}_{suffix}.pickle')):
        dataset = read_pickle_file(processed_data_path,
                                   f'{dataset_name}_LLM_padded_{split}_ratio{triplet_ratio}_{suffix}.pickle')

    else:
        for patient in tqdm(dataset.values(), desc='Padding'):

            visit_node_ids = deepcopy(patient['visit_node_ids'])
            visit_edge_ids = deepcopy(patient['visit_edge_ids'])
            node_padding_mask = deepcopy(patient['node_padding_mask'])
            visit_edge_index = deepcopy(patient['visit_edge_index'])
            visit_rel_times = deepcopy(patient['visit_rel_times'])
            visit_nodes = deepcopy(patient['visit_nodes'])
            ehr_nodes = deepcopy(patient['ehr_nodes'])

            # code padding and mask padding
            for idx, single_visit_ids in enumerate(visit_node_ids):
                if len(single_visit_ids) < code_pad_dim:
                    pad_length = code_pad_dim - len(single_visit_ids)
                    visit_node_ids[idx] = single_visit_ids + [PAD_ID] * pad_length
                    node_padding_mask[idx] += [True] * pad_length

                elif len(single_visit_ids) > code_pad_dim:
                    raise ValueError('code_pad_dim must exceed max code length')

            if len(visit_node_ids) < visit_pad_dim:
                pad_length = visit_pad_dim - len(visit_node_ids)

                for _ in range(pad_length):
                    visit_node_ids.append([PAD_ID] * code_pad_dim)
                    node_padding_mask.append([True] * code_pad_dim)

                visit_nodes_pad = torch.zeros((pad_length, visit_nodes.size(-1)), dtype=int)
                visit_nodes = torch.cat((visit_nodes, visit_nodes_pad), dim=0)

            elif len(visit_node_ids) > visit_pad_dim:
                raise ValueError('visit_pad_dim must exceed max visit num')

            visit_node_ids = torch.tensor(visit_node_ids,
                                          dtype=int) if not isinstance(visit_node_ids, torch.Tensor) else visit_node_ids

            for idx, single_visit_edge_index in enumerate(visit_edge_index):
                if single_visit_edge_index is None:
                    visit_edge_index[idx] = torch.tensor([[PAD_ID], [PAD_ID]], dtype=int)
                    visit_edge_ids[idx] = [PAD_ID]

                else:
                    if torch.eq(visit_node_ids[idx], PAD_ID).any():
                        visit_edge_index[idx] = torch.cat((torch.tensor([[PAD_ID], [PAD_ID]]), single_visit_edge_index),
                                                          dim=-1)
                        visit_edge_ids[idx].append(PAD_ID)

                visit_edge_ids[idx] = torch.tensor(visit_edge_ids[idx], dtype=int)

            visit_rel_times = torch.tensor(
                visit_rel_times, dtype=int) if not isinstance(visit_rel_times, torch.Tensor) else visit_rel_times
            pad = torch.zeros(visit_pad_dim - len(visit_rel_times))
            visit_rel_times = torch.cat((visit_rel_times, pad), dim=0).to(torch.int)

            visit_nodes = torch.tensor(visit_nodes,
                                       dtype=int) if not isinstance(visit_nodes, torch.Tensor) else visit_nodes
            ehr_nodes = torch.tensor(ehr_nodes, dtype=int) if not isinstance(ehr_nodes, torch.Tensor) else ehr_nodes
            node_padding_mask = torch.tensor(
                node_padding_mask, dtype=bool) if not isinstance(node_padding_mask, torch.Tensor) else node_padding_mask

            patient['visit_node_ids'] = visit_node_ids
            patient['visit_edge_ids'] = visit_edge_ids
            patient['visit_edge_index'] = visit_edge_index
            patient['visit_rel_times'] = visit_rel_times
            patient['visit_nodes'] = visit_nodes
            patient['ehr_nodes'] = ehr_nodes
            patient['node_padding_mask'] = node_padding_mask

        if save_dataset:
            save_with_pickle(dataset, processed_data_path,
                             f'{dataset_name}_LLM_padded_{split}_ratio{triplet_ratio}_{suffix}.pickle')

    return dataset


def run(args):
    seed_everything(args.seed)
    init_logger()

    if 'data_preparation' in args.task:
        task_name = args.task.split('-')[1]
        args.task = task_name

    raw_data_path = osp.join(args.raw_data_path, args.dataset)
    processed_data_path = osp.join(args.processed_data_path, args.dataset)

    file_lib = get_dependent_files(raw_data_path, processed_data_path, args.dataset)
    print('Got files')

    patient_info = format_from_csv(file_lib['patient_info'])
    diagnoses_code_to_name, diagnoses_name_to_code = format_code_map(file_lib['diagnoses_map'])
    procedures_code_to_name, procedures_name_to_code = format_code_map(file_lib['procedures_map'])
    prescriptions_code_to_name, prescriptions_name_to_code = format_code_map(file_lib['prescriptions_map'])

    dataset = get_dataset(raw_data_path=raw_data_path,
                          processed_data_path=processed_data_path,
                          dataset_name=args.dataset,
                          save_dataset=False)
    print('Got dataset')

    filtered_patients = filter_dataset(patients=dataset.patients,
                                       patient_info=patient_info,
                                       filter_args=args.dataset_filtering['args'],
                                       processed_data_path=processed_data_path,
                                       dataset_name=args.dataset,
                                       save_dataset=False)
    print('Filtered dataset')

    filtered_patients = read_pickle_file(processed_data_path, f'{args.dataset}_filtered.pickle')

    sorted_patients = sort_visits(filtered_patients,
                                  processed_data_path=processed_data_path,
                                  dataset_name=args.dataset,
                                  save_dataset=False)
    print('Sorted visits in dataset')

    sorted_patients = read_pickle_file(processed_data_path, f'{args.dataset}_sorted.pickle')

    triplet_id_to_info, node_attr, edge_type = get_triplets(args=args,
                                                                       node_name_to_id=file_lib['node_name_to_id'],
                                                                       processed_data_path=processed_data_path,
                                                                       save_triplets=False)
    print('Got triplets')

    graph, triplet_id_to_edge_index, triplet_edge_index_to_id = construct_graph(
        node_name_to_id=file_lib['node_name_to_id'],
        triplet_id_to_info=triplet_id_to_info,
        node_attr=node_attr,
        edge_type=edge_type,
        processed_data_path=processed_data_path,
        triplet_ratio=args.triplet_ratio,
        save_graph=False,
        save_triplet_maps=False)
    print('Graph constructed')

    organized_dataset = organize_dataset(patients=sorted_patients,
                                         graph=graph,
                                         visit_id_to_nodes=file_lib['visit_id_to_nodes'],
                                         node_name_to_id=file_lib['node_name_to_id'],
                                         triplet_edge_index_to_id=triplet_edge_index_to_id,
                                         processed_data_path=processed_data_path,
                                         dataset_name=args.dataset,
                                         save_dataset=True)
    print('Organized dataset')

    split_ratio = [args.train_proportion, args.val_proportion, args.test_proportion]
    split_patients = split_dataset(patients=organized_dataset,
                                   split_ratio=split_ratio,
                                   processed_data_path=processed_data_path,
                                   dataset_name=args.dataset,
                                   save_dataset=True)
    print('Got splitted dataset')
    
    if args.dataset == 'mimiciii' and args.task != 'mortality_prediction':
        for key, sub_dataset in split_patients.items():
            expanded_sub_dataset = expand_dataset(organized_patients=sub_dataset,
                                                processed_data_path=processed_data_path,
                                                dataset_name=args.dataset,
                                                split_key=key,
                                                save_dataset=True)
            
            split_patients[key] = expanded_sub_dataset
            print(f'Got {key} expanded dataset')

    if args.task == 'mortality_prediction':
        suffix = 'M'
    elif args.task == 'readmission_prediction':
        suffix = 'R'
    elif args.task == 'los_prediction':
        suffix = 'LOS'
    elif args.task == 'drug_recommendation':
        suffix = 'DR'
    elif args.task == 'phenotype_prediction':
        suffix = 'PH'

    full_configured_dataset = {}

    for key, sub_dataset in split_patients.items():
        configured_sub_dataset = configure_for_task(dataset=sub_dataset,
                                                    filtered_patients=filtered_patients,
                                                    sorted_patients=sorted_patients,
                                                    task=args.task,
                                                    node_id_to_name=file_lib['node_id_to_name'],
                                                    processed_data_path=processed_data_path,
                                                    dataset_name=args.dataset,
                                                    triplet_ratio=args.triplet_ratio,
                                                    split=key,
                                                    suffix=suffix,
                                                    save_dataset=False,
                                                    prescriptions_name_to_code=prescriptions_name_to_code,
                                                    diagnoses_name_to_code=diagnoses_name_to_code,
                                                    phenotype_names=file_lib['phenotype_names'])
                                                    
        full_configured_dataset[key] = configured_sub_dataset

    full_padded_dataset = {}
    for key, sub_dataset in full_configured_dataset.items():
        full_padded_dataset[key] = pad_dataset(dataset=sub_dataset,
                                                code_pad_dim=args.dataset_filtering['args']['code_thresh'],
                                                visit_pad_dim=args.dataset_filtering['args']['visit_thresh'],
                                                processed_data_path=processed_data_path,
                                                dataset_name=args.dataset,
                                                split=key,
                                                suffix=suffix,
                                                triplet_ratio=args.triplet_ratio,
                                                save_dataset=True)
    print('Padded dataset')
    print('Data preparation complete')


if __name__ == '__main__':
    args = get_args()
    run(args=args)
