import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from openai import OpenAI
from tqdm import tqdm
import math
import ast

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from utils.utils import *


def get_wordnet_pos(word):
    """Map POS tag to first character used by WordNetLemmatizer."""

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if POS tag not found


def lemmatize_string(s):
    """Lemmatize each word in the input string."""

    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(s.lower())  # Tokenize and convert to lowercase
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)  # Reconstruct the sentence


def merge_plural_singular(phrases):
    """Merge phrases that differ only in plural/singular forms."""

    # Dictionary to group original phrases by their lemmatized forms
    lemmatized_groups = {}
    merge_log = {}  # Log which words are merged together

    for phrase in phrases:
        lemmatized_phrase = lemmatize_string(phrase)

        # Group phrases by their lemmatized form
        if lemmatized_phrase in lemmatized_groups:
            lemmatized_groups[lemmatized_phrase].append(phrase)

            # Log the merged words
            if lemmatized_phrase not in merge_log:
                merge_log[lemmatized_phrase] = set(lemmatized_groups[lemmatized_phrase])
            else:
                merge_log[lemmatized_phrase].add(phrase)
        else:
            lemmatized_groups[lemmatized_phrase] = [phrase]
            merge_log[lemmatized_phrase] = set([phrase])

    return lemmatized_groups, merge_log


def merge_edge_type(edge_types: np.ndarray):
    relations = []
    remove_rel = []
    merge_map = {}

    for edge_type in edge_types:
        relations.append(edge_type.strip().lower())

    ### 'can be'
    for x in relations:
        for y in relations:
            if 'can be ' == y[0:7] and y[7:] == x:
                remove_rel.append(y)
                merge_map[y] = x
                break

    ### 'the'
    for x in relations:
        if ' the ' in x:
            xx = x.replace(' the ', ' ')
            for y in relations:
                if xx == y:
                    remove_rel.append(y)
                    merge_map[y] = x
                    break

    ### 'by v.s. with'
    for x in relations:
        if ' with' in x:
            xx = x.replace(' with', ' by')
            for y in relations:
                if xx == y:
                    remove_rel.append(y)
                    merge_map[y] = x
                    break

    ### 'help to v.s. help'
    for x in relations:
        if ' help to ' in x:
            xx = x.replace(' help to ', ' help ')
            for y in relations:
                if xx == y:
                    remove_rel.append(y)
                    merge_map[y] = x
                    break

    remove_rel = np.array(list(merge_map.keys()))

    manually_merged_relations = np.setdiff1d(relations, remove_rel)

    # Merge similar phrases and get log of merged words
    _, merge_log = merge_plural_singular(manually_merged_relations)

    for k, v in merge_log.items():
        for element in v:
            if element in merge_map.keys():
                print(k, v)
                print(element, merge_map[element])
                exit()
            else:
                merge_map[element] = k

    merge_map['uses'] = 'use'
    merge_map['can cause respiratory distress leading to increased intra-abdominal pressure and'] = 'can lead to'
    merge_map[
        'is a treatment option for patients who have experienced complications following'] = 'is a treatment option for'
    merge_map[
        'is a treatment option for patients who have undergone previous surgical interventions including'] = 'be a treatment option'

    return merge_map


def encode_for_BERT(strings: np.ndarray):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoding = tokenizer(strings.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask


def get_BERT_embeddings(strings: np.ndarray):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    inputs = tokenizer(strings.tolist(), return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]

    return embedding


def verify_clusters(edge_types: np.ndarray, assignments: np.ndarray):

    client = OpenAI(
        api_key=
        ""
    )
    MODEL = 'gpt-4'

    full_output = []

    for cluster_idx in tqdm(np.unique(assignments)):
        mask = assignments == cluster_idx
        cluster = edge_types[mask]

        for idx, edge in enumerate(cluster):
            cluster[idx] = str(idx + 1) + '-' + edge

        cluster_input = "[" + ", ".join(f"'{item}'" for item in cluster) + "]"

        completion = client.chat.completions.create(
            model=MODEL,
            seed=0,
            temperature=0,
            messages=[{
                "role":
                    "system",
                "content":
                    '''You are a medical expert that has rich knowledge. You can easily identify wether 
                    medical relations are semantically opposed.'''
            }, {
                "role":
                    "user",
                "content":
                    f'''given the following list of medical relations, assert if they are semantically opposed to each other. if no, 
                    output the original cluster; if yes, group words into 2 subgroups such that the subgroups have opposing semantic meaning, 
                    each relation in each subgroup is preceeded by the relationship id, then output the 2 subgroups.
                    No explanation is needed in the output.
                    Below is an example of input and output.

                    Example Input 1:
                    ['1-administered during',
                    '2-administered through',
                    '3-administered via',
                    '4-administered with',
                    '5-detect by',
                    '6-detected by',
                    '7-differentiate from',
                    '8-differentiated from',
                    '9-enhanced by',
                    '10-followed by',
                    '11-help monitor',
                    '12-monitor',
                    '13-monitor by',
                    '14-no know relationship',
                    '15-not detect by',
                    '16-not directly related to',
                    '17-not effective against',
                    '18-not effective for',
                    '19-not use for',
                    '20-regulate by',
                    '21-regulated by',
                    '22-related to',
                    '23-target',
                    '24-treated with',
                    '25-unrelated to']

                    Example Output 1:
                    ['1-administered during',
                    '2-administered through',
                    '3-administered via',
                    '4-administered with',
                    '5-detect by',
                    '6-detected by',
                    '7-differentiate from',
                    '8-differentiated from',
                    '9-enhanced by',
                    '10-followed by',
                    '11-help monitor',
                    '12-monitor',
                    '13-monitor by',
                    '14-no know relationship',
                    '15-not detect by',
                    '16-not directly related to',
                    '17-not effective against',
                    '18-not effective for',
                    '19-not use for',
                    '20-regulate by',
                    '21-regulated by',
                    '22-related to',
                    '23-target',
                    '24-treated with',
                    '25-unrelated to']

                    Example Input 2:
                    ['1-be not',
                    '2-be not typically associate with',
                    '3-be often associate with', 
                    '4-do not directly relate to',
                    '5-do not relate to',
                    '6-not prevent by',
                    '7-not treat by']

                    Example Output 2:
                    ['1-be not',
                    '2-be not typically associate with',
                    '4-do not directly relate to',
                    '5-do not relate to',
                    '6-not prevent by',
                    '7-not treat by'],
                    ['3-be often associate with']
                    
                    Input: {cluster_input}
                    Output:'''
            }])

        output = completion.choices[0].message.content

        if output[0] == '[' and output[1] == '[':
            output = ast.literal_eval(output)
        elif output[0] == '[' and output[1] != '[':
            output = "[" + output + "]"
            output = ast.literal_eval(output)
        else:
            raise ValueError

        num = 0
        for subgroup in output:
            num += len(subgroup)

        if len(cluster) != num:
            print(cluster_idx)
            print(cluster)
            print(cluster_input)
            print(output)
            exit()

        for sub_idx, subgroup in enumerate(output):
            for element_idx, element in enumerate(subgroup):
                output[sub_idx][element_idx] = element.split('-', 1)[1]

        full_output.extend(output)

    merge_map = {}
    for cluster in full_output:
        for element in cluster:
            merge_map[element] = cluster[0]

    return merge_map


def verify_embeddings(triplets: np.ndarray, pos_num: int = None):

    client = OpenAI(
        api_key=
        ""
    )
    MODEL = 'gpt-4'

    call_triplet_num = 50
    triplets = triplets.tolist()

    full_output = []
    pos = []
    neg = []

    for idx in tqdm(range(math.ceil(len(triplets) / call_triplet_num))):
        start_idx = idx * call_triplet_num
        stop_idx = (idx + 1) * call_triplet_num
        if stop_idx > len(triplets):
            stop_idx = len(triplets)

        input = triplets[start_idx:stop_idx]

        s0 = ''
        for ii, hh in enumerate(input):
            s0 += str(ii + 1) + '.(' + ', '.join(hh) + '); '

        completion = client.chat.completions.create(
            model=MODEL,
            seed=0,
            temperature=0,
            messages=[{
                "role":
                    "system",
                "content":
                    '''You are a medical expert that has rich knowledge on medical concepts including diagnosis, drugs, testing procedures, etc. 
                Especially, you can easily identify wether two medical concepts exist a given relationship or effect.'''
            }, {
                "role":
                    "user",
                "content":
                    f'''Given several triplets in the form of '1. (ENTITY_head, RELATION, ENTITY_tail); 2. (ENTITY_head, RELATION, ENTITY_tail); ...', the index denotes triple identifier,  
                    ENTITY_head and ENTITY_tail are either medical diagnoses, procedures, or drugs, RELATION is the guessed relationship or effect between ENTITY_head and ENTITY_tail. 
                    ascertain whether each the guessed relationship is true or false, and return the results in the form of 'index. True' or 'index. False', correspondingly. Each triplet should has output.
                    No explanation is needed in the output.
                    Below is an example of input and output.

                    Example:
                    Input: 1.('ABDOMINAL HERNIA', 'CAN AGGRAVATE', 'ASTHMA'); 2.('ABDOMINAL HERNIA', 'CAN AGGRAVATE', 'ESOPHAGEAL DISORDERS'); 3.('FRACTURE OF NECK OF FEMUR (HIP)', 'CAN CAUSE', 'OTHER HEMATOLOGIC CONDITIONS')
                    Output: 1.True; 2.True; 3.False
                    
                    Input: {s0}
                    Output:'''
            }])

        output = completion.choices[0].message.content
        output = [eval(item.split('.')[1]) for item in output.split(';') if item]
        assert len(input) == len(output), f'I/O shape mismatch: {len(input)} {len(output)}'
        full_output.extend(output)

        count = np.array(full_output, dtype=int)

        if pos_num is not None:
            if count.sum() > pos_num:
                break

    for idx, o in enumerate(full_output):
        if o:
            pos.append(triplets[idx])
        else:
            neg.append(triplets[idx])

    return np.array(pos), np.array(neg)
