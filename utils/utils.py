import pickle
import csv
import os.path as osp
from typing import List, Dict


def save_with_pickle(file: object, save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

    return


def read_pickle_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "rb") as f:
        file = pickle.load(f)

    return file


def save_with_csv(file: dict, header: list, save_path: str, file_name: str):

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(file)


def read_csv_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "r") as f:
        file = csv.DictReader(f)
        file = list(file)

    return file


def format_for_csv(input: dict, header: list):

    assert len(header) == 2, 'Formatting only works for length 2 dictionaries.'
    output = []

    for (k, v) in input.items():
        output.append({header[0]: k, header[1]: v})

    return output


def format_from_csv(input: list):

    output = {}
    header = list(input[0].keys())

    for head in header:
        output[head] = []

    for row in input:
        for (k, v) in row.items():
            output[k].append(v)

    return output


def save_txt(file, save_path: str, file_name: str):

    if not file_name.lower().endswith('.txt'):
        file_name += '.txt'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'w') as f:
        for entry in file:
            f.write(f'{entry}\n')

    return


def read_txt_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.txt'):
        file_name += '.txt'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "r") as f:
        data = f.read()

    data = data.split('\n')
    data = data[:-1] if data[-1] == '' else data

    return data


def StringtoList(input: list):

    output = []
    for row in input:
        entry = row.split(' ')
        # entry = re.findall(r'\'.*?\'', row)
        data = []
        data.append(entry[0][1:-2])
        # for element in entry:
        #     _element = element.replace('\'', '')
        #     _element = _element.replace('\'', '')
        #     data.append(_element)
        # output.append(data)

    return output


def format_code_map(input: List[Dict]) -> Dict:

    code_to_name = {}
    name_to_code = {}
    for row in input:
        code_to_name |= {row['code']: row['code_name']}
        name_to_code |= {row['code_name']: row['code']}

    return code_to_name, name_to_code


def convert_to_uppercase(s: str):

    return ''.join([char.upper() if char.isalpha() else char for char in s])
