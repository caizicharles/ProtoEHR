import os
import os.path as osp
import argparse


def create_project_structure(working_dir=""):
    datasets = [
        'mimiciii',
        'mimiciv'
    ]

    tasks = [
        'mortality_prediction',
        'readmission_prediction',
        'los_prediction',
        'drug_recommendation',
        'phenotype_prediction',
    ]

    models = [
        'Deepr',
        'AdaCare',
        'GRASP',
        'StageNet',
        'GraphCare',
        'KerPrint',
        'ProtoEHR',
    ]

    for dataset in datasets:
        path = osp.join(working_dir, 'data/raw', dataset)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")
        path = osp.join(working_dir, 'data/processed', dataset)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

    for dataset in datasets:
        for task in tasks:
            for model in models:
                path = osp.join(working_dir, 'logs', 'bootstrap_results', dataset, task, model)
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path}")
                path = osp.join(working_dir, 'logs', 'checkpoints', dataset, task, model)
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path}")

    path = osp.join(working_dir, 'logs', 'mlflow')
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")


if __name__ == "__main__":
    script_dir = osp.dirname(osp.abspath(__file__))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default=script_dir)
    args = parser.parse_args()

    create_project_structure(working_dir=args.working_dir)
