import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, jaccard_score, cohen_kappa_score


class MetricBase():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def calculate(self, probability, target):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError


class AUROC(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'AUROC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return roc_auc_score(target, probability)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            return roc_auc_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            return roc_auc_score(target, probability, multi_class="ovr", average="macro")

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class AUPRC(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'AUPRC'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            return average_precision_score(target, probability)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            return average_precision_score(target, probability, average="samples")

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Kappa(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'Kappa'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            probability = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, probability)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            return

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Accuracy(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'Accuracy'
        self.task = task

    def calculate(self, probability, target):
        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            pred = (probability >= 0.5).astype(int)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            probability = probability.flatten()
            target = target.flatten()
            pred = (probability >= 0.5).astype(int)
            return accuracy_score(target, pred)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return accuracy_score(target, pred)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class F1(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'F1'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            probability = (probability >= 0.5).astype(int)
            return f1_score(target, probability, average="macro", zero_division=1)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            pred = (probability >= 0.5).astype(int)
            return f1_score(target, pred, average="samples", zero_division=1)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return f1_score(target, pred, average="macro")

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Jaccard(MetricBase):
    def __init__(self, task, **kwargs):

        self.NAME = 'Jaccard'
        self.task = task

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            probability = np.squeeze(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            pred = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="macro", zero_division=1)

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            pred = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="samples", zero_division=1)

        elif self.task == 'los_prediction':
            pred = np.argmax(probability, axis=-1)
            target = np.squeeze(target, axis=-1)
            return cohen_kappa_score(target, pred)

        elif self.task == 'pretrain':
            pred = (probability >= 0.5).astype(int)
            return jaccard_score(target, pred, average="samples", zero_division=1)

    def log(self, score, logger):
        logger.info(f'{self.NAME}: {score:.4f}')


class Precision_K(MetricBase):
    def __init__(self, task, K, **kwargs):

        self.NAME = 'Precision_K'
        self.task = task
        self.K = K

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            return 0

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':
            results = []

            for K in self.K:
                top_k_preds = np.argsort(probability, axis=1)[:, -K:]
                precisions = []

                for i in range(target.shape[0]):
                    true_labels = set(np.where(target[i] == 1)[0])
                    top_k_labels = set(top_k_preds[i])

                    if len(true_labels) > 0:
                        precision_k = len(true_labels.intersection(top_k_labels)) / K
                        precisions.append(precision_k)

                results.append(np.mean(precisions))

            return results

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        for idx, s in enumerate(score):
            logger.info(f'{self.NAME}_{self.K[idx]}: {s:.4f}')


class Recall_K(MetricBase):
    def __init__(self, task, K, **kwargs):

        self.NAME = 'Recall_K'
        self.task = task
        self.K = K

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            return 0

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':

            results = []

            for K in self.K:
                top_k_preds = np.argsort(probability, axis=1)[:, -K:]
                recalls = []

                for i in range(target.shape[0]):
                    true_labels = set(np.where(target[i] == 1)[0])
                    top_k_labels = set(top_k_preds[i])

                    if len(true_labels) > 0:
                        recall_k = len(true_labels.intersection(top_k_labels)) / min(K, len(true_labels))
                        recalls.append(recall_k)

                results.append(np.mean(recalls))

            return results

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        for idx, s in enumerate(score):
            logger.info(f'{self.NAME}_{self.K[idx]}: {s:.4f}')


class NDCG_K(MetricBase):
    def __init__(self, task, K, **kwargs):

        self.NAME = 'NDCG_K'
        self.task = task
        self.K = K

    def dcg_at_k(self, relevance, K):

        relevance = np.asarray(relevance, dtype=float)[:K]
        if relevance.size:
            return np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
        return 0.

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            return 0

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':

            results = []

            for K in self.K:
                ndcg_scores = []

                for i in range(target.shape[0]):
                    true_labels = target[i]
                    pred_scores = probability[i]

                    top_k_indices = np.argsort(pred_scores)[::-1][:K]

                    relevance = true_labels[top_k_indices]

                    dcg = self.dcg_at_k(relevance, K)

                    ideal_relevance = np.sort(true_labels)[::-1]
                    ideal_dcg = self.dcg_at_k(ideal_relevance, K)

                    if ideal_dcg > 0:
                        ndcg_scores.append(dcg / ideal_dcg)
                    else:
                        ndcg_scores.append(0.)

                results.append(np.mean(ndcg_scores))

            return results

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        for idx, s in enumerate(score):
            logger.info(f'{self.NAME}_{self.K[idx]}: {s:.4f}')


class HR_K(MetricBase):
    def __init__(self, task, K, **kwargs):

        self.NAME = 'HR_K'
        self.task = task
        self.K = K

    def calculate(self, probability, target):

        if self.task == 'mortality_prediction' or self.task == 'readmission_prediction':
            return 0

        elif self.task == 'drug_recommendation' or self.task == 'phenotype_prediction':

            results = []

            for K in self.K:
                hits = []

                for i in range(probability.shape[0]):
                    true_labels = set(np.where(target[i] == 1)[0])
                    top_k_indices = set(np.argsort(probability[i])[::-1][:K])

                    if len(true_labels.intersection(top_k_indices)) > 0:
                        hits.append(1)
                    else:
                        hits.append(0)

                results.append(np.mean(hits))

            return results

        elif self.task == 'los_prediction':
            return 0

    def log(self, score, logger):
        for idx, s in enumerate(score):
            logger.info(f'{self.NAME}_{self.K[idx]}: {s:.4f}')


METRICS = {
    'AUROC': AUROC,
    'AUPRC': AUPRC,
    'F1': F1,
    'Accuracy': Accuracy,
    'Kappa': Kappa,
    'Jaccard': Jaccard,
    'Precision_K': Precision_K,
    'Recall_K': Recall_K,
    'NDCG_K': NDCG_K,
    'HR_K': HR_K
}
