import time
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.svm import SVC

from data_loading.candidate_set_type import CandidateSetType
from data_loading.data import Dataset, Datasplit
from encoder.encoder_type import EncoderType
from models.baseline.chance_clf import ChanceClf
from models.baseline.constant_clf import ConstantClassifier
from models.baseline.llama2 import Llama2Clf
from models.baseline.ovr_clf import OvrPll
from models.related_work.clpl_2011 import Clpl
from models.related_work.ipal_2015 import Ipal
from models.related_work.pl_ecoc_2017 import PlEcoc
from models.related_work.pl_knn_2005 import PlKnn
from result import Result

COLUMNS = [
    "algo",
    "seed",
    "test_acc",
    "test_mcc",
    "test_frac_guessing",
    "test_frac_sure",
    "test_acc_sure",
    "test_mcc_sure",
    "runtime",
    "encoder_type",
    "candidate_set_type",
    "features"
]

def get_eval_tuple(
        algo_name: str,
        seed: int,
        data: Datasplit,
        result: Result,
        runtime: float,
        encoder_type: EncoderType,
        candidate_set_type: CandidateSetType,
        features: int
) -> List:
    """
    Get a single result row.

    This code was partly imported from the repository https://github.com/anon1248/uncertainty-aware-pll

    Args:
        algo_name (str): The algo name.
        seed (int): The seed.
        data (Datasplit): The data used.
        result (Result): The prediction results.
        runtime (float): Runtime in seconds.
        encoder_type
        candidate_set_type
        features

    Returns:
        List: The result row.
    """

    # Test set evaluation
    if data.y_true_test.shape[0] != 0:
        test_acc = accuracy_score(data.y_true_test, result.get_test_result().pred)
        test_mcc = matthews_corrcoef(data.y_true_test, result.get_test_result().pred)
        test_frac_guessing = result.get_test_result().frac_guessing()
        test_frac_sure = result.get_test_result().frac_sure_predictions()

        test_acc_sure = accuracy_score(
            data.y_true_test[result.get_test_result().is_sure_pred],
            result.get_test_result().sure_predictions(),
        ) if test_frac_sure > 0 else 0.0
        test_mcc_sure = matthews_corrcoef(
            data.y_true_test[result.get_test_result().is_sure_pred],
            result.get_test_result().sure_predictions(),
        ) if test_frac_sure > 0 else 0.0
    else:
        test_acc, test_mcc = 0.0, 0.0
        test_frac_guessing, test_frac_sure = 0.0, 0.0
        test_acc_sure, test_mcc_sure = 0.0, 0.0

    # Build result list
    res_tup = [
        # Algorithm
        algo_name,
        # Seed
        f"{seed}",
        # Accuracy
        f"{test_acc:.6f}",
        # MCC
        f"{test_mcc:.6f}",
        # Fraction guessing
        f"{test_frac_guessing:.6f}",
        # Fraction sure predictions
        f"{test_frac_sure:.6f}",
        # Accuracy sure predictions
        f"{test_acc_sure:.6f}",
        # MCC sure predictions
        f"{test_mcc_sure:.6f}",
        # Runtime
        f"{runtime:.6f}",
        encoder_type,
        candidate_set_type,
        features,
    ]
    return res_tup


def reset_rng(seed: int) -> np.random.Generator:
    """ Creates a new random engine.

    Args:
        seed (int): The seed to use.

    Returns:
        np.random.Generator: The random generator.
    """

    return np.random.Generator(np.random.PCG64(seed))


def run_experiment(
        # Dataset
        dataset: Dataset,
        seed: int,
        algo: str,
):
    """
    Runs a single experiment.

    This code was partly imported from the repository https://github.com/anon1248/uncertainty-aware-pll
    """

    # Reset random engine; assign all algorithms different seeds
    # to ensure all results are mutually independent
    seed_offset = 0

    # Create datasplit
    datasplit = Datasplit.get_dataset(dataset)

    # Run experiments
    res = []

    seed_offset += 1
    if algo in ("all", "baselines", "chance"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # Naive chance baseline
        start = time.process_time()
        chance_clf = ChanceClf(datasplit.copy(), rng)
        result = Result(test_result=chance_clf.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "chance", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "baselines", "ovr-svm"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # One-vs-rest support-vector machine
        start = time.process_time()
        ovr_svm = OvrPll(datasplit.copy(), rng, SVC(random_state=rng.integers(int(1e6)), ))
        result = Result(test_result=ovr_svm.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "ovr-svm", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "baselines", "constant"):
        start = time.process_time()
        constant = ConstantClassifier(datasplit.copy(), 3)
        result = Result(test_result=constant.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "constant", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "baselines", "llama2"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # One-vs-rest support-vector machine
        start = time.process_time()
        llama2 = Llama2Clf(datasplit.copy(), rng)
        result = Result(test_result=llama2.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "llama2", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "pl-knn-2005"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PL-KNN
        start = time.process_time()
        knn = PlKnn(datasplit.copy(), rng)
        result = Result(test_result=knn.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "pl-knn-2005", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "pl-ecoc-2017"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # PL-ECOC
        start = time.process_time()
        pl_ecoc = PlEcoc(datasplit.copy(), rng)
        pl_ecoc.fit()
        result = Result(test_result=pl_ecoc.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "pl-ecoc-2017", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "clpl-2011"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # CLPL
        start = time.process_time()
        clpl = Clpl(datasplit.copy(), rng)
        clpl.fit()
        result = Result(test_result=clpl.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "clpl-2011", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    seed_offset += 1
    if algo in ("all", "ipal-2015"):
        # Reset random engine
        rng = reset_rng(seed + seed_offset)

        # IPAL
        start = time.process_time()
        ipal = Ipal(datasplit.copy(), rng)
        ipal.fit()
        result = Result(test_result=ipal.get_test_pred())
        time_used = time.process_time() - start

        # Append results
        res.append(get_eval_tuple(
            "ipal-2015", seed, datasplit, result, time_used, dataset.encoder_type, dataset.candidate_set_type,
            dataset.features
        ))

    return res
