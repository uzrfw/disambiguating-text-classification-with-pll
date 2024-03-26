""" Module for IPAL. """

from typing import List, Optional

import cvxpy as cp
import numpy as np
from scipy.sparse import lil_array, coo_matrix
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

from data_loading.data import Datasplit
from result import SplitResult


class Ipal:
    """
    IPAL by Zhang and Yu,
    "Solving the Partial Label Learning Problem: An Instance-Based Approach."

    This code was imported from the repository https://github.com/anon1248/uncertainty-aware-pll
    """

    def __init__(
        self, data: Datasplit,
        rng: np.random.Generator,
        k_neighbors: int = 10,
        alpha: float = 0.95,
        max_iterations: int = 3,
    ) -> None:
        self.data = data
        self.rng = rng
        self.num_classes = self.data.orig_dataset.l_classes
        self.alpha = alpha
        self.max_iterations = max_iterations

        # Compute nearest neighbors
        num_insts = self.data.x_train.shape[0]
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=1)
        self.knn.fit(self.data.x_train)
        self.weight_matrix = lil_array((num_insts, num_insts), dtype=float)
        self.initial_confidence_matrix: Optional[np.ndarray] = None
        self.final_confidence_matrix: Optional[np.ndarray] = None

        # Neighborhood weight optimization problem
        num_feats = self.data.orig_dataset.m_features
        self.inst_feats = cp.Parameter(num_feats)
        self.neighbor_feats = cp.Parameter((k_neighbors, num_feats))
        self.weight_vars = cp.Variable(k_neighbors)
        constraints = [self.weight_vars >= 0]
        cost = cp.sum_squares(
            self.inst_feats - self.neighbor_feats.T @ self.weight_vars)
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def _solve_neighbor_weights_prob(
        self, inst_feats: np.ndarray, inst_neighbors: np.ndarray,
    ) -> np.ndarray:
        # Formulate optimization problem
        self.inst_feats.value = inst_feats
        self.neighbor_feats.value = np.vstack([
            self.data.x_train[j] for j in inst_neighbors
        ])
        self.prob.solve(
            solver=cp.MOSEK, mosek_params={"MSK_IPAR_NUM_THREADS": 1})

        # Return weights
        if self.prob.status != "optimal":
            raise ValueError("Failed to find weights.")
        return self.weight_vars.value

    def fit(self) -> None:
        """ Fits the model to the training data. """

        # Compute neighbors for each instance
        nn_indices: np.ndarray = self.knn.kneighbors(return_distance=False)

        print("1")
        # Solve optimization problem to find weights
        for inst, inst_neighbors in enumerate(nn_indices):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                self.data.x_train[inst], inst_neighbors)

            # Store resulting weights
            for neighbor_idx, weight in zip(inst_neighbors, weight_vars):
                if float(weight) > 1e-10:
                    self.weight_matrix[neighbor_idx, inst] = float(weight)

        print("2")
        # Compact information and normalize
        self.weight_matrix = self.weight_matrix.tocoo()
        norm = self.weight_matrix.sum(axis=0)
        self.weight_matrix /= np.where(norm > 1e-10, norm, 1)

        print("3")
        # Initial labeling confidence
        num_insts = self.data.x_train.shape[0]
        initial_labeling_conf = np.zeros((num_insts, self.num_classes))
        for inst in range(num_insts):
            count_labels = max(1, np.count_nonzero(self.data.y_train[inst, :]))
            initial_labeling_conf[inst, self.data.y_train[inst, :] == 1] = \
                1 / count_labels

        print("4")
        # Iterative propagation
        block_size = 32

        curr_labeling_conf = coo_matrix(initial_labeling_conf.copy())
        weight_matrix_transpose_sparse = csr_matrix(coo_matrix(self.weight_matrix.T))
        for _ in range(self.max_iterations):
            # Propagation
            curr_labeling_conf = (
                self.alpha * weight_matrix_transpose_sparse @ curr_labeling_conf +
                (1 - self.alpha) * initial_labeling_conf
            )

            # Rescaling
            for inst in range(num_insts):
                sum_labels = np.sum(
                    curr_labeling_conf[inst, self.data.y_train[inst, :] == 1])
                curr_labeling_conf[inst, :] = np.where(
                    self.data.y_train[inst, :] == 1,
                    curr_labeling_conf[inst, :] / sum_labels,
                    0.0
                )

        # curr_labeling_conf = initial_labeling_conf.copy()
        #
        # initial_labeling_conf_sparse = coo_matrix(initial_labeling_conf)
        # weight_matrix_transpose_sparse = self.weight_matrix.T.tocsr()  # Annahme: weight_matrix ist bereits eine COO-Matrix
        #
        # for _ in range(self.max_iterations):
        #     # Propagation
        #     weighted_sum = self.alpha * weight_matrix_transpose_sparse @ initial_labeling_conf_sparse
        #     curr_labeling_conf = weighted_sum + (1 - self.alpha) * initial_labeling_conf_sparse
        #
        #     # Convert to CSR format for the rescaling step
        #     curr_labeling_conf_csr = curr_labeling_conf.tocsr()
        #
        #     # Rescaling
        #     # Rescaling
        #     for inst in range(num_insts):
        #         indices = self.data.y_train[inst, :] == 1
        #         sum_labels = np.sum(curr_labeling_conf_csr[inst, indices])
        #         non_zero_indices = np.where(indices)[0]
        #         for idx in non_zero_indices:
        #             curr_labeling_conf_csr[inst, idx] /= sum_labels
        #
        #     # Convert back to COO format for the next iteration
        #     curr_labeling_conf_sparse = curr_labeling_conf_csr.tocoo()
        #
        #     # Convert back to COO format for the next iteration
        #     curr_labeling_conf = curr_labeling_conf_csr.tocoo()

        # weight_matrix_T = self.weight_matrix.T
        #
        # num_insts, num_labels = curr_labeling_conf.shape

        # for _ in range(self.max_iterations):
        #     # Propagation with blocking and tiling
        #     for i in range(0, num_insts, block_size):
        #         for j in range(0, num_labels, block_size):
        #             # Compute block-wise matrix multiplication
        #             block_weight_matrix_T = coo_matrix(self.alpha * weight_matrix_T).col[j:j + block_size]
        #             block_curr_labeling_conf = curr_labeling_conf[i:i + block_size, :]
        #
        #             # Perform block-wise multiplication
        #             block_result = coo_matrix((block_size, num_labels))
        #             for k in range(block_size):
        #                 block_result += np.dot(block_weight_matrix_T, block_curr_labeling_conf[k, :])
        #
        #             curr_labeling_conf[i:i + block_size, :] = block_result.toarray() + (
        #                         1 - self.alpha) * initial_labeling_conf[i:i + block_size, :]
        #
        #     # Rescaling
        #     for inst in range(num_insts):
        #         mask = self.data.y_train[inst, :] == 1
        #         sum_labels = np.sum(curr_labeling_conf[inst, mask])
        #         curr_labeling_conf[inst, mask] /= sum_labels
        #         curr_labeling_conf[inst, ~mask] = 0.0

        print("5")
        # Set confidence matrices
        self.initial_confidence_matrix = initial_labeling_conf
        self.final_confidence_matrix = csr_matrix(curr_labeling_conf).toarray()

    def get_train_pred(self) -> SplitResult:
        """ Get the label predictions on the training set.

        Returns:
            SplitResult: The predictions.
        """

        if self.final_confidence_matrix is None or \
                self.initial_confidence_matrix is None:
            raise ValueError("Fit must be called before predict.")

        # Compute class probability masses
        initial_class_mass: np.ndarray =  np.array(np.sum(
            self.initial_confidence_matrix, axis=0))
        final_class_mass: np.ndarray = np.array(np.sum(
            self.final_confidence_matrix, axis=0))

        # Correct for imbalanced class masses
        scores = self.final_confidence_matrix.copy()
        for class_lbl in range(self.num_classes):
            if final_class_mass[class_lbl] > 1e-10:
                scores[:, class_lbl] *= initial_class_mass[class_lbl] / \
                    final_class_mass[class_lbl]

        # Scale probabilities
        prob_sum = np.sum(scores, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = scores / prob_sum
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in scaled_probs:
            max_idx = np.arange(self.data.orig_dataset.l_classes)[
                score_row == np.max(score_row)]
            is_sure.append(bool(np.max(score_row) > 0.5))
            if max_idx.shape[0] == 1:
                pred_list.append(int(max_idx[0]))
                guessing.append(False)
            else:
                pred_list.append(int(self.rng.choice(max_idx)))
                guessing.append(True)

        # Return predictions
        return SplitResult(
            pred=np.array(pred_list),
            is_sure_pred=np.array(is_sure),
            is_guessing=np.array(guessing),
        )

    def get_test_pred(self) -> Optional[SplitResult]:
        """ Get the label predictions on the test set.

        Returns:
            Optional[SplitResult]: The predictions.
        """

        if self.final_confidence_matrix is None or \
                self.initial_confidence_matrix is None:
            raise ValueError("Fit must be called before predict.")
        if self.data.x_test.shape[0] == 0:
            return None

        # Get disambiguated labels of train set
        train_res = self.get_train_pred()
        if train_res is None:
            raise ValueError("Disambiguated labels unavailable.")

        # Solve optimization problem to find weights
        nn_indices = self.knn.kneighbors(
            self.data.x_test, return_distance=False)
        scores_list: List[List[float]] = []
        for test_inst, train_inst_neighbors in enumerate(nn_indices):
            # Formulate optimization problem
            weight_vars = self._solve_neighbor_weights_prob(
                self.data.x_test[test_inst, :], train_inst_neighbors)

            # Use resulting weights
            scores_list.append([])
            for class_lbl in range(self.num_classes):
                class_vector = self.data.x_test[test_inst, :].copy()
                for train_neighbor_idx, train_neighbor_weight in zip(
                    train_inst_neighbors, weight_vars
                ):
                    if class_lbl == train_res.pred[train_neighbor_idx] and \
                            float(train_neighbor_weight) > 1e-10:
                        class_vector -= train_neighbor_weight * \
                            self.data.x_train[train_neighbor_idx]
                scores_list[-1].append(float(
                    np.dot(class_vector, class_vector)))

        # Scale probabilities
        prob = np.array(scores_list)
        prob = np.max(prob, axis=1, keepdims=True) - prob
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        scaled_probs = prob / prob_sum
        pred_list: List[int] = []
        is_sure: List[bool] = []
        guessing: List[bool] = []
        for score_row in scaled_probs:
            max_idx = np.arange(self.data.orig_dataset.l_classes)[
                score_row == np.max(score_row)]
            is_sure.append(bool(np.max(score_row) > 0.5))
            if max_idx.shape[0] == 1:
                pred_list.append(int(max_idx[0]))
                guessing.append(False)
            else:
                pred_list.append(int(self.rng.choice(max_idx)))
                guessing.append(True)

        # Return predictions
        return SplitResult(
            pred=np.array(pred_list),
            is_sure_pred=np.array(is_sure),
            is_guessing=np.array(guessing),
        )
