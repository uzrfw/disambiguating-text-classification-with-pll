class Evaluator:
    def __init__(self, model_outputs: [str], targets: [str]):
        """
        Initializes the Evaluator with model outputs and target values.

        :param: model_outputs: The outputs produced by the model.
        :param: targets: The target values for evaluation.
        """
        self.model_outputs = model_outputs
        self.targets = targets

    def evaluate(self):
        """
        Evaluates the model outputs against the target values.

        :raise: Exception: If the lengths of model_outputs and targets are not equal.
        """

        # Check if the lengths of model outputs and targets are the same
        if len(self.model_outputs) != len(self.targets):
            raise Exception("Error: Model outputs and targets have different lengths")

        class_labels = [str(i) for i in range(1, 10)]

        # Initialize metrics
        accuracy = 0
        precision = {label: 0 for label in class_labels}
        recall = {label: 0 for label in class_labels}

        total_samples = len(self.targets)

        for i in range(total_samples):
            true_labels = self.targets[i]
            predicted_labels = self.model_outputs[i]

            # Accuracy
            if any(label in true_labels for label in predicted_labels):
                accuracy += 1

            # Precision and Recall
            for label in class_labels:
                if label in predicted_labels:
                    if label in true_labels:
                        precision[label] += 1
                if label in true_labels:
                    if label in predicted_labels:
                        recall[label] += 1

        accuracy /= total_samples

        for label in class_labels:
            denominator_precision = max(sum(1 for j in range(total_samples) if label in self.model_outputs[j]), 1)
            denominator_recall = max(sum(1 for j in range(total_samples) if label in self.targets[j]), 1)

            precision[label] /= denominator_precision
            recall[label] /= denominator_recall

        print("Accuracy:" + str(accuracy))
        print("Precision:" + str(precision))
        print("Recall:" + str(recall))

        return accuracy
