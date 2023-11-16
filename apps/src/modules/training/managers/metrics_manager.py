from evaluate import load


class MetricsManager:
    ACCURACY_METRIC = load("accuracy")
    PRECISION_METRIC = load("precision")
    RECALL_METRIC = load("recall")
    F1_METRIC = load("f1")

    @classmethod
    def compute_metrics(cls, prediction_output):
        predictions = prediction_output.predictions.argmax(-1).tolist()
        labels = prediction_output.label_ids.tolist()

        accuracy = cls.ACCURACY_METRIC.compute(predictions=predictions, references=labels)
        precision = cls.PRECISION_METRIC.compute(predictions=predictions, references=labels, average='macro')
        recall = cls.RECALL_METRIC.compute(predictions=predictions, references=labels, average='macro')
        f1 = cls.F1_METRIC.compute(predictions=predictions, references=labels, average='macro')

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"]
        }
