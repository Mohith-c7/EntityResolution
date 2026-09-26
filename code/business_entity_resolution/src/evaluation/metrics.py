"""Challenge metric, including empty truth and empty prediction semantics."""

import numpy as np


def entity_f05(truth, predictions) -> float:
    truth, predictions = set(truth), set(predictions)
    if not truth:
        return float(not predictions)
    return 1.25 * len(truth & predictions) / (.25 * len(truth) + len(predictions))


def score_matches(truth: dict, predictions: dict, countries: dict | None = None) -> dict:
    if set(predictions) - set(truth):
        raise ValueError("Predictions contain unknown reference IDs")
    scores = []
    tp = total_pred = total_true = singleton = singleton_fp = non_singleton = empty_non_singleton = 0
    country_scores = {}
    for entity, expected in truth.items():
        expected, predicted = set(expected), set(predictions.get(entity, ()))
        value = entity_f05(expected, predicted)
        scores.append(value)
        tp += len(expected & predicted)
        total_true += len(expected)
        total_pred += len(predicted)
        if expected:
            non_singleton += 1
            empty_non_singleton += not bool(predicted)
        else:
            singleton += 1
            singleton_fp += bool(predicted)
        if countries is not None:
            country_scores.setdefault(countries[entity], []).append(value)
    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "entities": len(scores), "true_positive_links": tp, "predicted_links": total_pred, "true_links": total_true,
        "micro_precision": tp / total_pred if total_pred else 0.0,
        "micro_recall": tp / total_true if total_true else 0.0,
        "singleton_entities": singleton, "singleton_false_positives": singleton_fp,
        "singleton_false_positive_rate": singleton_fp / singleton if singleton else None,
        "non_singleton_empty_rate": empty_non_singleton / non_singleton if non_singleton else None,
        "country_macro_f05": {country: {"entities": len(values), "macro_f05": float(np.mean(values))} for country, values in country_scores.items()},
    }


def bootstrap_interval(truth: dict, predictions: dict, *, seed: int = 42, iterations: int = 2000) -> list[float]:
    scores = np.array([entity_f05(expected, predictions.get(eid, ())) for eid, expected in truth.items()])
    if not len(scores):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(scores, size=len(scores), replace=True).mean()) for _ in range(iterations)]
    return list(map(float, np.quantile(means, [.025, .975])))
