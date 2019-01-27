import csv
import re
import sys

import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_recall_curve
from nltk.corpus import stopwords


def read_test_data(fname):
    with open(fname) as f:
        cr = csv.reader(f)
        next(cr)  # headers
        for row in cr:
            prem, hypo, gold_label = row
            yield prem, hypo, gold_label == "yes"


def no_spaces(s):
    return s.replace(" ", "_SPC_")


def untypify(typed_relation):
    return re.sub(
        r'~[0-9]+~-([^~]+?)-~[0-9]+~',
        r'\1', typed_relation
    )


def words_from_path(path):
    return [
        w
        for i, w in enumerate(path.split("___"))
        if i % 2 == 1
    ]


def voice_of_path(path):
    if path.startswith("nsubjpass") or path.endswith("nsubjpass") or path.endswith("nsubjpass^-"):
        return "passive"
    else:
        return "active"


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.fitted_ = False

    def check_input(self, X):
        if any([len(x) != 1 for x in X]):
            raise ValueError("Every sample should have exactly one feature.")

    def fit(self, X, y):
        self.check_input(X)

        scores = [x[0] for x in X]
        pre, rec, thr = precision_recall_curve(y, scores)
        f1 = [
            2 * p * r / (p+r) if (p+r) > 0.0 else 0.0
            for p, r in zip(pre, rec)
        ]

        self.thr_ = max(zip(f1, thr), key=lambda x: x[0])[1]
        self.fitted_ = True

        return self

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("You have to call fit() first.")
        self.check_input(X)

        scores = [x[0] for x in X]
        pred = [s >= self.thr_ for s in scores]
        return pred


def lemma(path_prem, path_hypo):
    path_prem = untypify(path_prem)
    path_hypo = untypify(path_hypo)

    stop_words = stopwords.words('english')
    pr_lemmata = words_from_path(path_prem)
    hy_lemmata = words_from_path(path_hypo)

    # 1. Criterion: has prem all content words of hypo?
    all_content_words_there = True
    for w in hy_lemmata:
        if w in stop_words:
            continue
        if w not in pr_lemmata:
            all_content_words_there = False
            break

    pr_is_inversed = not path_prem.startswith("nsubj")
    hy_is_inversed = not path_hypo.startswith("nsubj")

    # 2. Criterion: is predicate the same?
    pr_pred = pr_lemmata[-1] if pr_is_inversed else pr_lemmata[0]
    hy_pred = hy_lemmata[-1] if hy_is_inversed else hy_lemmata[0]
    same_predicate = pr_pred == hy_pred

    # 3. Criterion: is voice and inversement the same?
    voice_pr = voice_of_path(path_prem)
    voice_hy = voice_of_path(path_hypo)
    same_voice = voice_pr == voice_hy
    same_inversement = pr_is_inversed == hy_is_inversed
    third_criterion = same_voice == same_inversement

    return all_content_words_there and same_predicate and third_criterion


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="This script evaluates relation embeddings "
        + "for relation inference via cross validation."
    )
    parser.add_argument(
        "embeddings", metavar="embeddings.txt",
        help="relation embeddings in word2vec text format"
    )
    parser.add_argument(
        "testset", metavar="test_set.csv",
        help="annotated relation pairs for testing (as .csv)"
    )
    parser.add_argument(
        "--num-folds", "-n", type=int,
        default=5, dest="num_folds",
        help="number of folds for cross validation (default: 5)"
    )
    parser.add_argument(
        "--with-types", "-t", action="store_true", dest="with_types",
        help="embeddings are typed (default: untyped)"
    )
    parser.add_argument(
        "--with-lemma", "-l", action="store_true", dest="with_lemma",
        help="embeddings are run on top of a simple lemma baseline (default: off)"
    )
    parser.add_argument(
        "--treat-missing-as-false", "-f", action="store_true", dest="missing_false",
        help="for relation pairs where the embedding of one relation is missing " +
        "a similarity of 0.0 is predicted (default: exclude those cases from test set or " +
        "fall back to lemma baseline) "
    )
    args = parser.parse_args()

    print("Loading embeddings ...", end=" ", file=sys.stderr, flush=True)
    vectors = KeyedVectors.load_word2vec_format(args.embeddings, binary=False)
    print("Done.", file=sys.stderr, flush=True)

    lemma_predictions = []
    scores, truth = [], []
    for prem, hypo, gold_label in read_test_data(args.testset):
        if args.with_lemma:
            lemma_predictions.append(
                lemma(prem, hypo)
            )

        if not args.with_types:
            prem = untypify(prem)
            hypo = untypify(hypo)

        try:
            scores.append(
                vectors.similarity(
                    no_spaces(prem),
                    no_spaces(hypo)
                )
            )
        except KeyError:
            if args.with_lemma or args.missing_false:
                scores.append(0.0)
            else:
                continue

        truth.append(gold_label)

    if not truth:
        print(
            "[ERROR] No valid samples for evaluation: " +
            "Embedding file does not contain enough embeddings.",
            file=sys.stderr
        )
        exit(-1)

    if args.with_lemma:
        lemma_score = max(scores)
        scores = [
            lemma_score if lemma_pred else s
            for s, lemma_pred in zip(scores, lemma_predictions)
        ]

    print(
        "Results based on cross validation on {} samples with {} folds.".format(
            len(truth), args.num_folds)
    )

    feats = np.array(scores).reshape(-1, 1)
    clf = ThresholdClassifier()
    cv = StratifiedKFold(n_splits=args.num_folds, random_state=2375)

    precs = cross_val_score(
        clf, feats, truth, groups=truth,
        cv=cv, scoring='precision'
    )
    recs = cross_val_score(
        clf, feats, truth, groups=truth,
        cv=cv, scoring='recall'
    )
    f1s = cross_val_score(
        clf, feats, truth, groups=truth,
        cv=cv, scoring='f1'
    )

    print(
        "Mean Precision: {:.3f}\nMean Recall: {:.3f}\nMean F1: {:.3f}".format(
            precs.mean(), recs.mean(), f1s.mean()
        )
    )
