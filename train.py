import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

# from categories.accuracy import get_bertscore
# from categories.fluency import ppll_loss, grammar_errors


def sigma(x, k, mu):
    return 100.0 / (1.0 + np.exp(-k * (x - mu)))


def load_dataset(fp):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_arrays(data):
    s_cos, J, G = [], [], []
    acc_targets, flu_targets = [], []

    for ex in data:
        src, tgt = ex["german"], ex["english"]

        # cosine similarity for accuracy
        # s = get_bertscore(src, tgt)          # [-1,1]
        # s_cos.append(s)

        # # pseudo‑perplexity loss J  and grammar score G
        # pp = ppll_loss(tgt)
        # J.append(pp["loss"] if "loss" in pp else pp["score"])  # we’ll use score below

        # ge = grammar_errors(tgt)
        # G.append(ge["score"])

        s_cos.append(ex["bertscore"])  # [0,1]
        J.append(ex["fluency_score"])  # [0,1]
        G.append(ex["grammar_score"])  # [0,1]

        acc_targets.append(ex["accuracy"])
        flu_targets.append(ex["fluency"])

    # to numpy
    return (
        np.array(s_cos),
        np.array(J, dtype=float),
        np.array(G, dtype=float),
        np.array(acc_targets, dtype=float),
        np.array(flu_targets, dtype=float),
    )


def fit_accuracy(s_cos, acc_target):
    def resid(params, x, y):
        lam, k1, mu1, k2, mu2 = params
        s1 = sigma(x, k1, mu1)
        s2 = sigma(x, k2, mu2)
        pred = lam * s1 + (1 - lam) * s2
        return pred - y

    init = [0.5, 5.0, 0.6, 11.0, 0.6]
    bounds = ([0.2, 1.0, 0.4, 5.0, 0.4], [0.8, 11.0, 0.8, 20.0, 0.8])

    res = least_squares(resid, init, args=(s_cos, acc_target), bounds=bounds)
    lam, k1, mu1, k2, mu2 = res.x
    return dict(lam=lam, k1=k1, mu1=mu1, k2=k2, mu2=mu2)


# ---------------------------------------------------------------------
# === 4. fit fluency parameters ===
def fit_fluency(J, G, flu_target):
    def resid(params, J_, G_, y):
        lam, kP, muP, kG, muG = params
        P = sigma(J_, kP, muP)
        G = sigma(G_, kG, muG)
        pred = lam * P + (1 - lam) * G
        return pred - y

    init = [0.5, 0.1, 5.0, 0.1, 5.0]
    bounds = ([0.2, 0, 0, 0, 0], [1, np.inf, np.inf, np.inf, np.inf])
    res = least_squares(resid, init, args=(J, G, flu_target), bounds=bounds)
    lam, kP, muP, kG, muG = res.x
    return dict(lambda_F=lam, k_P=kP, mu_P=muP, k_G=kG, mu_G=muG)


# ---------------------------------------------------------------------
def main(in_path, out_path):
    print("Loading dataset from", in_path)
    data = load_dataset(in_path)

    print("Building arrays...")
    s_cos, J, G, acc_t, flu_t = build_arrays(data)

    print("Fitting accuracy parameters...")
    acc_params = fit_accuracy(s_cos, acc_t)
    print("Fitting fluency parameters...")
    flu_params = fit_fluency(J, G, flu_t)

    # Round all parameters to the nearest hundredth
    acc_params = {k: round(v, 2) for k, v in acc_params.items()}
    flu_params = {k: round(v, 2) for k, v in flu_params.items()}

    params = {"accuracy_params": acc_params, "fluency_params": flu_params}

    Path(out_path).write_text(json.dumps(params, indent=2))
    print("Saved parameters to", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fit_qe_params.py translations.json fitted_params.json")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
