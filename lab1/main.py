import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris


def prepare_data():
    """Завантаження Iris, стратифікований поділ 70/30 та стандартизація."""
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(0)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for c in [0, 1, 2]:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train_idx, test_idx = idx[:35], idx[35:]
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def init_params():
    """Ініціалізація параметрів мережі за схемами He та Xavier."""
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4.0), size=(4, 8)).astype(np.float64)
    b1 = np.zeros(8, dtype=np.float64)
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8.0 + 3.0)), size=(8, 3)).astype(np.float64)
    b2 = np.zeros(3, dtype=np.float64)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(X, y, params):
    """Прямий прохід мережі та обчислення стабільної крос-ентропії."""
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2

    max_logits = np.max(Z2, axis=1, keepdims=True)
    shifted_logits = Z2 - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
    log_probs = shifted_logits - log_sum_exp

    N = X.shape[0]
    loss = -np.mean(log_probs[np.arange(N), y])

    cache = {"X": X, "y": y, "Z1": Z1, "A1": A1, "log_probs": log_probs}
    return loss, cache


def backward(params, cache, remove_n_division=False):
    """Зворотне поширення похибки для обчислення градієнтів."""
    X, y = cache["X"], cache["y"]
    Z1, A1, log_probs = cache["Z1"], cache["A1"], cache["log_probs"]
    W2 = params["W2"]
    N = X.shape[0]

    probs = np.exp(log_probs)
    dZ2 = probs.copy()
    dZ2[np.arange(N), y] -= 1.0
    if not remove_n_division:
        dZ2 /= N

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0.0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def pytorch_reference(X, y, params):
    """Еталонний розрахунок втрати та градієнтів за допомогою PyTorch."""
    torch.set_default_dtype(torch.float64)
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(params["W1"].T, dtype=torch.float64))
        model[0].bias.copy_(torch.tensor(params["b1"], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor(params["W2"].T, dtype=torch.float64))
        model[2].bias.copy_(torch.tensor(params["b2"], dtype=torch.float64))

    logits = model(X_t)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, y_t)
    loss.backward()

    pt_grads = {
        "W1": model[0].weight.grad.T.numpy(),
        "b1": model[0].bias.grad.numpy(),
        "W2": model[2].weight.grad.T.numpy(),
        "b2": model[2].bias.grad.numpy(),
    }
    return loss.item(), pt_grads


def numerical_gradients(X, y, params, eps=1e-6):
    """Чисельне диференціювання вибраних скалярних параметрів."""
    targets = [
        ("W1[0, 0]", "W1", (0, 0)),
        ("b1[0]", "b1", (0,)),
        ("W2[0, 0]", "W2", (0, 0)),
        ("b2[0]", "b2", (0,)),
    ]
    results = {}

    for label, name, idx in targets:
        orig_val = params[name][idx]

        params[name][idx] = orig_val + eps
        loss_plus, _ = forward(X, y, params)

        params[name][idx] = orig_val - eps
        loss_minus, _ = forward(X, y, params)

        params[name][idx] = orig_val
        g_num = (loss_plus - loss_minus) / (2.0 * eps)
        results[label] = (name, idx, g_num)

    return results


def run_checks(remove_n_division=False):
    X_train, y_train, _, _ = prepare_data()
    params = init_params()

    np_loss, cache = forward(X_train, y_train, params)
    np_grads = backward(params, cache, remove_n_division=remove_n_division)
    pt_loss, pt_grads = pytorch_reference(X_train, y_train, params)

    mode_title = (
        "ДОСЛІД ІЗ ПОМИЛКОЮ (без ділення на N)"
        if remove_n_division
        else "ОСНОВНА РЕАЛІЗАЦІЯ (правильна)"
    )
    print(f"=== {mode_title} ===\n")
    print(f"Втрата NumPy:   {np_loss:.15f}")
    print(f"Втрата PyTorch: {pt_loss:.15f}\n")

    print("### Звірка з PyTorch")
    print("| Величина | Максимальна абсолютна різниця NumPy / PyTorch | Перевірку пройдено |")
    print("| :--- | :--- | :--- |")

    loss_diff = abs(np_loss - pt_loss)
    loss_ok = np.isfinite(np_loss) and np.isfinite(pt_loss) and (loss_diff <= 1e-12)
    print(f"| Втрата | {loss_diff:.3e} | {'Так' if loss_ok else 'Ні'} |")

    for name in ["W1", "b1", "W2", "b2"]:
        g_np, g_pt = np_grads[name], pt_grads[name]
        finite = np.all(np.isfinite(g_np)) and np.all(np.isfinite(g_pt))
        max_diff = np.max(np.abs(g_np - g_pt))
        passed = finite and (max_diff <= 1e-12)
        print(f"| Градієнт {name} | {max_diff:.3e} | {'Так' if passed else 'Ні'} |")

    print("\n### Перевірка чисельним диференціюванням")
    print("| Параметр | Градієнт backward() | Чисельна похідна | Абсолютна різниця | Перевірку пройдено |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    num_res = numerical_gradients(X_train, y_train, params, eps=1e-6)
    for label, (name, idx, g_num) in num_res.items():
        g_man = np_grads[name][idx]
        diff = abs(g_num - g_man)
        passed = np.isfinite(g_num) and np.isfinite(g_man) and (diff <= 1e-7)
        print(f"| {label} | {g_man:.10f} | {g_num:.10f} | {diff:.3e} | {'Так' if passed else 'Ні'} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ЛР1: Зворотне поширення похибки")
    parser.add_argument(
        "--buggy",
        action="store_true",
        help="Запустити дослід із навмисною помилкою (без ділення на N у градієнті логітів)",
    )
    args = parser.parse_args()
    run_checks(remove_n_division=args.buggy)