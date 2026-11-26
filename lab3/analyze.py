"""ЛР3. Команда аналізу: перевірка журналів, криві, ефекти й тестова оцінка на CPU.

Запуск із теки lab3/:
    uv run python analyze.py --data data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from common import (
    BATCH_SIZE_EVAL,
    CLASSES,
    IMG_SIZE,
    NUM_WORKERS,
    IntelSplitDataset,
    build_eval_model,
    file_sha256,
    get_transforms,
    load_split_rows,
    predict_logits,
    resolve_dataset_dirs,
)

EXPERIMENT_RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def load_and_verify_runs(runs_dir: Path, split_csv: Path) -> list[dict[str, object]]:
    """Завантажує 7 журналів run.json і перевіряє збіг SHA-256 файлу split.csv."""
    current_hash = file_sha256(split_csv)
    records = []

    for regime, seed in EXPERIMENT_RUNS:
        run_dir = runs_dir / f"{regime}_seed{seed}"
        run_json = run_dir / "run.json"
        val_logits_path = run_dir / "val_logits.npy"

        if not run_json.is_file():
            raise FileNotFoundError(f"Не знайдено журнал запуску: {run_json}")
        if not val_logits_path.is_file():
            raise FileNotFoundError(f"Не знайдено логіти validation: {val_logits_path}")

        rec = json.loads(run_json.read_text(encoding="utf-8"))
        if rec.get("split_csv_sha256") != current_hash:
            raise ValueError(
                f"Хеш split.csv у {run_json} ({rec.get('split_csv_sha256')}) "
                f"не збігається з поточним split.csv ({current_hash})! "
                "Потрібно перенавчити всі 7 запусків на одному поділі."
            )

        logits = np.load(val_logits_path)
        if logits.shape != (1498, len(CLASSES)):
            raise ValueError(f"Неочікувана форма {val_logits_path}: {logits.shape}")

        records.append(rec)

    print(f"Перевірка SHA-256 split.csv у всіх 7 журналах: OK ({current_hash[:16]}...)")
    return records


def plot_training_curves(records: list[dict[str, object]], out_path: Path) -> None:
    """Будує один рисунок із двох панелей: втрата на train і macro-F1 на validation."""
    colors = {"scratch": "tab:gray", "frozen": "tab:blue", "unfreeze": "tab:orange"}
    linestyles = {1: "-", 2: "--", 3: ":"}

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))

    for rec in records:
        regime = str(rec["regime"])
        seed = int(rec["seed"])
        losses = rec["train_loss_per_epoch"]
        f1s = rec["val_macro_f1_per_epoch"]
        epochs = list(range(1, len(losses) + 1))

        label = f"{regime} (seed {seed})"
        style = {
            "color": colors[regime],
            "linestyle": linestyles[seed],
            "marker": "o",
            "markersize": 4,
            "linewidth": 1.8,
            "label": label,
        }
        ax_loss.plot(epochs, losses, **style)
        ax_f1.plot(epochs, f1s, **style)

    ax_loss.set_title("Втрата на train за епохами")
    ax_loss.set_xlabel("Епоха")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_xticks(range(1, 9))
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(fontsize=8)

    ax_f1.set_title("macro-F1 на validation за епохами")
    ax_f1.set_xlabel("Епоха")
    ax_f1.set_ylabel("macro-F1")
    ax_f1.set_xticks(range(1, 9))
    ax_f1.grid(True, linestyle="--", alpha=0.4)
    ax_f1.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Криві 7 запусків збережено: {out_path}")


def compute_effects(records: list[dict[str, object]]) -> dict[str, object]:
    """Пункт 4: обчислює середні, розмахи R_r, поріг шуму R та ефекти Delta_pre і Delta_unf."""
    by_regime: dict[str, list[float]] = {"scratch": [], "frozen": [], "unfreeze": []}
    for rec in records:
        by_regime[str(rec["regime"])].append(float(rec["final_val_macro_f1"]))

    f_scratch = by_regime["scratch"][0]
    f_frozen_mean = float(np.mean(by_regime["frozen"]))
    r_frozen = float(max(by_regime["frozen"]) - min(by_regime["frozen"]))

    f_unfreeze_mean = float(np.mean(by_regime["unfreeze"]))
    r_unfreeze = float(max(by_regime["unfreeze"]) - min(by_regime["unfreeze"]))

    r_noise = max(r_frozen, r_unfreeze)
    delta_pre = f_frozen_mean - f_scratch
    delta_unf = f_unfreeze_mean - f_frozen_mean
    total_gain = f_unfreeze_mean - f_scratch

    def interpret(delta: float) -> str:
        if abs(delta) <= r_noise:
            return "різниця не відрізняється від розкиду (|Δ| <= R)"
        return "ефект вартий перевірки, але не доведений (|Δ| > R)"

    print("\n=== ПУНКТ 4. РОЗКЛАД ЕФЕКТІВ ===")
    print(f"F_scratch,1                  = {f_scratch:.6f}")
    print(f"mean F_frozen,   R_frozen    = {f_frozen_mean:.6f},  R = {r_frozen:.6f}")
    print(f"mean F_unfreeze, R_unfreeze  = {f_unfreeze_mean:.6f},  R = {r_unfreeze:.6f}")
    print(f"Поріг шуму R = max(R_r)      = {r_noise:.6f}")
    print(f"Delta_pre (frozen - scratch) = {delta_pre:+.6f}  -> {interpret(delta_pre)}")
    print(f"Delta_unf (unfreeze - frozen)= {delta_unf:+.6f}  -> {interpret(delta_unf)}")
    print(f"Повний виграш (pre + unf)    = {total_gain:+.6f}")

    return {
        "F_scratch_1": f_scratch,
        "F_frozen_mean": f_frozen_mean,
        "R_frozen": r_frozen,
        "F_unfreeze_mean": f_unfreeze_mean,
        "R_unfreeze": r_unfreeze,
        "R": r_noise,
        "delta_pre": delta_pre,
        "delta_pre_verdict": interpret(delta_pre),
        "delta_unf": delta_unf,
        "delta_unf_verdict": interpret(delta_unf),
        "total_transfer_gain": total_gain,
    }


def verify_selected_model(
    records: list[dict[str, object]], model_path: Path
) -> tuple[dict[str, object], float]:
    """Перевіряє, що lab3_model.pt відповідає правилу вибору за збереженими журналами."""
    by_regime: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_regime.setdefault(str(rec["regime"]), []).append(rec)

    regime_means = {
        reg: float(np.mean([float(r["final_val_macro_f1"]) for r in recs]))
        for reg, recs in by_regime.items()
    }
    expected_regime = max(regime_means, key=lambda r: regime_means[r])
    expected_run = max(
        by_regime[expected_regime], key=lambda r: float(r["final_val_macro_f1"])
    )
    expected_seed = int(expected_run["seed"])
    expected_val_f1 = float(expected_run["final_val_macro_f1"])

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    required_keys = {"state_dict", "regime", "seed", "classes", "img_size"}
    if set(payload.keys()) != required_keys:
        raise ValueError(f"У {model_path} неочікувані ключі: {set(payload.keys())}")

    if payload["regime"] != expected_regime or int(payload["seed"]) != expected_seed:
        raise ValueError(
            f"Модель у {model_path} ({payload['regime']}, seed={payload['seed']}) "
            f"не відповідає правилу вибору ({expected_regime}, seed={expected_seed})!"
        )
    if list(payload["classes"]) != CLASSES or int(payload["img_size"]) != IMG_SIZE:
        raise ValueError("Метадані classes або img_size у lab3_model.pt не відповідають умові.")

    print(
        f"\nПеревірка вибору lab3_model.pt: OK -> regime={expected_regime}, "
        f"seed={expected_seed}, val macro-F1={expected_val_f1:.6f}"
    )
    return payload, expected_val_f1


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Зберігає рисунок матриці плутанини 6x6."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.set_title("Матриця плутанини на test (CPU)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Матрицю плутанини збережено: {out_path}")


def select_and_plot_five_errors(
    test_rows: list[dict[str, str]],
    test_logits: np.ndarray,
    test_targets: np.ndarray,
    cm: np.ndarray,
    data_dir: Path,
    out_path: Path,
) -> list[dict[str, object]]:
    """Пункт 5: відбирає 5 характерних помилок за правилом з умови і зберігає рисунок."""
    _, test_dir = resolve_dataset_dirs(data_dir)
    probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    preds = np.argmax(test_logits, axis=1)

    # Позадіагональні пари (true, pred), відсортовані: спадання кількості, потім зростання (true, pred)
    pairs = []
    for t_idx in range(len(CLASSES)):
        for p_idx in range(len(CLASSES)):
            if t_idx != p_idx and cm[t_idx, p_idx] > 0:
                pairs.append((int(cm[t_idx, p_idx]), t_idx, p_idx))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    # Для кожної пари збираємо всі помилкові файли, відсортовані як рядки
    pair_files: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for idx, row in enumerate(test_rows):
        t_idx = int(test_targets[idx])
        p_idx = int(preds[idx])
        if t_idx != p_idx:
            pair_files.setdefault((t_idx, p_idx), []).append(
                (row["file"], float(probs[idx, p_idx]))
            )
    for key in pair_files:
        pair_files[key].sort(key=lambda item: item[0])

    # Беремо по 1-му файлу з топ-5 пар; якщо пар < 5, добираємо 2-й з найчастішої і т.д.
    selected: list[dict[str, object]] = []
    rank_in_pair = 0
    while len(selected) < 5:
        added_any = False
        for count, t_idx, p_idx in pairs:
            if len(selected) >= 5:
                break
            candidates = pair_files[(t_idx, p_idx)]
            if rank_in_pair < len(candidates):
                fname, pred_prob = candidates[rank_in_pair]
                selected.append(
                    {
                        "file": fname,
                        "true_class": CLASSES[t_idx],
                        "pred_class": CLASSES[p_idx],
                        "pred_prob": pred_prob,
                        "pair_error_count": count,
                    }
                )
                added_any = True
        if not added_any:
            break
        rank_in_pair += 1

    # Малюємо 5 зображень поруч
    fig, axes = plt.subplots(1, len(selected), figsize=(3.4 * len(selected), 3.8))
    if len(selected) == 1:
        axes = [axes]

    for ax, err in zip(axes, selected):
        img_path = test_dir / str(err["true_class"]) / str(err["file"])
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"{err['file']}\n"
            f"true: {err['true_class']}\n"
            f"pred: {err['pred_class']} ({float(err['pred_prob']):.3f})",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle("П'ять характерних помилок на test", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Рисунок 5 характерних помилок збережено: {out_path}")

    print("\nП'ять характерних помилок на test:")
    print(f"{'Файл':<12} {'Справжній':<12} {'Передбач.':<12} {'P(pred)':>10} {'К-сть у test':>14}")
    for err in selected:
        print(
            f"{err['file']:<12} {err['true_class']:<12} {err['pred_class']:<12} "
            f"{float(err['pred_prob']):>10.4f} {int(err['pair_error_count']):>14}"
        )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛР3: аналіз журналів, ефекти та тестова оцінка lab3_model.pt на CPU."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Шлях до локальної копії Intel Image Classification (seg_train, seg_test)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Тека лабораторної з split.csv, runs/ і lab3_model.pt (за замовчуванням .)",
    )
    args = parser.parse_args()

    base_dir = args.dir
    split_csv = base_dir / "split.csv"
    runs_dir = base_dir / "runs"
    figures_dir = base_dir / "figures"
    model_path = base_dir / "lab3_model.pt"

    # 1. Перевірка журналів і SHA-256 split.csv
    records = load_and_verify_runs(runs_dir, split_csv)

    # 2. Таблиця 7 запусків
    print("\n=== ПУНКТ 3. ТАБЛИЦЯ СЕМИ ЗАПУСКІВ ===")
    print(f"{'#':<3} {'Режим':<10} {'Зерно':>5} {'Втрата (еп. 8)':>15} {'val macro-F1 (еп. 8)':>22} {'Час, с':>10}")
    for i, rec in enumerate(records, start=1):
        print(
            f"{i:<3} {rec['regime']:<10} {rec['seed']:>5} "
            f"{float(rec['final_train_loss']):>15.6f} "
            f"{float(rec['final_val_macro_f1']):>22.6f} "
            f"{float(rec['train_time_sec']):>10.2f}"
        )

    # 3. Криві 7 запусків і розклад ефектів
    plot_training_curves(records, figures_dir / "training_curves.png")
    effects = compute_effects(records)

    # 4. Перевірка вибору моделі та ОДНОРАЗОВА оцінка test на CPU
    payload, selected_val_f1 = verify_selected_model(records, model_path)
    cpu_device = torch.device("cpu")
    model = build_eval_model(payload["state_dict"], device=cpu_device)

    _, eval_tf = get_transforms()
    test_rows = load_split_rows(split_csv, part="test")
    test_dataset = IntelSplitDataset(test_rows, args.data, eval_tf)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\n=== ПУНКТ 5. ТЕСТОВА ОЦІНКА НА CPU ===")
    test_logits, test_targets = predict_logits(model, test_loader, device=cpu_device)
    test_preds = np.argmax(test_logits, axis=1)

    test_macro_f1 = float(f1_score(test_targets, test_preds, average="macro"))
    test_accuracy = float(np.mean(test_preds == test_targets))
    n_test = len(test_targets)
    acc_se = math.sqrt(test_accuracy * (1.0 - test_accuracy) / n_test)
    diff_test_val = test_macro_f1 - selected_val_f1

    print(f"Test macro-F1       = {test_macro_f1:.6f}")
    print(f"Selected val F1     = {selected_val_f1:.6f}")
    print(f"Різниця (test - val)= {diff_test_val:+.6f}")
    print(f"Test accuracy (p)   = {test_accuracy:.6f} (n = {n_test})")
    print(f"SE = sqrt(p(1-p)/n) = {acc_se:.6f}")

    cm = confusion_matrix(test_targets, test_preds, labels=list(range(len(CLASSES))))
    print("\nМатриця плутанини 6x6 (рядки = справжній, стовпці = передбачений):")
    header = " " * 12 + "".join(f"{c:>11}" for c in CLASSES)
    print(header)
    for cls, row_vals in zip(CLASSES, cm):
        print(f"{cls:<12}" + "".join(f"{int(v):>11}" for v in row_vals))

    plot_confusion_matrix(cm, figures_dir / "confusion_matrix.png")

    five_errors = select_and_plot_five_errors(
        test_rows=test_rows,
        test_logits=test_logits,
        test_targets=test_targets,
        cm=cm,
        data_dir=args.data,
        out_path=figures_dir / "five_errors.png",
    )

    analysis_summary = {
        "split_csv_sha256": file_sha256(split_csv),
        "effects": effects,
        "selected": {
            "regime": payload["regime"],
            "seed": int(payload["seed"]),
            "val_macro_f1": selected_val_f1,
        },
        "test": {
            "device": "cpu",
            "n_test": n_test,
            "macro_f1": test_macro_f1,
            "accuracy": test_accuracy,
            "accuracy_se": acc_se,
            "test_minus_val_macro_f1": diff_test_val,
            "confusion_matrix": cm.tolist(),
            "five_errors": five_errors,
        },
    }
    (runs_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nПідсумок аналізу збережено у: {(runs_dir / 'analysis_summary.json').resolve()}")


if __name__ == "__main__":
    main()