"""ЛР2. Аналіз порога рішення під асиметричну ціну помилок (пункти 2–4).

Запуск без повторного навчання та без creditcard.csv:
    uv run python analyze.py --saved results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Вимикаємо інтерактивні вікна, щоб графіки зберігалися у PNG на будь-якій машині
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions

FN_COST = 10
FP_COST = 1


def evaluate_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int | float]:
    """Самостійно обчислює TP, FP, FN, TN і C = 10*FN + FP за правилом s_i >= t."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    # Прогнозуємо шахрайство (1), якщо оцінка >= порога
    y_pred = (s >= threshold).astype(np.int64)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    cost = FN_COST * fn + FP_COST * fp

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": int(cost),
    }


def compute_precision_recall(
    tp: int, fp: int, fn: int
) -> tuple[float | None, float | None]:
    """Обчислює precision і recall; повертає None, якщо знаменник дорівнює 0."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return precision, recall


def search_threshold_fast(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[pd.DataFrame, float, int]:
    """Пошук оптимального порога за O(n log n) через сортування і кумулятивні суми.

    Кандидати: +inf та всі унікальні оцінки у порядку спадання.
    Рівні оцінки завжди отримують однаковий прогноз.
    За рівної мінімальної вартості обирається найбільший поріг.
    """
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    if len(y) != len(s) or len(y) == 0:
        raise ValueError("Масиви y_true та scores мають бути непорожніми і рівної довжини.")

    # 1. Сортуємо оцінки за спаданням за O(n log n)
    order = np.argsort(s, kind="mergesort")[::-1]
    s_sorted = s[order]
    y_sorted = y[order]

    total_pos = int(np.sum(y_sorted == 1))
    total_neg = len(y_sorted) - total_pos

    # 2. Кумулятивні суми за O(n): скільки TP і FP накопичилося серед перших k+1 елементів
    cum_tp = np.cumsum(y_sorted)
    cum_pred_pos = np.arange(1, len(y_sorted) + 1, dtype=np.int64)
    cum_fp = cum_pred_pos - cum_tp

    # 3. Знаходимо кінець кожної групи рівних оцінок, щоб не "розрізати" однакові s_i
    is_group_end = np.empty(len(s_sorted), dtype=bool)
    is_group_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    is_group_end[-1] = True

    # 4. Додаємо на початок граничний поріг +inf ("усі негативні": TP=0, FP=0)
    thresholds = np.concatenate(([np.inf], s_sorted[is_group_end]))
    tp = np.concatenate(([0], cum_tp[is_group_end]))
    fp = np.concatenate(([0], cum_fp[is_group_end]))
    fn = total_pos - tp
    tn = total_neg - fp
    cost = FN_COST * fn + FP_COST * fp

    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": cost,
        }
    )

    # 5. Оскільки пороги йдуть строго за спаданням, перший мінімум = найбільший поріг
    best_idx = int(np.argmin(cost))
    best_threshold = float(thresholds[best_idx])
    best_cost = int(cost[best_idx])

    return table, best_threshold, best_cost


def search_threshold_naive(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[pd.DataFrame, float, int]:
    """Незалежний прямий перебір за правилом s_i >= t (лише для перевірки на наборах A/B/C)."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)

    # Формуємо кандидатів незалежно від швидкої функції: +inf та унікальні оцінки за спаданням
    unique_desc = np.unique(s)[::-1]
    candidates = np.concatenate(([np.inf], unique_desc))

    rows = []
    best_threshold = float("inf")
    best_cost = None

    for t in candidates:
        row = evaluate_threshold(y, s, float(t))
        rows.append(row)

        # Явне правило розв'язання нічиєї: менша вартість, а при рівній — більший поріг
        c = int(row["C"])
        t_val = float(row["threshold"])
        if best_cost is None or c < best_cost or (c == best_cost and t_val > best_threshold):
            best_cost = c
            best_threshold = t_val

    table = pd.DataFrame(rows)
    return table, best_threshold, int(best_cost)


def verify_on_small_sets(out_dir: Path) -> dict[str, dict[str, object]]:
    """Пункт 3: перевірка швидкого алгоритму проти прямого перебору на наборах A, B, C."""
    datasets = {
        "A": {
            "scores": [0.9, 0.7, 0.7, 0.4, 0.2, 0.1],
            "y": [0, 1, 0, 1, 0, 0],
        },
        "B": {
            "scores": [0.9, 0.6, 0.3],
            "y": [0, 0, 0],
        },
        "C": {
            # Одинадцять оцінок 0.5 підряд: перша з міткою 1, решта десять з міткою 0
            "scores": [0.8] + [0.5] * 11 + [0.2],
            "y": [1, 1] + [0] * 10 + [0],
        },
    }

    summary = {}
    print("\n=== ПУНКТ 3. ПЕРЕВІРКА ПОШУКУ ПОРОГА НА НАБОРАХ A, B, C ===")

    for name, data in datasets.items():
        y_arr = np.array(data["y"], dtype=np.int64)
        s_arr = np.array(data["scores"], dtype=np.float64)

        fast_df, fast_t, fast_c = search_threshold_fast(y_arr, s_arr)
        naive_df, naive_t, naive_c = search_threshold_naive(y_arr, s_arr)

        # Точна перевірка збігу всіх стовпців і обраного порога
        thresholds_match = np.array_equal(
            fast_df["threshold"].to_numpy(), naive_df["threshold"].to_numpy()
        )
        counts_match = all(
            np.array_equal(fast_df[col].to_numpy(), naive_df[col].to_numpy())
            for col in ("TP", "FP", "FN", "TN", "C")
        )
        best_match = (fast_t == naive_t) and (fast_c == naive_c)

        if not (thresholds_match and counts_match and best_match):
            raise AssertionError(
                f"Перевірка на наборі {name} не пройдена! "
                f"fast=(t*={fast_t}, C={fast_c}), naive=(t*={naive_t}, C={naive_c})"
            )

        # Зберігаємо таблицю перевірки у CSV
        fast_df.to_csv(out_dir / f"check_{name}.csv", index=False)

        print(f"\nНабір {name}: OK | t* = {fast_t}, min C = {fast_c}")
        print(fast_df.to_string(index=False))

        summary[name] = {
            "passed": True,
            "best_threshold": "inf" if np.isinf(fast_t) else fast_t,
            "min_cost": fast_c,
            "n_candidates": int(len(fast_df)),
        }

    print("\nУсі три перевірки (A, B, C) пройдено з точним збігом!")
    return summary


def plot_validation_curves(
    y_val: np.ndarray,
    s_val: np.ndarray,
    res_default: dict[str, int | float],
    res_best: dict[str, int | float],
    out_dir: Path,
) -> float:
    """Будує та зберігає PR-криву і ROC-криву на validation, повертає ROC-AUC."""
    # --- 1. Precision-Recall (PR) крива ---
    precisions, recalls, _ = precision_recall_curve(y_val, s_val)
    p_def, r_def = compute_precision_recall(
        int(res_default["TP"]), int(res_default["FP"]), int(res_default["FN"])
    )
    p_best, r_best = compute_precision_recall(
        int(res_best["TP"]), int(res_best["FP"]), int(res_best["FN"])
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, linewidth=2, label="PR-крива (validation)")
    if p_def is not None and r_def is not None:
        plt.scatter(
            [r_def],
            [p_def],
            color="tab:orange",
            s=60,
            zorder=5,
            label=f"t = 0.5 (P={p_def:.3f}, R={r_def:.3f})",
        )
    if p_best is not None and r_best is not None:
        plt.scatter(
            [r_best],
            [p_best],
            color="tab:red",
            s=60,
            zorder=5,
            label=f"t* = {res_best['threshold']:.6f} (P={p_best:.3f}, R={r_best:.3f})",
        )
    plt.xlabel("Повнота (Recall)")
    plt.ylabel("Точність (Precision)")
    plt.title("Крива точність–повнота (PR) на validation")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve_validation.png", dpi=150)
    plt.close()

    # --- 2. ROC-крива та ROC-AUC ---
    fpr, tpr, _ = roc_curve(y_val, s_val)
    roc_auc = float(roc_auc_score(y_val, s_val))

    fpr_def = int(res_default["FP"]) / (int(res_default["FP"]) + int(res_default["TN"]))
    tpr_def = int(res_default["TP"]) / (int(res_default["TP"]) + int(res_default["FN"]))
    fpr_best = int(res_best["FP"]) / (int(res_best["FP"]) + int(res_best["TN"]))
    tpr_best = int(res_best["TP"]) / (int(res_best["TP"]) + int(res_best["FN"]))

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-крива (AUC = {roc_auc:.6f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Випадковий класифікатор")
    plt.scatter([fpr_def], [tpr_def], color="tab:orange", s=60, zorder=5, label="t = 0.5")
    plt.scatter(
        [fpr_best],
        [tpr_best],
        color="tab:red",
        s=60,
        zorder=5,
        label=f"t* = {res_best['threshold']:.6f}",
    )
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("ROC-крива на validation")
    plt.xlim(-0.01, 1.01)
    plt.ylim(0.0, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_validation.png", dpi=150)
    plt.close()

    return roc_auc


def analyze_test_errors(
    y_test: np.ndarray,
    s_test: np.ndarray,
    amount_test: np.ndarray,
    row_ids_test: np.ndarray,
    t_star: float,
    out_dir: Path,
) -> dict[str, object]:
    """Пункт 4: оцінка зафіксованого порога t* на test і розбір помилок FP/FN."""
    test_metrics = evaluate_threshold(y_test, s_test, t_star)
    precision, recall = compute_precision_recall(
        int(test_metrics["TP"]), int(test_metrics["FP"]), int(test_metrics["FN"])
    )

    # Для зручності порівняння також подивимося стандартний поріг 0.5 на test
    test_default = evaluate_threshold(y_test, s_test, 0.5)
    prec_def, rec_def = compute_precision_recall(
        int(test_default["TP"]), int(test_default["FP"]), int(test_default["FN"])
    )

    s_f64 = np.asarray(s_test, dtype=np.float64)
    y_pred = (s_f64 >= t_star).astype(np.int64)

    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    def build_error_df(mask: np.ndarray, error_type: str) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "csv_row": row_ids_test[mask],
                "error_type": error_type,
                "label": y_test[mask],
                "score": s_f64[mask],
                "amount": amount_test[mask],
                "distance_to_threshold": np.abs(s_f64[mask] - t_star),
            }
        )
        # Сортуємо за номером рядка у вихідному CSV (від 0, без заголовка)
        return df.sort_values("csv_row", ascending=True).reset_index(drop=True)

    fp_all = build_error_df(fp_mask, "FP")
    fn_all = build_error_df(fn_mask, "FN")

    top_errors = pd.concat([fp_all.head(5), fn_all.head(5)], ignore_index=True)
    top_errors.to_csv(out_dir / "test_errors_top5.csv", index=False)

    print("\n=== ПУНКТ 4. ТЕСТОВА ОЦІНКА ТА АНАЛІЗ ПОМИЛОК ===")
    print(f"Застосовано зафіксований поріг t* = {t_star:.10f}")
    print(
        f"Test (t*):  TP={test_metrics['TP']}, FP={test_metrics['FP']}, "
        f"FN={test_metrics['FN']}, TN={test_metrics['TN']}, C={test_metrics['C']}"
    )
    print(
        f"            Precision = {precision:.6f}"
        if precision is not None
        else "            Precision = не визначена"
    )
    print(
        f"            Recall    = {recall:.6f}"
        if recall is not None
        else "            Recall    = не визначена"
    )
    print(
        f"Для порівняння на test при t=0.5: TP={test_default['TP']}, FP={test_default['FP']}, "
        f"FN={test_default['FN']}, TN={test_default['TN']}, C={test_default['C']}, "
        f"Precision={prec_def:.6f}, Recall={rec_def:.6f}"
    )

    print("\nПерші 5 FP і перші 5 FN за номером рядка CSV:")
    print(top_errors.to_string(index=False))

    def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
        if len(df) == 0:
            return {"count": 0}
        return {
            "count": int(len(df)),
            "amount_min": float(df["amount"].min()),
            "amount_median": float(df["amount"].median()),
            "amount_mean": float(df["amount"].mean()),
            "amount_max": float(df["amount"].max()),
            "dist_min": float(df["distance_to_threshold"].min()),
            "dist_median": float(df["distance_to_threshold"].median()),
            "dist_mean": float(df["distance_to_threshold"].mean()),
            "dist_max": float(df["distance_to_threshold"].max()),
        }

    fp_top5_stats = summarize_group(fp_all.head(5))
    fn_top5_stats = summarize_group(fn_all.head(5))
    fp_all_stats = summarize_group(fp_all)
    fn_all_stats = summarize_group(fn_all)

    print("\nСтатистика по перших 5 FP:", json.dumps(fp_top5_stats, indent=2))
    print("Статистика по перших 5 FN:", json.dumps(fn_top5_stats, indent=2))
    print("Статистика по ВСІХ FP на test:", json.dumps(fp_all_stats, indent=2))
    print("Статистика по ВСІХ FN на test:", json.dumps(fn_all_stats, indent=2))

    return {
        "t_star": {
            **test_metrics,
            "precision": precision,
            "recall": recall,
        },
        "default_0_5_reference": {
            **test_default,
            "precision": prec_def,
            "recall": rec_def,
        },
        "fp_top5_stats": fp_top5_stats,
        "fn_top5_stats": fn_top5_stats,
        "fp_all_stats": fp_all_stats,
        "fn_all_stats": fn_all_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛР2: аналіз порога, перевірки A/B/C та тестова оцінка зі збережених прогнозів."
    )
    parser.add_argument(
        "--saved",
        type=Path,
        default=Path("results"),
        help="Тека зі збереженими прогнозами (predictions.npz)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Тека для збереження графіків і таблиць (за замовчуванням та сама, що --saved)",
    )
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else args.saved
    out_dir.mkdir(parents=True, exist_ok=True)

    # Завантажуємо збережені прогнози без навчання і без creditcard.csv
    saved = load_predictions(args.saved)
    y_val = saved["validation_y"]
    s_val = saved["validation_scores"]
    y_test = saved["test_y"]
    s_test = saved["test_scores"]
    amount_test = saved["test_amount"]
    row_ids_test = saved["test_row_ids"]

    # === ПУНКТ 2. ВИБІР ПОРОГА НА VALIDATION ===
    print("=== ПУНКТ 2. ВИБІР ПОРОГА НА VALIDATION ===")
    candidates_df, t_star, min_cost = search_threshold_fast(y_val, s_val)
    candidates_df.to_csv(out_dir / "validation_candidates.csv", index=False)

    res_default = evaluate_threshold(y_val, s_val, 0.5)
    res_best = evaluate_threshold(y_val, s_val, t_star)
    res_all_neg = evaluate_threshold(y_val, s_val, float("inf"))

    # Переконуємося, що швидкий пошук і пряма оцінка на t* дають однакову вартість
    if int(res_best["C"]) != min_cost:
        raise AssertionError("Розбіжність між швидким пошуком і evaluate_threshold на t*!")

    val_summary_df = pd.DataFrame(
        [
            {"rule": "Стандартний поріг", **res_default},
            {"rule": "Мінімум вартості (t*)", **res_best},
            {"rule": "Усі негативні", **res_all_neg},
        ]
    )
    val_summary_df.to_csv(out_dir / "validation_comparison.csv", index=False)

    roc_auc = plot_validation_curves(y_val, s_val, res_default, res_best, out_dir)

    print(f"Усього унікальних кандидатів (разом з +inf): {len(candidates_df)}")
    print(f"Validation ROC-AUC = {roc_auc:.6f}")
    print(f"Оптимальний поріг t* = {t_star:.10f} (min C = {min_cost})\n")
    print(val_summary_df.to_string(index=False))

    # === ПУНКТ 3. НЕЗАЛЕЖНІ ПЕРЕВІРКИ НА НАБОРАХ A, B, C ===
    # За умовою: до тестової оцінки переходимо ТІЛЬКИ після проходження всіх трьох перевірок
    checks_summary = verify_on_small_sets(out_dir)

    # === ПУНКТ 4. ТЕСТОВА ОЦІНКА ТА АНАЛІЗ ПОМИЛОК ===
    test_summary = analyze_test_errors(
        y_test, s_test, amount_test, row_ids_test, t_star, out_dir
    )

    # Зберігаємо підсумковий JSON із зафіксованим порогом t* та всіма метриками
    report_data = {
        "validation": {
            "roc_auc": roc_auc,
            "n_candidates": int(len(candidates_df)),
            "t_star": t_star,
            "rules": {
                "default_0_5": res_default,
                "min_cost_t_star": res_best,
                "all_negative_inf": {
                    **res_all_neg,
                    "threshold": "inf",
                },
            },
        },
        "checks": checks_summary,
        "test": test_summary,
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nУсі таблиці, графіки та результати збережено у: {out_dir.resolve()}")


if __name__ == "__main__":
    main()