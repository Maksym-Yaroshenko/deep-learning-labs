"""ЛР3. Команда навчання: поділ, перевірки пунктів 1–2, сім запусків і вибір моделі.

Запуск із теки lab3/:
    uv run python train.py --data data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import (
    BATCH_SIZE_EVAL,
    BATCH_SIZE_TRAIN,
    CLASSES,
    EPOCHS,
    EXPECTED_TOTAL_PARAMS,
    EXPECTED_TRAINABLE_PARAMS,
    FROZEN_EVAL_MODULES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    LR_HEAD,
    LR_LAYER4,
    NUM_WORKERS,
    TOP_LEVEL_MODULES,
    WEIGHT_DECAY,
    IntelSplitDataset,
    build_model_and_optimizer,
    file_sha256,
    get_transforms,
    load_split_rows,
    predict_logits,
    resolve_dataset_dirs,
    select_train_device,
    set_train_mode,
)

# Контрольна таблиця з умови: (кількість файлів, сума числових імен файлів)
EXPECTED_SPLIT_CONTROL = {
    "buildings": {"train": (200, 1_996_761), "validation": (218, 4_837_323), "test": (219, 4_837_559)},
    "forest": {"train": (200, 2_032_625), "validation": (237, 5_276_962), "test": (237, 5_291_326)},
    "glacier": {"train": (200, 2_183_923), "validation": (276, 6_135_652), "test": (277, 6_157_250)},
    "mountain": {"train": (200, 1_950_345), "validation": (262, 5_789_535), "test": (263, 5_849_765)},
    "sea": {"train": (200, 1_987_613), "validation": (255, 5_655_068), "test": (255, 5_668_011)},
    "street": {"train": (200, 1_841_817), "validation": (250, 5_541_845), "test": (251, 5_547_533)},
}

EXPERIMENT_RUNS = [
    ("scratch", 1),
    ("frozen", 1),
    ("frozen", 2),
    ("frozen", 3),
    ("unfreeze", 1),
    ("unfreeze", 2),
    ("unfreeze", 3),
]


def get_git_info() -> tuple[str, bool]:
    """Повертає хеш поточного git-коміту та прапорець незакомічених змін."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit, bool(status)
    except Exception:
        return "unknown", True


def create_and_verify_split(data_dir: Path, split_csv: Path) -> dict[str, object]:
    """Будує split.csv за фіксованим протоколом і звіряє всі 18 контрольних клітинок."""
    train_dir, test_dir = resolve_dataset_dirs(data_dir)
    rows: list[dict[str, str]] = []

    # 1. train: один генератор для всіх класів у фіксованому порядку
    rng_train = np.random.default_rng(0)
    for cls in CLASSES:
        files = sorted(
            p.name
            for p in (train_dir / cls).iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        )
        idx = np.arange(len(files))
        rng_train.shuffle(idx)
        for i in idx[:200]:
            rows.append({"file": files[int(i)], "label": cls, "part": "train"})

    # 2. validation і test: новий генератор для всіх класів seg_test
    rng_eval = np.random.default_rng(0)
    for cls in CLASSES:
        files = sorted(
            p.name
            for p in (test_dir / cls).iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        )
        idx = np.arange(len(files))
        rng_eval.shuffle(idx)
        cut = len(files) // 2
        for i in idx[:cut]:
            rows.append({"file": files[int(i)], "label": cls, "part": "validation"})
        for i in idx[cut:]:
            rows.append({"file": files[int(i)], "label": cls, "part": "test"})

    with split_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "part"])
        writer.writeheader()
        writer.writerows(rows)

    # Перевірка на відсутність дублікатів між частинами
    all_files = [r["file"] for r in rows]
    if len(set(all_files)) != len(all_files):
        raise AssertionError("Знайдено файл, який входить у кілька частин поділу!")

    # Перевірка всіх 18 клітинок контрольної таблиці
    print("=== ПУНКТ 1. КОНТРОЛЬ ПОДІЛУ (split.csv) ===")
    print(f"{'Клас':<12} {'train':>20} {'validation':>20} {'test':>20}")
    control_result = {}
    all_match = True

    for cls in CLASSES:
        control_result[cls] = {}
        cells_str = []
        for part in ("train", "validation", "test"):
            part_files = [r["file"] for r in rows if r["label"] == cls and r["part"] == part]
            count = len(part_files)
            name_sum = sum(int(Path(fname).stem) for fname in part_files)
            exp_count, exp_sum = EXPECTED_SPLIT_CONTROL[cls][part]
            ok = (count == exp_count) and (name_sum == exp_sum)
            if not ok:
                all_match = False
            control_result[cls][part] = {
                "count": count,
                "sum": name_sum,
                "expected_count": exp_count,
                "expected_sum": exp_sum,
                "match": ok,
            }
            mark = "" if ok else " [MISMATCH!]"
            cells_str.append(f"{count}/{name_sum:,}".replace(",", " ") + mark)
        print(f"{cls:<12} {cells_str[0]:>20} {cells_str[1]:>20} {cells_str[2]:>20}")

    if not all_match:
        raise AssertionError("Поділ не збігається з контрольною таблицею умови!")

    split_hash = file_sha256(split_csv)
    print("Усі 18 клітинок контрольної таблиці збіглися точно; дублікатів файлів немає.")
    print(f"SHA-256 split.csv: {split_hash}\n")

    return {"all_match": True, "no_duplicates": True, "sha256": split_hash, "table": control_result}


def save_train_batch_figure(
    train_dataset: IntelSplitDataset, out_path: Path
) -> None:
    """Пункт 1: зберігає 16 перших зображень навчального пакета після перетворень."""
    torch.manual_seed(1)
    np.random.seed(1)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    images, labels = next(iter(loader))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        img = images[i].numpy() * std + mean
        img = np.clip(img, 0.0, 1.0).transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(CLASSES[int(labels[i])], fontsize=10)
        ax.axis("off")

    fig.suptitle("Перші 16 зображень навчального пакета (після аугментацій, обернене нормування)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Рисунок пакета після перетворень збережено: {out_path}")


def check_initial_loss(
    train_dataset: IntelSplitDataset, device: torch.device
) -> dict[str, float]:
    """Пункт 1: обчислює початкову втрату L0 для scratch і frozen (зерно 1) без кроків оптимізатора."""
    print("\n=== ПУНКТ 1. ПОЧАТКОВА ВТРАТА L0 (порівняння з ln(6)) ===")
    ln6 = math.log(len(CLASSES))
    criterion = torch.nn.CrossEntropyLoss()
    results = {"ln_6": ln6}

    for regime in ("scratch", "frozen"):
        model, _ = build_model_and_optimizer(regime, seed=1, device=device)
        loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
            drop_last=False,
            num_workers=NUM_WORKERS,
        )
        x, y = next(iter(loader))
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        set_train_mode(model, regime)
        with torch.no_grad():
            loss_0 = float(criterion(model(x), y).item())
        results[regime] = loss_0
        print(f"  {regime:<10}: L0 = {loss_0:.6f}  (ln(6) = {ln6:.6f}, різниця = {loss_0 - ln6:+.6f})")

    return results


def check_parameters_and_optimizer(device: torch.device) -> dict[str, object]:
    """Пункт 2: перевірка кількості параметрів і груп оптимізатора для всіх 3 режимів."""
    print("\n=== ПУНКТ 2. ПЕРЕВІРКА ПАРАМЕТРІВ І ГРУП ОПТИМІЗАТОРА ===")
    summary = {}

    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer = build_model_and_optimizer(regime, seed=1, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        groups_info = []
        opt_total = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    raise AssertionError(
                        f"У режимі {regime} в оптимізатор потрапив параметр із requires_grad=False!"
                    )
            g_count = sum(p.numel() for p in group["params"])
            opt_total += g_count
            groups_info.append({"num_params": g_count, "lr": float(group["lr"])})

        if total_params != EXPECTED_TOTAL_PARAMS:
            raise AssertionError(f"{regime}: очікувалося {EXPECTED_TOTAL_PARAMS}, отримано {total_params}")
        if trainable_params != EXPECTED_TRAINABLE_PARAMS[regime]:
            raise AssertionError(
                f"{regime}: очікувалося {EXPECTED_TRAINABLE_PARAMS[regime]} trainable, отримано {trainable_params}"
            )
        if opt_total != trainable_params:
            raise AssertionError(
                f"{regime}: в оптимізаторі {opt_total} параметрів, а trainable = {trainable_params}"
            )

        summary[regime] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "optimizer_groups": groups_info,
        }
        groups_str = ", ".join(f"{g['num_params']:,} (lr={g['lr']})" for g in groups_info)
        print(
            f"  {regime:<10} | усього: {total_params:,} | trainable: {trainable_params:,} | "
            f"групи AdamW: {groups_str}"
        )

    print("Усі кількості параметрів збіглися з таблицею режимів; заморожені параметри до оптимізатора не потрапили.")
    return summary


def snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Точна копія всіх параметрів і буферів моделі на CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def inspect_module_changes(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    regime: str,
    use_buggy_train: bool = False,
) -> tuple[list[dict[str, object]], bool]:
    """Перевіряє, чи змінилися параметри і ковзні статистики BN у кожному модулі верхнього рівня."""
    rows = []
    all_match_expected = True

    for mod_name in TOP_LEVEL_MODULES:
        mod = getattr(model, mod_name)

        # 1. Перевірка параметрів модуля
        param_keys = [f"{mod_name}.{p_name}" for p_name, _ in mod.named_parameters()]
        params_changed = any(not torch.equal(before[k], after[k]) for k in param_keys)

        # 2. Перевірка буферів BatchNorm (running_mean, running_var, num_batches_tracked)
        bn_submodules = [
            sub_name
            for sub_name, sub_mod in mod.named_modules()
            if isinstance(sub_mod, torch.nn.modules.batchnorm._BatchNorm)
        ]
        if not bn_submodules:
            bn_status = "немає BN"
        else:
            bn_keys = []
            for sub_name in bn_submodules:
                prefix = f"{mod_name}.{sub_name}" if sub_name else mod_name
                for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                    full_key = f"{prefix}.{stat_name}"
                    if full_key in before:
                        bn_keys.append(full_key)
            bn_changed = any(not torch.equal(before[k], after[k]) for k in bn_keys)
            bn_status = "так" if bn_changed else "ні"

        # 3. Очікуваний стан (для коректного set_train_mode)
        is_frozen_mod = mod_name in FROZEN_EVAL_MODULES[regime]
        expected_params = not is_frozen_mod
        if bn_status == "немає BN":
            expected_bn = "немає BN"
        else:
            expected_bn = "ні" if is_frozen_mod else "так"

        matches_expected = (params_changed == expected_params) and (bn_status == expected_bn)
        if not use_buggy_train and not matches_expected:
            all_match_expected = False

        rows.append(
            {
                "regime": f"{regime} (model.train bug)" if use_buggy_train else regime,
                "module": mod_name,
                "params_changed": "так" if params_changed else "ні",
                "bn_stats_changed": bn_status,
                "matches_expected": "так" if matches_expected else "ні",
            }
        )

    return rows, all_match_expected


def run_five_step_check(
    train_dataset: IntelSplitDataset,
    regime: str,
    device: torch.device,
    use_buggy_train: bool = False,
) -> tuple[list[dict[str, object]], bool, dict[str, float]]:
    """Виконує 5 кроків оптимізатора на новому екземплярі моделі (seed=1) і перевіряє зміни."""
    model, optimizer = build_model_and_optimizer(regime, seed=1, device=device)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    criterion = torch.nn.CrossEntropyLoss()

    before = snapshot_state_dict(model)

    if use_buggy_train:
        model.train()
    else:
        set_train_mode(model, regime)

    step_losses = []
    for step, (x, y) in enumerate(loader):
        if step >= 5:
            break
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.detach().item()))

    after = snapshot_state_dict(model)
    rows, all_match = inspect_module_changes(
        model, before, after, regime=regime, use_buggy_train=use_buggy_train
    )

    extra = {
        "layer1_0_bn1_max_abs_diff_running_mean": float(
            (after["layer1.0.bn1.running_mean"] - before["layer1.0.bn1.running_mean"]).abs().max().item()
        ),
        "layer1_0_bn1_max_abs_diff_running_var": float(
            (after["layer1.0.bn1.running_var"] - before["layer1.0.bn1.running_var"]).abs().max().item()
        ),
        "step_losses": step_losses,
    }
    return rows, all_match, extra


def check_training_regimes(
    train_dataset: IntelSplitDataset, device: torch.device
) -> dict[str, object]:
    """Пункт 2: перевірка змін параметрів/BN у frozen та unfreeze + дослід із помилкою."""
    print("\n=== ПУНКТ 2. ЩО ЗМІНЮЄТЬСЯ ПІД ЧАС НАВЧАННЯ (5 кроків) ===")
    all_tables = {}

    for regime in ("frozen", "unfreeze"):
        rows, ok, extra = run_five_step_check(
            train_dataset, regime=regime, device=device, use_buggy_train=False
        )
        print(f"\nРежим: {regime} (через set_train_mode)")
        print(f"{'Модуль':<10} {'Параметри':<12} {'Ковзні BN':<14} {'Очікувано?':<10}")
        for r in rows:
            print(
                f"{r['module']:<10} {r['params_changed']:<12} "
                f"{r['bn_stats_changed']:<14} {r['matches_expected']:<10}"
            )
        if not ok:
            raise AssertionError(f"Перевірка 5 кроків для режиму {regime} не збіглася з очікуваною!")
        all_tables[regime] = {"rows": rows, "all_match": ok, "extra": extra}

    # Дослід із навмисною помилкою: frozen + звичайний model.train()
    print("\n=== ПУНКТ 2. ДОСЛІД ІЗ НАВМИСНОЮ ПОМИЛКОЮ (frozen + model.train()) ===")
    bug_rows, _, bug_extra = run_five_step_check(
        train_dataset, regime="frozen", device=device, use_buggy_train=True
    )
    print(f"{'Модуль':<10} {'Параметри':<12} {'Ковзні BN':<14} {'Збіг із норм. frozen?':<20}")
    for r in bug_rows:
        print(
            f"{r['module']:<10} {r['params_changed']:<12} "
            f"{r['bn_stats_changed']:<14} {r['matches_expected']:<20}"
        )
    print(
        f"layer1.0.bn1 max |Δ running_mean| = {bug_extra['layer1_0_bn1_max_abs_diff_running_mean']:.6f}"
    )
    print(
        f"layer1.0.bn1 max |Δ running_var|  = {bug_extra['layer1_0_bn1_max_abs_diff_running_var']:.6f}"
    )
    print(
        "Втрата на 5 кроках (model.train bug):",
        [round(v, 4) for v in bug_extra["step_losses"]],
    )

    all_tables["frozen_buggy_train"] = {"rows": bug_rows, "extra": bug_extra}
    print("\nУсі перевірки пунктів 1–2 пройшли успішно! Переходимо до 7 запусків.")
    return all_tables


def train_single_run(
    regime: str,
    seed: int,
    train_dataset: IntelSplitDataset,
    val_dataset: IntelSplitDataset,
    device: torch.device,
    split_sha256: str,
    git_commit: str,
    git_dirty: bool,
    run_dir: Path,
    ckpt_path: Path,
) -> dict[str, object]:
    """Виконує один повний запуск навчання (8 епох), зберігає run.json, val_logits.npy і ваги."""
    print(f"\n--- Запуск: regime={regime}, seed={seed}, device={device} ---")
    model, optimizer = build_model_and_optimizer(regime, seed=seed, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_f1s = []
    last_val_logits = None
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        loss_sum = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_len = y.size(0)
            loss_sum += float(loss.detach().item()) * batch_len
            n_seen += batch_len

        epoch_loss = loss_sum / n_seen
        val_logits, val_targets = predict_logits(model, val_loader, device=device)
        val_preds = np.argmax(val_logits, axis=1)
        val_f1 = float(f1_score(val_targets, val_preds, average="macro"))

        train_losses.append(epoch_loss)
        val_f1s.append(val_f1)
        last_val_logits = val_logits
        print(f"  епоха {epoch}/{EPOCHS} | train_loss={epoch_loss:.6f} | val_macro_f1={val_f1:.6f}")

    elapsed = time.perf_counter() - start_time

    # Зберігаємо логіти validation після 8-ї епохи (1498 x 6)
    run_dir.mkdir(parents=True, exist_ok=True)
    assert last_val_logits is not None and last_val_logits.shape == (len(val_dataset), len(CLASSES))
    np.save(run_dir / "val_logits.npy", last_val_logits)

    # Зберігаємо ваги локально (до вибору найкращої моделі у пункті 5)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(cpu_state, ckpt_path)

    # Повний журнал run.json з усіма налаштуваннями (включно зі значеннями за замовчуванням)
    if regime == "scratch":
        lr_groups = [{"modules": "all", "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY}]
    elif regime == "frozen":
        lr_groups = [{"modules": ["fc"], "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY}]
    else:
        lr_groups = [
            {"modules": ["fc"], "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
            {"modules": ["layer4"], "lr": LR_LAYER4, "weight_decay": WEIGHT_DECAY},
        ]

    run_record = {
        "regime": regime,
        "seed": seed,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "split_csv_sha256": split_sha256,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "device": str(device),
            "dtype": "float32",
        },
        "transforms": {
            "train": {
                "RandomResizedCrop": {"size": IMG_SIZE, "scale": [0.7, 1.0], "ratio": [0.75, 1.3333333333333333]},
                "RandomHorizontalFlip": {"p": 0.5},
                "Normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            },
            "validation_and_test": {
                "Resize": {"size": [IMG_SIZE, IMG_SIZE]},
                "Normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            },
        },
        "model_and_regime": {
            "architecture": "resnet18",
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "num_classes": len(CLASSES),
            "classes": CLASSES,
            "total_params": EXPECTED_TOTAL_PARAMS,
            "trainable_params": EXPECTED_TRAINABLE_PARAMS[regime],
            "eval_modules_during_train": list(FROZEN_EVAL_MODULES[regime]),
            "bn_momentum": 0.1,
            "bn_eps": 1e-5,
            "optimizer": "AdamW",
            "optimizer_groups": lr_groups,
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1e-8,
            "adamw_amsgrad": False,
        },
        "training_protocol": {
            "epochs": EPOCHS,
            "batch_size_train": BATCH_SIZE_TRAIN,
            "batch_size_eval": BATCH_SIZE_EVAL,
            "shuffle_train": True,
            "drop_last": False,
            "num_workers": NUM_WORKERS,
            "steps_per_epoch": math.ceil(len(train_dataset) / BATCH_SIZE_TRAIN),
            "total_steps": math.ceil(len(train_dataset) / BATCH_SIZE_TRAIN) * EPOCHS,
            "loss_fn": "CrossEntropyLoss(reduction='mean')",
            "lr_scheduler": None,
            "grad_clip": None,
            "mixed_precision": False,
            "label_smoothing": 0.0,
            "class_weights": None,
            "early_stopping": False,
        },
        "train_loss_per_epoch": train_losses,
        "val_macro_f1_per_epoch": val_f1s,
        "final_train_loss": train_losses[-1],
        "final_val_macro_f1": val_f1s[-1],
        "train_time_sec": round(elapsed, 2),
    }

    (run_dir / "run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  Готово за {elapsed:.2f} с | Фінальний val macro-F1 = {val_f1s[-1]:.6f}")
    return run_record


def select_and_save_best_model(
    records: list[dict[str, object]],
    ckpt_dir: Path,
    out_model_path: Path,
) -> dict[str, object]:
    """Пункт 5 (частина 1): вибір режиму за середнім F на validation і найкращого зерна."""
    by_regime: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_regime.setdefault(str(rec["regime"]), []).append(rec)

    regime_means = {
        reg: float(np.mean([float(r["final_val_macro_f1"]) for r in recs]))
        for reg, recs in by_regime.items()
    }
    best_regime = max(regime_means, key=lambda r: regime_means[r])
    best_run = max(by_regime[best_regime], key=lambda r: float(r["final_val_macro_f1"]))
    best_seed = int(best_run["seed"])

    ckpt_path = ckpt_dir / f"{best_regime}_seed{best_seed}.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    payload = {
        "state_dict": state_dict,
        "regime": best_regime,
        "seed": best_seed,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
    }
    torch.save(payload, out_model_path)

    print("\n=== ПУНКТ 5. ВИБІР КОНФІГУРАЦІЇ ЗА VALIDATION ===")
    for reg, mean_f1 in regime_means.items():
        print(f"  Середній macro-F1 ({reg:<8}): {mean_f1:.6f}")
    print(
        f"Обрано: regime={best_regime}, seed={best_seed} "
        f"(val macro-F1 = {float(best_run['final_val_macro_f1']):.6f})"
    )
    print(f"Модель збережено у: {out_model_path.resolve()}")

    return {
        "regime_means": regime_means,
        "selected_regime": best_regime,
        "selected_seed": best_seed,
        "selected_val_macro_f1": float(best_run["final_val_macro_f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛР3: поділ, перевірки пунктів 1–2, сім запусків і збереження lab3_model.pt."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Шлях до локальної копії Intel Image Classification (де лежать seg_train і seg_test)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("."),
        help="Тека виходу (за замовчуванням поточна тека lab3/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Пристрій для навчання (за замовчуванням auto: cuda -> mps -> cpu)",
    )
    args = parser.parse_args()

    out_dir = args.out
    runs_dir = out_dir / "runs"
    figures_dir = out_dir / "figures"
    ckpt_dir = out_dir / "local_checkpoints"
    split_csv = out_dir / "split.csv"

    device = select_train_device(args.device)
    git_commit, git_dirty = get_git_info()
    print(f"Пристрій для навчання: {device} | git commit: {git_commit[:8]} (dirty={git_dirty})")

    # 1. Поділ і контроль 18 клітинок
    split_info = create_and_verify_split(args.data, split_csv)

    # Завантажуємо ТІЛЬКИ train і validation (test у train.py не використовується!)
    train_tf, eval_tf = get_transforms()
    train_rows = load_split_rows(split_csv, part="train")
    val_rows = load_split_rows(split_csv, part="validation")
    train_dataset = IntelSplitDataset(train_rows, args.data, train_tf)
    val_dataset = IntelSplitDataset(val_rows, args.data, eval_tf)

    # 2. Перевірки пунктів 1–2
    save_train_batch_figure(train_dataset, figures_dir / "train_batch_sample.png")
    l0_info = check_initial_loss(train_dataset, device=device)
    params_info = check_parameters_and_optimizer(device=device)
    regime_checks_info = check_training_regimes(train_dataset, device=device)

    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "checks_summary.json").write_text(
        json.dumps(
            {
                "split": split_info,
                "initial_loss_L0": l0_info,
                "parameters_and_optimizer": params_info,
                "regime_checks": regime_checks_info,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # 3. Сім запусків за протоколом
    print("\n=== ПУНКТ 3. ОСНОВНИЙ ЕКСПЕРИМЕНТ (7 ЗАПУСКІВ) ===")
    records = []
    for regime, seed in EXPERIMENT_RUNS:
        rec = train_single_run(
            regime=regime,
            seed=seed,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            split_sha256=str(split_info["sha256"]),
            git_commit=git_commit,
            git_dirty=git_dirty,
            run_dir=runs_dir / f"{regime}_seed{seed}",
            ckpt_path=ckpt_dir / f"{regime}_seed{seed}.pt",
        )
        records.append(rec)

    # 4. Вибір найкращої моделі за validation і збереження lab3_model.pt
    select_and_save_best_model(records, ckpt_dir=ckpt_dir, out_model_path=out_dir / "lab3_model.pt")


if __name__ == "__main__":
    main()