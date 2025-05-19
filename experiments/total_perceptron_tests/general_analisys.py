#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа серии дампов (pickle-файлов) нейронной сети, включающих гессиан, спектральные характеристики,
а также веса, градиенты, их собственные числа и т.д.

СТРУКТУРА дампа (пример):
{
  "layer.0": {
    "weights":                ...,
    "weights_spectral":       {...},
    "gradient":               ...,
    "gradient_spectral":      {...},
    "bias":                   ...,
    "bias_spectral":          {...},
    "bias_gradient":          ...,
    "bias_gradient_spectral": {...},
    "hessian":                ...,
    "hessian_spectral":       {...},
    "hessian_eigens":         [...],
    "hessian_eigens_spectral": {...},  # поля mean, std, min, max, histogram, welch, top_peaks...
    "hessian_rank":           ...,
    "hessian_condition":      ...      # Отношение max/min собственного числа
  },
  "layer.1": {...},
  ...
  "iteration": <int>,
  "scores": {
    "Accuracy": ...,
    "Precision": ...,
    "Recall": ...,
    "F1": ...,
    "AUC": ...,
    "train_loss": ...
  }
}

Выполняемые шаги:
1) ДИНАМИКА (plot) по всем числовым полям (включая hessian_condition, rank, hessian_eigens_spectral, weights и т.д.).
2) КОРРЕЛЯЦИИ (heatmap, при малом числе признаков — pairplot).
3) РАСШИРЕННЫЙ АНАЛИЗ (describe, Shapiro, NaN) по интересующим полям (hessian_rank, hessian_condition,
   hessian_eigens_min/max/mean/sum, weights_mean_val, spectral поля и т.д.).
4) Опциональный CCA (каноническая корреляция) между метриками качества и некоторыми параметрами.

Запуск:
  python analyze_dumps.py --dumps_folder /path/to/pickle/dumps
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_decomposition import CCA
from scipy.stats import shapiro


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze model dumps (Hessian, spectral, weights, etc.)."
    )
    parser.add_argument(
        "--dumps_folder",
        type=str,
        required=True,
        help="Path to folder containing *.pickle dumps with the specified structure.",
    )
    return parser.parse_args()


def list_pickle_files(folder_path: str):
    """
    Находит все *.pickle в папке и сортирует их лексикографически.
    """
    all_files = os.listdir(folder_path)
    pkl_files = [f for f in all_files if f.lower().endswith(".pickle")]
    pkl_files.sort()
    return [os.path.join(folder_path, fn) for fn in pkl_files]


def load_dump(pickle_path: str) -> dict:
    """
    Загружает один pickle-файл -> dict.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data


def calc_tensor_stats(tensor_data) -> dict | None:
    """
    Вычисляет mean, std, min, max для массива numpy (или torch.Tensor).
    Возвращает словарь {mean, std, min, max}, либо None, если нет данных.
    """
    if tensor_data is None:
        return None
    # Преобразуем к np.array при необходимости
    arr = None
    # Если уже np.ndarray
    if isinstance(tensor_data, np.ndarray):
        arr = tensor_data
    else:
        # torch.Tensor ?
        try:
            import torch

            if isinstance(tensor_data, torch.Tensor):
                arr = tensor_data.detach().cpu().numpy()
        except ImportError:
            pass
        # Или просто np.array(...)
        if arr is None:
            arr = np.array(tensor_data)

    if arr.size == 0:
        return None

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def calc_eigens_stats(eigens_data) -> dict | None:
    """
    Собственные числа гессиана (или др.), список/массив eigens_data.
    Возвращает {eigens_min, eigens_max, eigens_mean, eigens_sum} либо None.
    """
    if eigens_data is None:
        return None
    arr = np.array(eigens_data, dtype=float)
    if arr.size == 0:
        return None
    return {
        "eigens_min": float(np.min(arr)),
        "eigens_max": float(np.max(arr)),
        "eigens_mean": float(np.mean(arr)),
        "eigens_sum": float(np.sum(arr)),
    }


def extract_spectral_info(layer_dict: dict, field_name: str, prefix: str) -> dict:
    """
    Извлекаем спектральные поля (mean, std, min, max, skewness, kurtosis, histogram, welch, top_peaks)
    из layer_dict[field_name], напр. "hessian_eigens_spectral".
    Дополнительно считаем:
     - hist_bins_count, hist_counts_sum,
     - welch_freqs_count, welch_psd_sum,
     - top_peaks_count
    """
    result = {}
    sp_data = layer_dict.get(field_name, {})
    if not sp_data:
        # пусто -> всё None
        for basek in ["mean", "std", "min", "max", "skewness", "kurtosis"]:
            result[f"{prefix}_{basek}"] = None
        result[f"{prefix}_hist_bins_count"] = None
        result[f"{prefix}_hist_counts_sum"] = None
        result[f"{prefix}_welch_freqs_count"] = None
        result[f"{prefix}_welch_psd_sum"] = None
        result[f"{prefix}_top_peaks_count"] = None
        return result

    # базовые
    for basek in ["mean", "std", "min", "max", "skewness", "kurtosis"]:
        result[f"{prefix}_{basek}"] = sp_data.get(basek, None)

    # histogram
    hist_ = sp_data.get("histogram", {})
    bins_ = hist_.get("bins", [])
    counts_ = hist_.get("counts", [])
    result[f"{prefix}_hist_bins_count"] = len(bins_) if bins_ else None
    result[f"{prefix}_hist_counts_sum"] = sum(counts_) if counts_ else None

    # welch
    welch_ = sp_data.get("welch", {})
    freqs_ = welch_.get("freqs", [])
    psd_ = welch_.get("psd", [])
    result[f"{prefix}_welch_freqs_count"] = len(freqs_) if freqs_ else None
    result[f"{prefix}_welch_psd_sum"] = float(np.sum(psd_)) if len(psd_) > 0 else None

    # top_peaks
    top_peaks_ = sp_data.get("top_peaks", [])
    result[f"{prefix}_top_peaks_count"] = len(top_peaks_) if top_peaks_ else 0

    return result


def extract_fields_one_dump(dump_dict: dict) -> dict:
    """
    Извлекает из одного дампа все нужные поля (weights, gradient, bias, hessian_eigens,
    hessian_eigens_spectral, hessian_condition и т.д.)
    Возвращает плоский dict (ключ -> значение).
    """
    row = {}
    row["iteration"] = dump_dict.get("iteration", None)

    # Метрики
    scores = dump_dict.get("scores", {})
    for m in ["Accuracy", "Precision", "Recall", "F1", "AUC", "train_loss"]:
        row[m] = scores.get(m, None)

    # Слои
    for key, layer_data in dump_dict.items():
        if not key.startswith("layer."):
            continue
        layer_name = key.replace(".", "_")  # "layer.0" -> "layer_0"

        # hessian_rank, hessian_condition
        row[f"{layer_name}_hessian_rank"] = layer_data.get("hessian_rank", None)
        row[f"{layer_name}_hessian_condition"] = layer_data.get("hessian_condition", None)

        # hessian_eigens -> min, max, mean, sum
        eigens_data = layer_data.get("hessian_eigens", None)
        eig_stats = calc_eigens_stats(eigens_data)
        if eig_stats:
            for st_key, st_val in eig_stats.items():
                row[f"{layer_name}_{st_key}"] = st_val
        else:
            row[f"{layer_name}_eigens_min"] = None
            row[f"{layer_name}_eigens_max"] = None
            row[f"{layer_name}_eigens_mean"] = None
            row[f"{layer_name}_eigens_sum"] = None

        # weights -> mean,std,min,max
        w_data = layer_data.get("weights", None)
        w_stats = calc_tensor_stats(w_data)
        if w_stats:
            row[f"{layer_name}_weights_mean_val"] = w_stats["mean"]
            row[f"{layer_name}_weights_std_val"] = w_stats["std"]
            row[f"{layer_name}_weights_min_val"] = w_stats["min"]
            row[f"{layer_name}_weights_max_val"] = w_stats["max"]
        else:
            row[f"{layer_name}_weights_mean_val"] = None
            row[f"{layer_name}_weights_std_val"] = None
            row[f"{layer_name}_weights_min_val"] = None
            row[f"{layer_name}_weights_max_val"] = None

        # gradient
        g_data = layer_data.get("gradient", None)
        g_stats = calc_tensor_stats(g_data)
        if g_stats:
            row[f"{layer_name}_gradient_mean_val"] = g_stats["mean"]
            row[f"{layer_name}_gradient_std_val"] = g_stats["std"]
            row[f"{layer_name}_gradient_min_val"] = g_stats["min"]
            row[f"{layer_name}_gradient_max_val"] = g_stats["max"]
        else:
            row[f"{layer_name}_gradient_mean_val"] = None
            row[f"{layer_name}_gradient_std_val"] = None
            row[f"{layer_name}_gradient_min_val"] = None
            row[f"{layer_name}_gradient_max_val"] = None

        # bias
        b_data = layer_data.get("bias", None)
        b_stats = calc_tensor_stats(b_data)
        if b_stats:
            row[f"{layer_name}_bias_mean_val"] = b_stats["mean"]
            row[f"{layer_name}_bias_std_val"] = b_stats["std"]
            row[f"{layer_name}_bias_min_val"] = b_stats["min"]
            row[f"{layer_name}_bias_max_val"] = b_stats["max"]
        else:
            row[f"{layer_name}_bias_mean_val"] = None
            row[f"{layer_name}_bias_std_val"] = None
            row[f"{layer_name}_bias_min_val"] = None
            row[f"{layer_name}_bias_max_val"] = None

        # Спектральные поля (включая hessian_eigens_spectral)
        sp_fields = [
            "weights_spectral",
            "gradient_spectral",
            "bias_spectral",
            "bias_gradient_spectral",
            "hessian_spectral",
            "hessian_eigens_spectral",
        ]
        for sf in sp_fields:
            prefix = f"{layer_name}_{sf}"
            sp_dict = extract_spectral_info(layer_data, sf, prefix)
            row.update(sp_dict)

    return row


def build_dataframe(folder_path: str) -> pd.DataFrame:
    """
    Считывает все дампы в заданной папке, сортирует, формирует DataFrame.
    """
    pkl_list = list_pickle_files(folder_path)
    records = []
    for pf in pkl_list:
        dmp = load_dump(pf)
        row = extract_fields_one_dump(dmp)
        records.append(row)
    df = pd.DataFrame(records)
    df.sort_values("iteration", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_all_parameters_dynamics(df: pd.DataFrame, output_dir: str = "plots_dynamics"):
    """
    Пошаговая визуализация ВСЕХ числовых полей (кроме iteration).
    Группируем по 6 столбцов на один график.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "iteration" in numeric_cols:
        numeric_cols.remove("iteration")

    numeric_cols.sort()
    chunk_size = 6

    def chunkify(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    splitted = list(chunkify(numeric_cols, chunk_size))
    for idx, group in enumerate(splitted, start=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in group:
            # если все NaN — пропустим
            if df[col].notna().sum() == 0:
                continue
            ax.plot(df["iteration"], df[col], label=col)
        ax.set_xlabel("Iteration")
        ax.set_title(f"Dynamics chunk {idx}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        outpath = os.path.join(output_dir, f"dynamics_{idx}.png")
        plt.savefig(outpath)
        plt.close(fig)

    print(f"[plot_all_parameters_dynamics] Графики сохранены в папке '{output_dir}'.")


def correlation_analysis(df: pd.DataFrame, output_dir: str = "plots_correlations"):
    """
    Считает корреляцию (Pearson) для всех числовых полей, строит heatmap,
    а при <=8 столбцах делает pairplot.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "iteration" in numeric_cols:
        numeric_cols.remove("iteration")

    if len(numeric_cols) < 2:
        print("[correlation_analysis] Недостаточно числовых колонок для анализа корреляции.")
        return

    corr_df = df[numeric_cols].corr(method="pearson")

    plt.figure(figsize=(min(20, 0.5 * len(numeric_cols)), min(20, 0.5 * len(numeric_cols))))
    sns.heatmap(corr_df, annot=False, cmap="RdBu", center=0)
    plt.title("Correlation Matrix (Pearson)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    print("\n=== Корреляционная матрица (округлённая) ===")
    print(corr_df.round(3))

    # pairplot
    if len(numeric_cols) <= 8:
        subdf = df[numeric_cols].dropna()
        if subdf.shape[0] > 1:
            sns.pairplot(subdf)
            plt.suptitle("Pairplot of numeric features", y=1.02)
            pairplot_path = os.path.join(output_dir, "pairplot.png")
            plt.savefig(pairplot_path)
            plt.close()


def advanced_statistical_analysis(df: pd.DataFrame):
    """
    Анализ интересных полей:
      - hessian_rank, hessian_condition
      - hessian_eigens_(min/max/mean/sum)
      - weights_mean_val / gradient_mean_val / bias_mean_val
      - spectral поля (..._spectral...)
    Вывод: describe, Shapiro, пропуски.
    """
    # Поиск колонок
    interesting_keys = []
    for c in df.columns:
        if any(
            sub in c
            for sub in [
                "hessian_rank",
                "hessian_condition",
                "eigens_min",
                "eigens_max",
                "eigens_mean",
                "eigens_sum",
                "weights_mean_val",
                "gradient_mean_val",
                "bias_mean_val",
                "_spectral",
            ]
        ):
            interesting_keys.append(c)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    interesting_keys = list(set(interesting_keys).intersection(numeric_cols))
    interesting_keys.sort()
    if not interesting_keys:
        print("[advanced_statistical_analysis] Нет подходящих колонок для расширенного анализа.")
        return

    print("\n=== Расширенный анализ hessian, eigens, weights и spectral-полей ===")
    df_sel = df[interesting_keys].copy()

    # describe
    desc = df_sel.describe().T
    print("\n--- describe() по этим полям ---")
    print(desc)

    # Shapiro на первом поле, содержащем "_mean"
    test_candidates = [c for c in interesting_keys if "_mean" in c]
    if test_candidates:
        tcol = test_candidates[0]
        series_ = df_sel[tcol].dropna()
        if len(series_) >= 3:
            stat, pval = shapiro(series_)
            print(f"\n[Shapiro] '{tcol}': stat={stat:.4f}, p-value={pval:.4f}")
            if pval < 0.05:
                print(" => Распределение, скорее всего, не нормальное (p<0.05).")
            else:
                print(" => Нет оснований отвергать нормальность (p>=0.05).")
        else:
            print(f"[Shapiro] Недостаточно данных для колонки '{tcol}'.")
    else:
        print("\n[Shapiro] Нет колонок, содержащих '_mean'.")

    # Пропуски
    nan_stats = df_sel.isna().sum()
    print("\n--- Количество NaN в интересующих полях ---")
    print(nan_stats)


def optional_cca(df: pd.DataFrame):
    """
    Пример CCA:
     - Группа A: метрики качества (Accuracy, Precision, Recall, F1, AUC)
     - Группа B: всё, что имеет '_mean_val' (weights/gradient/bias) и 'eigens_mean'
       (т.е. среднее собственных чисел гессиана).
    """
    groupA = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    groupA = [g for g in groupA if g in df.columns]

    groupB = []
    for c in df.columns:
        if c.endswith("_mean_val") or c.endswith("_eigens_mean"):
            groupB.append(c)

    if len(groupA) < 2 or len(groupB) < 2:
        print("[CCA] Недостаточно признаков в A или B для канонического анализа. Пропускаем.")
        return

    subdf = df[groupA + groupB].dropna()
    if subdf.shape[0] < 6:
        print("[CCA] Слишком мало строк после удаления NaN.")
        return

    X = subdf[groupA].values
    Y = subdf[groupB].values
    ncomp = min(2, len(groupA), len(groupB))
    cca = CCA(n_components=ncomp)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)

    print("\n=== CCA Результаты ===")
    print("GroupA:", groupA)
    print("GroupB:", groupB)
    print("X_weights:\n", cca.x_weights_)
    print("Y_weights:\n", cca.y_weights_)

    # scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(X_c[:, 0], Y_c[:, 0], alpha=0.7)
    plt.xlabel("CCA comp #1 (X)")
    plt.ylabel("CCA comp #1 (Y)")
    plt.title("Canonical Correlation Analysis: 1st component")
    plt.grid(True)
    plt.show()


def main():
    args = parse_arguments()
    folder = args.dumps_folder

    # 1) Формируем DataFrame
    df = build_dataframe(folder)
    print(f"DataFrame shape = {df.shape}")
    print("Колонки DataFrame:", df.columns.tolist())

    # 2) ДИНАМИКА ВСЕХ ЧИСЛОВЫХ ПАРАМЕТРОВ
    plot_all_parameters_dynamics(df)

    # 3) КОРРЕЛЯЦИИ
    correlation_analysis(df)

    # 4) РАСШИРЕННЫЙ АНАЛИЗ (hessian_rank, hessian_condition, eigens, weights, spectral)
    advanced_statistical_analysis(df)

    # 5) (Опционально) CCA
    optional_cca(df)

    print("\n[Done] Скрипт завершил работу.")


if __name__ == "__main__":
    main()
