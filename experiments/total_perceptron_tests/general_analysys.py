#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа множества датасетов и архитектур нейронных сетей через анализ pickle-дампов,
содержащих гессиан, спектральные характеристики, веса, градиенты и другие параметры.

СТРУКТУРА папок:
/root_folder/
    dataset_1/
        no/
            dump_1.pkl
            dump_2.pkl
            ...
        sure/
            dump_1.pkl
            ...
        huge/
            dump_1.pkl
            ...
    dataset_2/
        no/...
        sure/...
        huge/...
    ...

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

Для каждого датасета и типа архитектуры выполняются:
1) ДИНАМИКА (plot) по всем числовым полям.
2) КОРРЕЛЯЦИИ (heatmap, при малом числе признаков — pairplot).
3) РАСШИРЕННЫЙ АНАЛИЗ (describe, Shapiro, NaN) важных полей.
4) CCA-анализ между метриками качества и параметрами сети.

Результаты сохраняются в JSON-файлы:
- analysis_results_of_{dataset}_{arch_type}.json - для каждого датасета и архитектуры
- 0_final_analysis_results_{arch_type}.json - агрегированные результаты по всем датасетам

Запуск:
    python analyze_dumps.py --root_folder /path/to/datasets --output_folder analysis
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import CCA
from scipy.stats import shapiro


class AnalysisError(Exception):
    """Базовый класс для исключений анализа"""

    pass


class DatasetNotFoundError(AnalysisError):
    """Ошибка при отсутствии папок с датасетами"""

    pass


class DumpReadError(AnalysisError):
    """Ошибка при чтении pickle-файла"""

    pass


class ProcessingError(AnalysisError):
    """Ошибка при обработке данных"""

    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze model dumps (Hessian, spectral, weights, etc.)."
    )
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root path containing dataset folders with pickle dumps",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="analysis",
        help="Output folder for analysis results",
    )
    return parser.parse_args()


def list_pickle_files(folder_path: str):
    """
    Находит все *.pkl и *.pickle в папке и сортирует их лексикографически.
    """
    all_files = os.listdir(folder_path)
    pkl_files = [f for f in all_files if f.lower().endswith((".pkl", ".pickle"))]
    if not pkl_files:
        return []
    pkl_files.sort()
    return [os.path.join(folder_path, fn) for fn in pkl_files]


def load_dump(pickle_path: str) -> dict:
    """
    Загружает один pickle-файл -> dict.
    Raises:
        DumpReadError: если возникла ошибка при чтении файла
    """
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return data
    except (IOError, pickle.UnpicklingError) as e:
        raise DumpReadError(f"Ошибка при чтении {pickle_path}: {str(e)}") from e


def calc_tensor_stats(tensor_data) -> dict:
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


def calc_eigens_stats(eigens_data) -> dict:
    """
    Собственные числа гессиана (или др.), список/массив eigens_data.
    Возвращает {eigens_min, eigens_max, eigens_mean, eigens_sum} либо None.
    """
    if eigens_data is None:
        return None
    if isinstance(eigens_data, dict):
        return {
            "eigens_min": float(0),
            "eigens_max": float(0),
            "eigens_mean": float(0),
            "eigens_sum": float(0),
        }
    # Исправлено: корректно обрабатываем list, np.ndarray, torch.Tensor
    arr = None
    if isinstance(eigens_data, (list, np.ndarray)):
        arr = np.array(eigens_data, dtype=float)
    else:
        try:
            import torch

            if isinstance(eigens_data, torch.Tensor):
                arr = eigens_data.detach().cpu().numpy()
            else:
                arr = np.array(eigens_data, dtype=float)
        except ImportError:
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
    result[f"{prefix}_hist_bins_count"] = len(bins_) if bins_ else 0
    result[f"{prefix}_hist_counts_sum"] = sum(counts_) if counts_ else 0

    # welch
    welch_ = sp_data.get("welch", {})
    freqs_ = welch_.get("freqs", [])
    psd_ = welch_.get("psd", [])
    result[f"{prefix}_welch_freqs_count"] = len(freqs_) if freqs_ else 0
    result[f"{prefix}_welch_psd_sum"] = float(np.sum(psd_)) if len(psd_) > 0 else 0.0

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
        row[f"{layer_name}_hessian_condition"] = layer_data.get(
            "hessian_condition", None
        )

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

        # bias_gradient
        bg_data = layer_data.get("bias_gradient", None)
        bg_stats = calc_tensor_stats(bg_data)
        if bg_stats:
            row[f"{layer_name}_bias_gradient_mean_val"] = bg_stats["mean"]
            row[f"{layer_name}_bias_gradient_std_val"] = bg_stats["std"]
            row[f"{layer_name}_bias_gradient_min_val"] = bg_stats["min"]
            row[f"{layer_name}_bias_gradient_max_val"] = bg_stats["max"]
        else:
            row[f"{layer_name}_bias_gradient_mean_val"] = None
            row[f"{layer_name}_bias_gradient_std_val"] = None
            row[f"{layer_name}_bias_gradient_min_val"] = None
            row[f"{layer_name}_bias_gradient_max_val"] = None

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

    if df.empty:
        print(f"[build_dataframe] Папка '{folder_path}' не содержит подходящих дампов.")
        return df
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
        print(
            "[correlation_analysis] Недостаточно числовых колонок для анализа корреляции."
        )
        return

    corr_df = df[numeric_cols].corr(method="pearson")

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(min(20, 0.5 * len(numeric_cols)), min(20, 0.5 * len(numeric_cols)))
        )
        sns.heatmap(corr_df, annot=False, cmap="RdBu", center=0)
        plt.title("Correlation Matrix (Pearson)")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
    except ImportError:
        print(
            "[correlation_analysis] Внимание: seaborn не установлен, график heatmap не будет построен."
        )

    print("\n=== Корреляционная матрица (округлённая) ===")
    print(corr_df.round(3))

    # pairplot
    if len(numeric_cols) <= 8:
        subdf = df[numeric_cols].dropna()
        if subdf.shape[0] > 1:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt

                sns.pairplot(subdf)
                plt.suptitle("Pairplot of numeric features", y=1.02)
                pairplot_path = os.path.join(output_dir, "pairplot.png")
                plt.savefig(pairplot_path)
                plt.close()
            except ImportError:
                print(
                    "[correlation_analysis] Внимание: seaborn не установлен, pairplot не будет построен."
                )


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
        print(
            "[advanced_statistical_analysis] Нет подходящих колонок для расширенного анализа."
        )
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


def find_dataset_folders(root_folder: str) -> dict:
    """
    Рекурсивно находит все папки с датасетами в формате:
    /root_folder/dataset_name/{no,sure,huge}/*.pkl

    Возвращает словарь:
    {
        "dataset_name": {
            "no": "/path/to/no",
            "sure": "/path/to/sure",
            "huge": "/path/to/huge"
        },
        ...
    }
    """
    result = {}
    arch_types = ["no", "sure", "huge"]

    for item in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, item)
        if not os.path.isdir(dataset_path):
            continue

        arch_paths = {}
        for arch_type in arch_types:
            arch_dir = os.path.join(dataset_path, arch_type)
            if os.path.isdir(arch_dir) and any(
                f.endswith(".pkl") for f in os.listdir(arch_dir)
            ):
                arch_paths[arch_type] = arch_dir

        if arch_paths:
            result[item] = arch_paths

    return result


def aggregate_cca_results(cca_list: list) -> dict:
    """
    Агрегирует результаты CCA анализа по всем датасетам.
    Анализирует паттерны в весах и вычисляет общие тенденции.
    """
    if not cca_list:
        return None

    # Собираем все веса и характеристики
    x_weights_all = []
    y_weights_all = []
    scores = []
    features_sets = []

    for cca_data in cca_list:
        if cca_data.get("x_weights") and cca_data.get("y_weights"):
            x_weights_all.append(np.array(cca_data["x_weights"]))
            y_weights_all.append(np.array(cca_data["y_weights"]))
            scores.append(cca_data.get("score", 0))
            features_sets.append(cca_data.get("features", {}))

    if not x_weights_all:
        return None

    # Преобразуем в numpy массивы
    x_weights_all = np.array(x_weights_all)
    y_weights_all = np.array(y_weights_all)

    # Анализируем веса
    x_weights_stats = {
        "mean": np.mean(x_weights_all, axis=0).tolist(),
        "median": np.median(x_weights_all, axis=0).tolist(),
        "std": np.std(x_weights_all, axis=0).tolist(),
        "var": np.var(x_weights_all, axis=0).tolist(),
    }

    y_weights_stats = {
        "mean": np.mean(y_weights_all, axis=0).tolist(),
        "median": np.median(y_weights_all, axis=0).tolist(),
        "std": np.std(y_weights_all, axis=0).tolist(),
        "var": np.var(y_weights_all, axis=0).tolist(),
    }

    # Анализируем scores
    score_stats = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }

    # Анализируем частоту появления признаков
    feature_frequency = {"groupA": {}, "groupB": {}}
    for feat_set in features_sets:
        for group in ["groupA", "groupB"]:
            for feature in feat_set.get(group, []):
                feature_frequency[group][feature] = (
                    feature_frequency[group].get(feature, 0) + 1
                )

    # Нормализуем частоты
    total_sets = len(features_sets)
    for group in feature_frequency:
        for feature in feature_frequency[group]:
            feature_frequency[group][feature] /= total_sets

    return {
        "aggregated_x_weights": x_weights_stats,
        "aggregated_y_weights": y_weights_stats,
        "score_statistics": score_stats,
        "feature_importance": feature_frequency,
        "total_samples": total_sets,
    }


def aggregate_metrics(metrics_list: list) -> dict:
    """
    Агрегирует числовые метрики из списка словарей.
    Возвращает расширенный набор статистик.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

    result = {}

    # Собираем CCA результаты отдельно
    cca_results = []
    for metrics in metrics_list:
        if "cca_results" in metrics:
            cca_results.append(metrics["cca_results"])

    # Агрегируем CCA если есть результаты
    if cca_results:
        result["aggregated_cca"] = aggregate_cca_results(cca_results)

    # Преобразуем список словарей в DataFrame для числовых метрик
    df = pd.DataFrame(metrics_list)

    # Удаляем нечисловые колонки и мета-информацию
    cols_to_drop = ["dataset", "cca_results", "_meta"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            series = df[column].dropna()
            if len(series) > 0:
                # Расширенная статистика
                result[column] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "var": float(series.var()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skew": float(stats.skew(series, nan_policy="omit")),
                    "kurtosis": float(stats.kurtosis(series, nan_policy="omit")),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                    "mode": (
                        float(series.mode().iloc[0])
                        if not series.mode().empty
                        else None
                    ),
                    "non_na_count": int(series.count()),
                    "na_count": int(df[column].isna().sum()),
                }

                # Проверка на нормальность распределения
                if len(series) >= 3:  # минимум для теста Шапиро-Уилка
                    stat, p_value = stats.shapiro(series)
                    result[column]["normality_test"] = {
                        "shapiro_stat": float(stat),
                        "shapiro_p_value": float(p_value),
                        "is_normal": p_value > 0.05,
                    }

    # Добавляем мета-информацию об агрегации
    result["_aggregation_meta"] = {
        "total_datasets": len(metrics_list),
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics_aggregated": list(result.keys()),
    }

    return result


def save_analysis_results(results: dict, filepath: str):
    """
    Сохраняет результаты анализа в JSON файл
    """
    import json

    # Конвертируем numpy типы в обычные Python типы
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results = convert_to_python_types(results)
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_dataset_folder(
    folder_path: str, dataset_name: str, arch_type: str, output_folder: str
) -> dict:
    """
    Обрабатывает одну папку с дампами и возвращает результаты анализа

    Raises:
        ProcessingError: если возникла ошибка при обработке данных
    """
    try:
        print(f"\nОбработка датасета {dataset_name} (архитектура: {arch_type})")

        # Проверяем наличие pickle файлов
        pkl_files = list_pickle_files(folder_path)
        if not pkl_files or len(pkl_files) == 0:
            print(f"Предупреждение: в папке {folder_path} нет pickle-файлов")

        # Собираем данные
        df = build_dataframe(folder_path)
        if df.empty:
            raise ProcessingError(
                f"DataFrame пустой после обработки файлов из {folder_path}"
            )

        # Сохраняем графики в отдельную папку для этого датасета
        try:
            plots_dir = os.path.join(output_folder, f"plots_{dataset_name}_{arch_type}")
            plot_all_parameters_dynamics(df, plots_dir)
        except Exception as e:
            print(f"Предупреждение: не удалось создать графики динамики: {e}")

        # Выполняем корреляционный анализ
        try:
            corr_dir = os.path.join(
                output_folder, f"correlations_{dataset_name}_{arch_type}"
            )
            correlation_analysis(df, corr_dir)
        except Exception as e:
            print(f"Предупреждение: не удалось выполнить корреляционный анализ: {e}")

        # Расширенный статистический анализ (describe, Shapiro, NaN)
        try:
            print("\n[Расширенный статистический анализ]")
            advanced_statistical_analysis(df)
        except Exception as e:
            print(
                f"Предупреждение: не удалось выполнить расширенный статистический анализ: {e}"
            )

        # Собираем все числовые метрики
        metrics = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ProcessingError("Не найдено числовых колонок в данных")

        for column in numeric_cols:
            if column != "iteration":
                series = df[column]
                if series.notna().sum() > 0:  # если есть не-NA значения
                    metrics[column] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "non_na_count": int(series.notna().sum()),
                        "na_count": int(series.isna().sum()),
                    }

        # Добавляем результаты CCA
        try:
            cca_results = perform_cca_analysis(df)
            if cca_results:
                metrics["cca_results"] = cca_results
        except Exception as e:
            print(f"Предупреждение: не удалось выполнить CCA анализ: {e}")

        if not metrics:
            raise ProcessingError("Не удалось собрать ни одной метрики")

        metrics["_meta"] = {
            "dataset": dataset_name,
            "architecture": arch_type,
            "num_samples": len(df),
            "num_features": len(numeric_cols),
            "processed_files": len(pkl_files),
        }

        return metrics

    except Exception as e:
        raise ProcessingError(
            f"Ошибка при обработке {dataset_name} ({arch_type}): {str(e)}"
        ) from e


def perform_cca_analysis(df: pd.DataFrame) -> dict:
    """
    Выполняет CCA анализ и возвращает результаты в структурированном виде
    """
    groupA = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    groupA = [g for g in groupA if g in df.columns]

    groupB = []
    for c in df.columns:
        if c.endswith("_mean_val") or c.endswith("_eigens_mean"):
            groupB.append(c)

    if len(groupA) < 2 or len(groupB) < 2:
        return None

    subdf = df[groupA + groupB].dropna()
    if subdf.shape[0] < 6:
        return None

    X = subdf[groupA].values
    Y = subdf[groupB].values
    ncomp = min(2, len(groupA), len(groupB))

    try:
        cca = CCA(n_components=ncomp)
        cca.fit(X, Y)

        return {
            "x_weights": cca.x_weights_.tolist(),
            "y_weights": cca.y_weights_.tolist(),
            "score": float(cca.score(X, Y)),
            "features": {"groupA": groupA, "groupB": groupB},
        }
    except:
        return None


def main():
    try:
        args = parse_arguments()
        root_folder = args.root_folder
        output_folder = args.output_folder

        if not os.path.isdir(root_folder):
            raise DatasetNotFoundError(f"Папка {root_folder} не существует")

        # Создаем выходную директорию
        os.makedirs(output_folder, exist_ok=True)

        # Находим все папки с датасетами
        dataset_folders = find_dataset_folders(root_folder)
        if not dataset_folders:
            raise DatasetNotFoundError(f"Не найдено папок с датасетами в {root_folder}")

        print(f"Найдено датасетов: {len(dataset_folders)}")
        for dataset, archs in dataset_folders.items():
            print(f"- {dataset}: {', '.join(archs.keys())}")

        # Для хранения агрегированных результатов по архитектурам
        arch_results = {"no": [], "sure": [], "huge": []}
        processed_count = {"success": 0, "error": 0}

        # Обрабатываем каждый датасет
        for dataset_name, arch_paths in dataset_folders.items():
            for arch_type, folder_path in arch_paths.items():
                print(f"\n{'='*50}")
                print(f"Обработка: {dataset_name} ({arch_type})")
                print(f"{'='*50}")

                try:
                    # Анализируем датасет
                    results = process_dataset_folder(
                        folder_path, dataset_name, arch_type, output_folder
                    )

                    # Сохраняем результаты для этого датасета
                    output_file = os.path.join(
                        output_folder,
                        f"analysis_results_of_{dataset_name}_{arch_type}.json",
                    )
                    save_analysis_results(results, output_file)
                    print(f"✓ Результаты сохранены в {output_file}")

                    # Добавляем в общие результаты
                    arch_results[arch_type].append({"dataset": dataset_name, **results})

                    processed_count["success"] += 1

                except (ProcessingError, DumpReadError) as e:
                    print(f"✗ {str(e)}")
                    processed_count["error"] += 1
                except Exception as e:
                    print(f"✗ Неожиданная ошибка: {str(e)}")
                    processed_count["error"] += 1

        # Агрегируем и сохраняем общие результаты по архитектурам
        successful_aggregations = 0
        for arch_type, results_list in arch_results.items():
            if results_list:
                try:
                    aggregated = aggregate_metrics(results_list)
                    output_file = os.path.join(
                        output_folder, f"0_final_analysis_results_{arch_type}.json"
                    )
                    save_analysis_results(aggregated, output_file)
                    print(
                        f"\n✓ Итоговые результаты для архитектуры {arch_type} сохранены в {output_file}"
                    )
                    successful_aggregations += 1
                except Exception as e:
                    print(
                        f"\n✗ Не удалось создать итоговый анализ для {arch_type}: {e}"
                    )

        print("\n=== Итоги выполнения ===")
        print(f"Успешно обработано датасетов: {processed_count['success']}")
        print(f"Ошибок обработки: {processed_count['error']}")
        print(f"Создано итоговых отчетов: {successful_aggregations}")

        if processed_count["error"] > 0:
            print("\n⚠ Были ошибки при обработке некоторых датасетов")
        else:
            print("\n✓ Анализ успешно завершен")

    except DatasetNotFoundError as e:
        print(f"\n✗ Ошибка: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
