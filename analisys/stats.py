import torch
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def analyze_tensor_extended(t: torch.Tensor, fs=1.0):
    """
    Выполняет расширенный статистический анализ тензора PyTorch.
    Эта функция вычисляет различные статистические показатели и спектральный анализ входного тензора,
    включая базовую статистику, гистограмму, спектральную плотность мощности методом Вэлча и топ частотных пиков.
    Параметры
    ----------
    t : torch.Tensor
        Входной тензор PyTorch для анализа
    fs : float, optional
        Частота дискретизации для спектрального анализа, по умолчанию 1.0
    Возвращает
    -------
    dict
        Словарь, содержащий следующие результаты анализа:
        - mean: Среднее значение тензора
        - std: Стандартное отклонение
        - min: Минимальное значение
        - max: Максимальное значение
        - skewness: Показатель асимметрии распределения
        - kurtosis: Показатель эксцесса распределения
        - histogram: Словарь, содержащий границы и значения гистограммы
        - welch: Словарь, содержащий частоты и спектральную плотность мощности по методу Вэлча
        - top_peaks: Список 3 наивысших пиков с их частотами и амплитудами
    Пример
    -------
    >>> tensor = torch.randn(1000)
    >>> results = analyze_tensor_extended(tensor, fs=100)
    >>> print(results['mean'])
    0.023
    """

    # Преобразуем в numpy
    arr = t.flatten().cpu().numpy()
    
    # Базовая статистика
    results = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "skewness": float(skew(arr)),
        "kurtosis": float(kurtosis(arr)),
    }
    
    # Гистограмма
    hist_vals, bin_edges = np.histogram(arr, bins=20)
    results["histogram"] = {
        "bins": bin_edges.tolist(),
        "counts": hist_vals.tolist()
    }
    
    # Метод Вэлча для спектра
    freqs, psd = welch(arr, fs=fs, nperseg=min(len(arr), 256))
    results["welch"] = {
        "freqs": freqs.tolist(),
        "psd": psd.tolist()
    }
    
    # Самые высокие пики
    peak_indices = np.argsort(psd)[-3:][::-1]
    results["top_peaks"] = [
        {"frequency": float(freqs[i]), "amplitude": float(psd[i])}
        for i in peak_indices
    ]
    
    return results
