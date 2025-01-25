import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def spectral_analysis(hessian_matrices, layer_names=None):
    """
    Выполняет спектральный анализ собственных значений матриц Гессе по слоям.

    Параметры:
    ----------
    hessian_matrices : list of np.ndarray
        Список квадратных матриц Гессе, соответствующих каждому слою модели.
    layer_names : list of str, optional
        Названия слоев для визуализации. Если None, будут использоваться индексы.

    Возвращает:
    ----------
    None (рисует графики и выводит ключевую информацию в консоль).
    """
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(hessian_matrices))]

    for idx, (hessian, layer_name) in enumerate(zip(hessian_matrices, layer_names)):
        # Проверка формы матрицы
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError(f"Гессиан слоя '{layer_name}' не является квадратной матрицей.")

        # Вычисление собственных значений
        eigenvalues = np.linalg.eigvalsh(hessian)

        # Основные характеристики спектра
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        mean_eigenvalue = np.mean(eigenvalues)
        std_eigenvalue = np.std(eigenvalues)
        negative_count = np.sum(eigenvalues < 0)

        # Вывод характеристик в консоль
        print(f"=== Спектральный анализ: {layer_name} ===")
        print(f"Минимальное собственное значение: {min_eigenvalue:.6f}")
        print(f"Максимальное собственное значение: {max_eigenvalue:.6f}")
        print(f"Среднее собственное значение: {mean_eigenvalue:.6f}")
        print(f"Стандартное отклонение: {std_eigenvalue:.6f}")
        print(f"Число отрицательных значений: {negative_count}")
        print("===========================================")

        # Построение гистограммы
        plt.figure(figsize=(8, 5))
        plt.hist(eigenvalues, bins=30, alpha=0.7, color="blue", label="Гистограмма")
        plt.title(f"Распределение собственных значений: {layer_name}")
        plt.xlabel("Собственное значение")
        plt.ylabel("Частота")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Построение плотности (если данных достаточно)
        if len(eigenvalues) > 10:
            try:
                # Добавляем небольшую регуляризацию, чтобы избежать сингулярности
                kde = gaussian_kde(eigenvalues + 1e-6)
                x_range = np.linspace(min_eigenvalue, max_eigenvalue, 500)
                plt.figure(figsize=(8, 5))
                plt.plot(x_range, kde(x_range), color="red", label="Плотность KDE")
                plt.title(f"Функция плотности собственных значений: {layer_name}")
                plt.xlabel("Собственное значение")
                plt.ylabel("Плотность")
                plt.legend()
                plt.grid(True)
                plt.show()
            except np.linalg.LinAlgError as e:
                print(f"[Предупреждение] KDE не может быть построена для слоя '{layer_name}': {e}")


def compute_condition_number(hessian_matrices, layer_names=None):
    """
    Вычисляет отношение числа обусловленности (\kappa = \lambda_max / \lambda_min)
    для каждой матрицы Гессе.

    Параметры:
    ----------
    hessian_matrices : list of np.ndarray
        Список квадратных матриц Гессе, соответствующих каждому слою модели.
    layer_names : list of str, optional
        Названия слоев для вывода результата. Если None, будут использоваться индексы.

    Возвращает:
    ----------
    result : dict
        Словарь, где ключ — название слоя, а значение — число обусловленности \kappa.
    """
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(hessian_matrices))]

    result = {}

    for idx, (hessian, layer_name) in enumerate(zip(hessian_matrices, layer_names)):
        # Проверка формы матрицы
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError(f"Гессиан слоя '{layer_name}' не является квадратной матрицей.")
        
        # Вычисление собственных значений
        eigenvalues = np.linalg.eigvalsh(hessian)
        
        # Исключение возможных проблем с нулевыми собственными значениями
        lambda_min = np.min(eigenvalues)
        lambda_max = np.max(eigenvalues)
        
        if lambda_min == 0:
            kappa = np.inf  # Условное число бесконечно большое
        else:
            kappa = lambda_max / lambda_min
        
        # Сохранение результата
        result[layer_name] = kappa
        
        # Вывод промежуточных результатов
        print(f"=== Число обусловленности для {layer_name} ===")
        print(f"λ_min: {lambda_min:.6f}, λ_max: {lambda_max:.6f}, κ: {kappa}")
        print("=========================================")
    
    return result
