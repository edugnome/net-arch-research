import torch
import torch.nn as nn
from torch import Tensor
import sympy as sp
import numpy as np
from hessian import HessianCalculator

# Определяем модель персептрона с 5 входами и 1 выходом
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # Определяем линейный слой с 5 входами и 1 выходом
        self.linear = nn.Linear(5, 1)
        # Инициализируем веса и смещение заданными значениями
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]]))
            self.linear.bias = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход модели с сигмоидной активацией.
        """
        z = self.linear(x)
        y = torch.sigmoid(z)
        return y

def test_perceptron_hessian():
    model = Perceptron()
    loss_fn = nn.MSELoss()
    hessian_calculator = HessianCalculator(model, loss_fn)

    x_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    x = torch.tensor([x_values], requires_grad=True)
    target = torch.tensor([[0.0]])

    hessian_inputs = hessian_calculator.hessian_wrt_inputs(x, target)
    hessian_params = hessian_calculator.hessian_wrt_params(x, target)

    # Теперь вычислим аналитические значения с помощью SymPy
    x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5', real=True)
    # Объявляем символы для весов
    w1, w2, w3, w4, w5 = sp.symbols('w1 w2 w3 w4 w5', real=True)
    # Объявляем символ для смещения (биаса)
    b = sp.symbols('b', real=True)
    # Объявляем символ для целевого значения (таргета)
    t = sp.symbols('t', real=True)

    x_sym = sp.Matrix([x1, x2, x3, x4, x5])
    w = sp.Matrix([w1, w2, w3, w4, w5])

    # Вычисляем линейную комбинацию входов и весов с учетом смещения
    z = w.dot(x_sym) + b

    # Определяем сигмоидную функцию активации
    y = 1 / (1 + sp.exp(-z))

    # Определяем функцию потерь (MSE с таргетом t)
    loss = (y - t)**2

    # Вычисляем градиент функции потерь по входам
    inputs = [x1, x2, x3, x4, x5]
    grad_inputs = [sp.diff(loss, xi) for xi in inputs]

    # Вычисляем Гессиан функции потерь по входам (матрица вторых производных)
    hessian_inputs_sym = sp.Matrix([[sp.diff(gi, xj) for xj in inputs] for gi in grad_inputs])

    # Вычисляем градиент функции потерь по параметрам (веса и смещение)
    params = [w1, w2, w3, w4, w5, b]
    grad_params = [sp.diff(loss, pi) for pi in params]

    # Вычисляем Гессиан функции потерь по параметрам
    hessian_params_sym = sp.Matrix([[sp.diff(gi, pj) for pj in params] for gi in grad_params])

    # Подставляем численные значения
    subs_dict = {
        x1: x_values[0],
        x2: x_values[1],
        x3: x_values[2],
        x4: x_values[3],
        x5: x_values[4],
        w1: 0.1,
        w2: -0.2,
        w3: 0.3,
        w4: -0.4,
        w5: 0.5,
        b: 0.1,
        t: 0.0  # Целевое значение
    }

    # Вычисляем численные значения Гессиана по входам
    hessian_inputs_values_sym = hessian_inputs_sym.evalf(subs=subs_dict)
    hessian_inputs_values_sym = np.array(hessian_inputs_values_sym.tolist()).astype(np.float64)

    # Вычисляем численное значение Гессиана по входам из PyTorch
    hessian_inputs_values_torch = hessian_inputs.detach().numpy().reshape(5, 5)

    # Вычисляем разницу между Гессианами по входам
    diff_inputs = hessian_inputs_values_torch - hessian_inputs_values_sym
    norm_diff_inputs = np.linalg.norm(diff_inputs)

    # Вычисляем численные значения Гессиана по параметрам
    hessian_params_values_sym = hessian_params_sym.evalf(subs=subs_dict)
    hessian_params_values_sym = np.array(hessian_params_values_sym.tolist()).astype(np.float64)

    # Вычисляем численное значение Гессиана по параметрам из PyTorch
    hessian_params_values_torch = hessian_params.detach().numpy()
    # Убедимся, что размеры совпадают
    hessian_params_values_torch = hessian_params_values_torch.reshape(len(params), len(params))

    # Вычисляем разницу между Гессианами по параметрам
    diff_params = hessian_params_values_torch - hessian_params_values_sym
    norm_diff_params = np.linalg.norm(diff_params)

    # Выводим результаты
    print("Норма разницы Гессианов по входам:", norm_diff_inputs)
    print("Норма разницы Гессианов по параметрам:", norm_diff_params)

    # Опционально выводим сами матрицы для визуальной проверки
    print("\nГессиан по входам (PyTorch):")
    print(hessian_inputs_values_torch)

    print("\nГессиан по входам (SymPy):")
    print(hessian_inputs_values_sym)

    print("\nРазница Гессианов по входам:")
    print(diff_inputs)

    print("\nГессиан по параметрам (PyTorch):")
    print(hessian_params_values_torch)

    print("\nГессиан по параметрам (SymPy):")
    print(hessian_params_values_sym)

    print("\nРазница Гессианов по параметрам:")
    print(diff_params)

if __name__ == "__main__":
    test_perceptron_hessian()