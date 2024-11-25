import torch
import torch.nn as nn
from torch import Tensor
from hessian import HessianCalculator

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Вычисляем y = a * sin(exp(x))
        """
        return self.a * torch.sin(torch.exp(x))

def test_hessian_calculator():
    model = TestModel()
    loss_fn = nn.MSELoss()
    hessian_calculator = HessianCalculator(model, loss_fn)

    x = torch.tensor([0.5], requires_grad=True)
    target = torch.tensor([0.0])

    hessian_inputs = hessian_calculator.hessian_wrt_inputs(x, target)
    hessian_params = hessian_calculator.hessian_wrt_params(x, target)

    # Аналитическое вычисление второй производной по x
    with torch.no_grad():
        exp_x = torch.exp(x)
        sin_exp_x = torch.sin(exp_x)
        cos_exp_x = torch.cos(exp_x)
        a = model.a
        f = a * sin_exp_x
        # Первая производная f' по x
        f_prime = a * cos_exp_x * exp_x
        # Вторая производная f'' по x
        f_double_prime = a * (-sin_exp_x * exp_x**2 + cos_exp_x * exp_x)
        # Первая производная потери по f
        loss_df = 2 * f  # Так как loss = f^2, то d(loss)/df = 2f
        # Первая производная потери по x
        loss_dx = loss_df * f_prime
        # Вторая производная потери по x
        loss_dxx = 2 * (f_prime * f_prime + f * f_double_prime)

    print("Гессиан по входам (численный):", hessian_inputs.item())
    print("Гессиан по входам (аналитический):", loss_dxx.item())
    print("Разница:", abs(hessian_inputs.item() - loss_dxx.item()))

    # Аналитическое вычисление второй производной по параметру a
    with torch.no_grad():
        # Первая производная f по a
        f_da = sin_exp_x
        # Вторая производная f по a
        f_daa = torch.tensor(0.0)  # Вторая производная по a равна нулю
        # Первая производная потери по a
        loss_da = 2 * f * f_da
        # Вторая производная потери по a
        loss_daa = 2 * (f_da * f_da + f * f_daa)  # Так как f_daa = 0, то loss_daa = 2 * (f_da)^2

    print("Гессиан по параметрам (численный):", hessian_params.item())
    print("Гессиан по параметрам (аналитический):", loss_daa.item())
    print("Разница:", abs(hessian_params.item() - loss_daa.item()))

if __name__ == "__main__":
    test_hessian_calculator()