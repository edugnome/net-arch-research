import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, List
from torch.autograd.functional import hessian
from functorch import make_functional

class HessianCalculator:
    def __init__(self, model: nn.Module, loss_fn: Callable[[Tensor, Tensor], Tensor]) -> None:
        """
        Инициализируем класс с моделью и функцией потерь.
        """
        self.model: nn.Module = model
        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = loss_fn
        # Преобразуем модель в функциональную форму
        self.fmodel, self.params = make_functional(model)

    def hessian_wrt_inputs(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Вычисляем матрицу Гессе по входным данным.
        """
        inputs = inputs.clone().requires_grad_(True)

        def loss_fn_wrapped(x: Tensor) -> Tensor:
            outputs: Tensor = self.model(x)
            return self.loss_fn(outputs, targets)

        return hessian(loss_fn_wrapped, inputs)

    def hessian_wrt_params(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Вычисляем матрицу Гессе по параметрам модели.
        """
        # Преобразуем параметры в один плоский тензор
        flat_params: Tensor = torch.cat([p.view(-1) for p in self.params]).requires_grad_(True)

        def loss_fn_wrapped(flat_params: Tensor) -> Tensor:
            # Восстанавливаем список параметров из плоского тензора
            param_shapes: List[torch.Size] = [p.shape for p in self.params]
            param_sizes: List[int] = [p.numel() for p in self.params]
            param_tensors: List[Tensor] = []
            idx: int = 0
            for size, shape in zip(param_sizes, param_shapes):
                param = flat_params[idx:idx+size].view(shape)
                param_tensors.append(param)
                idx += size
            # Вычисляем выход модели с новыми параметрами
            outputs: Tensor = self.fmodel(param_tensors, inputs)
            return self.loss_fn(outputs, targets)

        return hessian(loss_fn_wrapped, flat_params)