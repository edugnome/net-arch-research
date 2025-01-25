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
        Вычисляем полную матрицу Гессе по входным данным (по всем входам как одному блоку).
        """
        inputs = inputs.clone().requires_grad_(True)

        def loss_fn_wrapped(x: Tensor) -> Tensor:
            outputs: Tensor = self.model(x)
            return self.loss_fn(outputs, targets)

        return hessian(loss_fn_wrapped, inputs)

    def hessian_wrt_params(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Вычисляем полную матрицу Гессе по всем параметрам (как по одному большому вектору).
        """
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
            outputs: Tensor = self.fmodel(param_tensors, inputs)
            return self.loss_fn(outputs, targets)

        return hessian(loss_fn_wrapped, flat_params)
    
    def hessian_wrt_each_layer_params(self, inputs: Tensor, targets: Tensor) -> List[Tensor]:
        """
        Вычисляет матрицу Гессе для каждого слоя с параметрами отдельно.
        
        Возвращает список матриц Гессе, соответствующих каждому слою с параметрами.
        """
        layer_hessians: List[Tensor] = []
        
        for layer in self.model.children():
            if any(param.requires_grad for param in layer.parameters()):
                params = [param for param in layer.parameters() if param.requires_grad]
                flat_params = torch.cat([p.view(-1) for p in params]).requires_grad_(True)
                
                def loss_fn_wrapped(flat_p: Tensor) -> Tensor:
                    param_tensors = []
                    idx = 0
                    for param in params:
                        size = param.numel()
                        tensor = flat_p[idx:idx + size].view(param.shape)
                        param_tensors.append(tensor)
                        idx += size
                    # Восстанавливаем параметры слоя
                    with torch.no_grad():
                        for p, new_p in zip(params, param_tensors):
                            p.copy_(new_p)
                    # Прогоняем ввод через модель до текущего слоя
                    outputs = inputs
                    for sub_layer in self.model.children():
                        outputs = sub_layer(outputs)
                        if sub_layer == layer:
                            break
                    return self.loss_fn(outputs, targets)
                
                hess = hessian(loss_fn_wrapped, flat_params)
                layer_hessians.append(hess)
        
        return layer_hessians