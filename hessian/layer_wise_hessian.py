import torch
import torch.nn as nn
from collections import OrderedDict

def _is_activation_module(mod: nn.Module) -> bool:
    """
    Вспомогательная функция: возвращает True, если у модуля нет параметров
    (обычно это чистая активация: Sigmoid, ReLU, Tanh и т.п.).
    """
    return sum(p.numel() for p in mod.parameters()) == 0

def flatten_grads(grad_list, layer_params):
    """
    Склеиваем список градиентов (по каждому параметру слоя) в один вектор,
    подставляя нули там, где grad = None.
    Возвращает (grads_vector, shapes).
    """
    flat = []
    shapes = []
    for g, p in zip(grad_list, layer_params):
        if g is None:
            flat.append(torch.zeros_like(p).view(-1))
            shapes.append(p.shape)
        else:
            flat.append(g.contiguous().view(-1))
            shapes.append(g.shape)
    return torch.cat(flat), shapes

def compute_hessian_of_scalar_function(scalar_output, layer_params):
    """
    Вычисляет Hessian (вторые производные) размерности (P, P),
    где P = сумма чисел параметров в layer_params.
    Используем построчный вызов grad для каждой компоненты первого градиента.
    """
    # 1) Градиент (Jacobian) scalar_output по параметрам
    grads = torch.autograd.grad(
        scalar_output,
        layer_params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    grads_vector, _ = flatten_grads(grads, layer_params)
    param_count = grads_vector.numel()

    H = torch.zeros(param_count, param_count, device=grads_vector.device)

    for i in range(param_count):
        if not grads_vector[i].requires_grad:
            # компонента градиента константна => её вторые произв. = 0
            continue
        g2 = torch.autograd.grad(
            grads_vector[i],
            layer_params,
            retain_graph=True,
            allow_unused=True
        )
        g2_vector, _ = flatten_grads(g2, layer_params)
        H[i, :] = g2_vector

    return H

def chunkify_model_with_activations(model: nn.Module):
    """
    Разбивает модель (последовательную) на куски вида (param_module, activation_module_или_None).
    Идея:
    - Идём по списку модулей (model.children()) последовательно.
    - Если встречаем модуль с параметрами (например, Linear),
      смотрим, является ли следующий модуль чистой активацией (нет параметров).
      Если да, объединяем их в один chunk:
          (Linear, Sigmoid)
      Если нет, делаем chunk вида (Linear, None).
    - Модули-активации, у которых нет параметров (ReLU, Sigmoid, ...) пропускаются, если они уже
      добавлены к chunk. Если встретится активация не после параметризованного модуля,
      она просто игнорируется (хотя в реальных сетях обычно не бывает «голой» активации).
    Возвращает список chunk'ов: [ (mod0, act0), (mod1, act1), ... ]
    """
    modules = list(model.children())
    def flatten_modules(modules):
        flattened = []
        for mod in modules:
            if isinstance(mod, nn.ModuleList):
                flattened.extend(flatten_modules(mod))
            else:
                flattened.append(mod)
        return flattened

    modules = flatten_modules(modules)
    chunks = []
    skip_next = False
    i = 0
    while i < len(modules):
        if sum(p.numel() for p in modules[i].parameters()) > 0:
            # Это параметризованный модуль
            param_mod = modules[i]
            act_mod = None
            # Посмотрим, есть ли "следующий" модуль и не является ли он параметризованным
            if i + 1 < len(modules) and _is_activation_module(modules[i+1]):
                # Берём это как активацию
                act_mod = modules[i+1]
                skip_next = True
            chunks.append((param_mod, act_mod))
        else:
            # Модуль без параметров — активация. Если мы не присоединили её к предыдущему,
            # пропустим её (или можно завести chunk (None, activation), но тогда Hessian = 0,
            # т.к. нет параметров)
            pass
        if skip_next:
            # мы использовали i+1 как активацию, пропустим его
            skip_next = False
            i += 2
        else:
            i += 1
    return chunks


def compute_local_hessians_for_chunks(model: nn.Module,
                                     x: torch.Tensor,
                                     aggregator=torch.sum):
    """
    1) Автоматически разбивает модель на "слои+(следующую)активацию" —
       используя chunkify_model_with_activations.
    2) Для каждого chunk[i], делаем forward от chunk[0] до chunk[i], превращаем выход
       в скаляр (aggregator), вычисляем Гессе по параметрам chunk[i]. 
       (Внутри chunk[i] могут быть: (Linear, Sigmoid), (Conv, ReLU), или (Linear, None) и т.д.)

    Возвращает OrderedDict[(i)-th_chunk, Hessian].
    """
    # 1) Разобьём модель на chunks
    chunks = chunkify_model_with_activations(model)
    # Превратим chunks в nn.Sequential, где каждый элемент = chunk
    # для удобства последовательного прохождения.
    # Но нужно иметь возможность "прогнать" первые i chunk'ов.
    # Сделаем список, где chunk -- это nn.Module (обёртка).
    # Или просто вручную будем применять chunk.

    # Обёртка для одного chunk
    class ChunkModule(nn.Module):
        def __init__(self, param_mod, act_mod=None):
            super().__init__()
            self.param_mod = param_mod  # например, Linear
            self.act_mod = act_mod      # например, Sigmoid или None
        def forward(self, inp):
            out = self.param_mod(inp)
            if self.act_mod is not None:
                out = self.act_mod(out)
            return out

    chunk_modules = [ChunkModule(pm, am) for (pm, am) in chunks]

    # Теперь у нас есть список chunk_modules (каждый сам по себе nn.Module).
    # Сформируем "квази-последовательность":
    #   full_sequence = [chunk0, chunk1, chunk2, ...]
    # Чтобы прогонять по шагам.

    # 2) Вычислим "локальные" Гессе
    layerwise_hessians = OrderedDict()

    # Для i-го куска:
    #   - делаем forward через все куски [0..i-1] для получения входа в кусок i,
    #   - прогоняем кусок i, aggregator,
    #   - считаем Гессе по param_mod i-го куска.
    # Обратите внимание: если кусок i это (Linear, Sigmoid), то параметры только у Linear.
    # Сигмоида параметров не имеет, так что Hessian будет (P_i x P_i), где P_i — число параметров Linear.

    def forward_up_to(chunks_list, end_idx, x):
        """
        Прогоняем вход x последовательно через chunks_list[0], chunks_list[1], ..., chunks_list[end_idx-1].
        Возвращаем выход.
        """
        out = x
        for idx in range(end_idx):
            out = chunks_list[idx](out)
        return out

    for i, chunk_mod in enumerate(chunk_modules):
        # Параметры текущего "куска" = параметры chunk_mod.param_mod
        param_list = list(chunk_mod.param_mod.parameters())
        if not param_list:
            # Нет параметров => Hessian = 0, пропустим (или можно записать нулевую матрицу)
            layerwise_hessians[i] = torch.zeros(0, 0)
            continue

        # Получаем вход для i-го куска
        # (прогоняем x через первые i кусков)
        with torch.no_grad():
            # Здесь можно без градиентов, потому что далее мы заново 
            # сделаем enable_grad для самого куска i
            current_input = forward_up_to(chunk_modules, i, x)

        # Подключим gradient tracking на вход, чтобы всё считалось
        current_input = current_input.detach()
        current_input.requires_grad_(True)

        # Прогоняем сам i-й кусок под grad 
        # (важно включить torch.enable_grad() / no_grad=False)
        with torch.enable_grad():
            # out_i = chunk i ( current_input )
            out_i = chunk_mod(current_input)
            # Превращаем выход куска в скаляр
            scalar_output = aggregator(out_i)

            # Вычисляем Гессе
            H_i = compute_hessian_of_scalar_function(scalar_output, param_list)
            layerwise_hessians[i] = H_i.detach()

    return layerwise_hessians
