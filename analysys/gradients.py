def extract_gradients_by_layer(model):
    """
    Извлекает и выводит градиенты для каждого слоя модели PyTorch.

    Эта функция перебирает все именованные параметры модели, собирает их
    градиенты (если они существуют) и выводит их послойно.

    Параметры:
    ----------
    model : torch.nn.Module
        Модель PyTorch, из которой извлекаются градиенты. Для наличия градиентов
        модель должна пройти обратное распространение ошибки.

    Возвращает:
    -----------
    dict
        Словарь, сопоставляющий имена слоев с их соответствующими градиентами.
        Если у слоя нет градиента, его значение будет None.

    Пример:
    -------
    >>> model = torch.nn.Linear(10, 5)
    >>> loss = criterion(model(x), y)
    >>> loss.backward()
    >>> grads = extract_gradients_by_layer(model)
    """

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
           grads[name] = param.grad.detach().clone()
        else:
            grads[name] = None
    
    return grads
