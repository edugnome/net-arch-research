def extract_gradients_by_layer(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
           grads[name] = param.grad.detach().clone()
        else:
            grads[name] = None
    print("Градиенты по слоям:")
    for layer_name, grad in grads.items():
        print(f"{layer_name}: {grad}")
