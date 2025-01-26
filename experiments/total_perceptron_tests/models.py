"""
Модуль, содержащий индивидуальные настройки (in/out, тип задачи, размер скрытых слоёв,
функцию активации и её параметры, а также включение/выключение BatchNorm) 
для каждого датасета из datasets_dict.

Важные замечания:
-------------------------
1. У каждого датасета одинаковое ЧИСЛО скрытых слоёв для трёх вариантов ("no"/"sure"/"huge"),
   но размеры слоёв различаются. 
2. В рамках данной конфигурации мы также храним "architecture_params", где:
   - "activation": строка с названием активации ("ReLU", "LeakyReLU", "Tanh", ...).
   - "activation_args": dict с параметрами для активации (например, {"inplace": True} или
     {"negative_slope": 0.01}).
   - "batchnorm": bool, использовать ли BatchNorm1d после каждого слоя.
3. На практике эти настройки можно дорабатывать и уточнять под реальные эксперименты.

Пример использования:
-------------------------
    from models import DATASET_CONFIGS, create_model

    model_no = create_model("MNIST", "no")     # слишком маленькие слои
    model_ok = create_model("MNIST", "sure")   # оптимальные
    model_huge = create_model("MNIST", "huge") # избыточные
"""

import torch
import torch.nn as nn

###############################################################################
# Глобальный словарь конфигураций для всех датасетов                          #
###############################################################################

DATASET_ARCHITECTURE_CONFIGS = {
    # -------------------- Classification --------------------
    # "MNIST": {
    #     "type": "classification",
    #     "in": 784,
    #     "out": 10,
    #     "hidden_layers": {
    #         "no":   [32, 16],
    #         "sure": [128, 64],
    #         "huge": [512, 256],
    #     },
    #     "architecture_params": {
    #         "activation": "ReLU",
    #         "activation_args": {"inplace": True},
    #         "batchnorm": False
    #     }
    # },
    "Fashion-MNIST": {
        "type": "classification",
        "in": 784,
        "out": 10,
        "hidden_layers": {
            # Было no=[32,16,8], sure=[1024,512,256], huge=[4096,2048,1024]
            # Переформатировано для большего баланса
            "no":   [64, 32, 16],
            "sure": [256, 128, 64],
            "huge": [1024, 512, 256],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    # "CIFAR-10": {
    #     "type": "classification",
    #     "in": 3 * 32 * 32,  # 3072
    #     "out": 10,
    #     "hidden_layers": {
    #         "no":   [64, 32],
    #         "sure": [256, 128],
    #         "huge": [1024, 512],
    #     },
    #     # Большое число входных признаков -> LeakyReLU
    #     "architecture_params": {
    #         "activation": "LeakyReLU",
    #         "activation_args": {"negative_slope": 0.01, "inplace": True},
    #         "batchnorm": False
    #     }
    # },
    "CIFAR-100": {
        "type": "classification",
        "in": 3 * 32 * 32,  # 3072
        "out": 100,
        "hidden_layers": {
            "no":   [128, 64, 32],
            "sure": [1024, 512, 256],
            "huge": [2048, 1024, 512],
        },
        "architecture_params": {
            "activation": "LeakyReLU",
            "activation_args": {"negative_slope": 0.01, "inplace": True},
            "batchnorm": False
        }
    },
    "KMNIST": {
        "type": "classification",
        "in": 784,
        "out": 10,
        "hidden_layers": {
            # Было no=[32,16,8], sure=[512,256,128], huge=[2048,1024,512]
            "no":   [64, 32, 16],
            "sure": [256, 128, 64],
            "huge": [1024, 512, 256],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "EMNIST": {
        "type": "classification",
        "in": 784,
        "out": 47,
        "hidden_layers": {
            # Аналогично
            "no":   [64, 32, 16],
            "sure": [256, 128, 64],
            "huge": [1024, 512, 256],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Iris": {
        "type": "classification",
        "in": 4,
        "out": 3,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Wine": {
        "type": "classification",
        "in": 13,
        "out": 3,
        "hidden_layers": {
            "no":   [4, 2, 2],
            "sure": [16, 8, 4],
            "huge": [64, 32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Breast Cancer Wisconsin": {
        "type": "classification",
        "in": 30,
        "out": 2,
        "hidden_layers": {
            "no":   [8, 4],
            "sure": [32, 16],
            "huge": [128, 64],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Digits": {
        "type": "classification",
        "in": 64,
        "out": 10,
        "hidden_layers": {
            "no":   [16, 8, 4],
            "sure": [64, 32, 16],
            "huge": [256, 128, 64],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "SpamBase": {
        "type": "classification",
        "in": 57,
        "out": 2,
        "hidden_layers": {
            "no":   [8, 4],
            "sure": [32, 16],
            "huge": [128, 64],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Classification": {
        "type": "classification",
        "in": 10,
        "out": 2,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Blobs": {
        "type": "classification",
        "in": 2,
        "out": 3,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Titanic Dataset": {
        "type": "classification",
        "in": 4,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Adult Income": {
        "type": "classification",
        "in": 4,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Credit Card Fraud Detection": {
        "type": "classification",
        "in": 30,
        "out": 2,
        "hidden_layers": {
            "no":   [8, 4],
            "sure": [32, 16],
            "huge": [128, 64],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Biclusters": {
        "type": "classification",
        "in": 5,
        "out": 2,
        "hidden_layers": {
            "no":   [1, 1],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Checkerboard": {
        "type": "classification",
        "in": 5,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Circles": {
        "type": "classification",
        "in": 2,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Gaussian Quantiles": {
        "type": "classification",
        "in": 2,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Hastie 10 2": {
        "type": "classification",
        "in": 10,
        "out": 2,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Moons": {
        "type": "classification",
        "in": 2,
        "out": 2,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Multilabel Classification": {
        "type": "classification",
        "in": 10,
        "out": 3,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },

    # --------------------- Regression -----------------------
    "Diabetes": {
        "type": "regression",
        "in": 10,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Energy Efficiency": {
        "type": "regression",
        "in": 9,
        "out": 1,
        "hidden_layers": {
            "no":   [8, 4, 2],
            "sure": [32, 16, 8],
            "huge": [64, 32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Airfoil Self-Noise": {
        "type": "regression",
        "in": 5,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Concrete Compressive Strength": {
        "type": "regression",
        "in": 8,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Regression": {
        "type": "regression",
        "in": 10,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "House Prices Dataset": {
        "type": "regression",
        "in": 4,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Friedman1": {
        "type": "regression",
        "in": 10,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Friedman2": {
        "type": "regression",
        "in": 4,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Friedman3": {
        "type": "regression",
        "in": 4,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Low Rank Matrix": {
        "type": "regression",
        "in": 10,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make S Curve": {
        "type": "regression",
        "in": 3,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Sparse SPD Matrix": {
        "type": "regression",
        "in": 100,
        "out": 1,
        "hidden_layers": {
            "no":   [16, 8, 4],
            "sure": [128, 64, 32],
            "huge": [512, 256, 128],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Sparse Uncorrelated": {
        "type": "regression",
        "in": 10,
        "out": 1,
        "hidden_layers": {
            "no":   [4, 2],
            "sure": [16, 8],
            "huge": [64, 32],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make SPD Matrix": {
        "type": "regression",
        "in": 100,
        "out": 1,
        "hidden_layers": {
            "no":   [16, 8, 4],
            "sure": [128, 64, 32],
            "huge": [512, 256, 128],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
    "Make Swiss Roll": {
        "type": "regression",
        "in": 3,
        "out": 1,
        "hidden_layers": {
            "no":   [2, 2],
            "sure": [8, 4],
            "huge": [32, 16],
        },
        "architecture_params": {
            "activation": "ReLU",
            "activation_args": {"inplace": True},
            "batchnorm": False
        }
    },
}


###############################################################################
# Класс DynamicMLP, учитывающий активацию, её параметры, а также BatchNorm    #
###############################################################################
class DynamicMLP(nn.Module):
    """
    MLP: [Linear -> (BatchNorm1d) -> Activation] x N скрытых слоёв -> Linear(out_features).

    Параметры:
    ----------
    in_features: int
        Размер входного вектора (число признаков).
    hidden_dims: list[int]
        Число нейронов для каждого скрытого слоя. 
        Если список пуст, модель будет содержать только один Linear in->out.
    out_features: int
        Размер выходного вектора.
    is_classification: bool
        Если True, модель под задачу классификации (без SoftMax, т.к. CrossEntropyLoss).
        Если False, модель для регрессии.
    activation: str
        Название активации ("ReLU", "LeakyReLU", "Tanh", "Sigmoid", и т.д.).
    activation_args: dict
        Словарь параметров для соответствующего модуля активации (например, {"inplace": True}).
    use_batchnorm: bool
        Если True, после каждого Linear делаем BatchNorm1d.
    """
    def __init__(self, 
                 in_features: int,
                 hidden_dims: list,
                 out_features: int,
                 is_classification: bool,
                 activation: str,
                 activation_args: dict,
                 use_batchnorm: bool):
        super().__init__()
        self.is_classification = is_classification

        valid_activations = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "Tanh": nn.Tanh,
            "Sigmoid": nn.Sigmoid,
        }
        if activation not in valid_activations:
            raise ValueError(f"Неизвестная активация: {activation}")
        activation_cls = valid_activations[activation]

        layers = []
        prev_dim = in_features

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hdim))
            # Добавляем активацию
            layers.append(activation_cls(**activation_args))
            prev_dim = hdim

        # Финальный слой (без активации)
        layers.append(nn.Linear(prev_dim, out_features))
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


###############################################################################
# Функция для создания модели под конкретный датасет и вариант ("no"/"sure"/"huge")
###############################################################################
def create_model(dataset_name: str, variant: str) -> nn.Module:
    """
    Создать модель для указанного датасета `dataset_name` и варианта:
    - 'no':   слишком маленькие слои (скорее всего не обучится)
    - 'sure': адекватная архитектура
    - 'huge': избыточная архитектура (почти наверняка обучится)

    Возвращает объект nn.Module (DynamicMLP).
    """
    if dataset_name not in DATASET_ARCHITECTURE_CONFIGS:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

    conf = DATASET_ARCHITECTURE_CONFIGS[dataset_name]
    if variant not in conf["hidden_layers"]:
        raise ValueError(f"Для датасета '{dataset_name}' нет варианта '{variant}'")

    hidden_dims = conf["hidden_layers"][variant]
    in_dim = conf["in"]
    out_dim = conf["out"]
    is_cls = (conf["type"] == "classification")

    # Параметры архитектуры (активация, её аргументы, batchnorm)
    arch_params = conf["architecture_params"]
    act_name = arch_params["activation"]
    act_args = arch_params["activation_args"]
    use_bn = arch_params["batchnorm"]

    model = DynamicMLP(
        in_features=in_dim,
        hidden_dims=hidden_dims,
        out_features=out_dim,
        is_classification=is_cls,
        activation=act_name,
        activation_args=act_args,
        use_batchnorm=use_bn
    )
    return model


if __name__ == "__main__":
    model_no   = create_model("Airfoil Self-Noise", "no")
    model_sure = create_model("Airfoil Self-Noise", "sure")
    model_huge = create_model("Airfoil Self-Noise", "huge")
    
    print("Airfoil Self-Noise, NO:")
    print(model_no.layers)
    print()
    print("Airfoil Self-Noise, SURE:")
    print(model_sure.layers)
    print()
    print("Airfoil Self-Noise, HUGE:")
    print(model_huge.layers)
    
    # Проверим выходные размеры
    x_dummy = torch.randn(2, DATASET_ARCHITECTURE_CONFIGS["Airfoil Self-Noise"]["in"])
    out_no   = model_no(x_dummy)
    out_sure = model_sure(x_dummy)
    out_huge = model_huge(x_dummy)
    print("\nShapes:", out_no.shape, out_sure.shape, out_huge.shape)
    # Должно быть (2, 10) для всех.
