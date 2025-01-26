"""
Данный модуль содержит реализацию двух типов "трейнеров" (Trainer) для обучения 
моделей (персептронов) на классификационных и регрессионных датасетах:

1. BaseTrainer (абстрактный класс):
   - Содержит общие методы загрузки датасетов из конфигурации
   - Создаёт DataLoader-ы для train/test
   - Организует обучение (цикл по итерациям)
   - Логирует процесс обучения в файл
   - Сохраняет модель и метаданные на каждой итерации

2. ClassificationTrainer(BaseTrainer):
   - Реализует compute_metrics() для задач классификации:
       Accuracy, Precision, Recall, F1, ROC, AUC
   - Сохраняет чекпоинты и метаданные

3. RegressionTrainer(BaseTrainer):
   - Реализует compute_metrics() для задач регрессии:
       R2, MAE, RMSE
   - Сохраняет чекпоинты и метаданные

**Примечания**:
- Для датасетов используем структуру (из `DATASET_CONFIG`), где есть:
    iterations, DatasetPreparer, ...
- Для моделей используем структуру (из `models.py`), где есть:
    create_model(dataset_name, variant)

- Запуск на GPU, если доступен (torch.cuda.is_available()).
- После каждой итерации (batch):
    - Логируем в файл `results/<date>/<dataset_name>/<model_type>/logs.log`
    - Сохраняем модель: `results/<date>/<dataset_name>/<model_type>/<iteration>-model.pth`
    - Сохраняем pickle с метаданными: `results/<date>/<dataset_name>/<model_type>/<iteration>-data.pkl`

- "temp" поле в метаданном pkl — заглушка, можно хранить что угодно.
"""

import os
import sys
import pickle
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)

from datasets import DATASET_CONFIG

# Для импорта модулей из корня проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from hessian.layer_wise_hessian import compute_local_hessians_for_chunks
from analysys.linal import get_eigenvalues, get_rank
from analysys.stats import compute_tensor_statistics
from analysys.gradients import extract_gradients_by_layer
from analysys.hessian import compute_condition_number
from models import create_model

import warnings

warnings.filterwarnings("ignore")


###############################################################################
#                 Базовый класс Trainer (абстрактный)                         #
###############################################################################
class Trainer(ABC):
    def __init__(self,
                 dataset_name: str,
                 dataset_type: str,  # "classification" или "regression"
                 datetime_str: str = None,
                 batch_size: int = 64):
        """
        Параметры
        ---------
        dataset_name : str
            Название датасета (например, "MNIST").
        dataset_type : str
            "classification" или "regression", чтобы знать, где искать в DATASET_CONFIG.
        date_str : str
            Строка с датой (для сохранения в results/<date_str> ...). 
            Если None, то берётся текущая дата.
        batch_size : int
            Размер batch для DataLoader.
        """
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

        if datetime_str is None:
            datetime_str = datetime.datetime.now().isoformat()
        self.date_str = datetime_str

        self.batch_size = batch_size

        self.dataset_conf = DATASET_CONFIG[dataset_type][dataset_name]

        self.iterations = self.dataset_conf["iterations"]
        loss_cls = self.dataset_conf["loss_function"]["type"]
        loss_args = self.dataset_conf["loss_function"]["args"]
        self.loss_fn = loss_cls(**loss_args)

        self.optimizer_cls = self.dataset_conf["optimizer"]["type"]
        self.optimizer_args = self.dataset_conf["optimizer"]["args"]

        self.dataset_preparer = self.dataset_conf["DatasetPreparer"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.test_loader = self._prepare_dataloaders()

    def _prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Вызывает dataset_preparer для получения (X_train, y_train, X_test, y_test)
        и создаёт DataLoader-ы (train_loader, test_loader).
        """
        if self.dataset_preparer is None:
            raise ValueError(
                f"У датасета {self.dataset_name} не назначена функция DatasetPreparer!"
            )

        X_train, X_test, y_train, y_test = self.dataset_preparer()

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train)
        X_test_t  = torch.tensor(X_test, dtype=torch.float32)
        y_test_t  = torch.tensor(y_test)

        if self.dataset_type == "classification":
            y_train_t = y_train_t.long()
            y_test_t = y_test_t.long()
        elif self.dataset_type == "regression":
            y_train_t = y_train_t.float()
            y_test_t = y_test_t.float()

        train_ds = TensorDataset(X_train_t, y_train_t)
        test_ds  = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train_all_variants(self):
        """
        Запускает обучение для всех трёх типов персептронов: "no", "sure", "huge".
        """
        for model_type in ["no", "sure", "huge"]:
            self._train_one_variant(model_type)

    def _train_one_variant(self, model_type: str):
        """
        Обучение одной модели (одного варианта).
          - Создаём модель
          - Настраиваем оптимизатор
          - В цикле итераций (до self.iterations) делаем:
              - одну порцию batch
              - forward, backward, optimizer.step
              - собираем метрики (train, а также eval)
              - логируем
              - сохраняем чекпоинт + data.pkl
        """
        model = create_model(self.dataset_name, model_type).to(self.device)
        optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_args)

        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "results", 
            self.date_str, 
            self.dataset_name, 
            model_type
        )
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "logs.log")

        with open(log_path, "a", encoding="utf-8") as log_f:
            iteration_count = 0

            train_iter = iter(self.train_loader)
            for _ in range(self.iterations):
                try:
                    batch_x, batch_y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch_x, batch_y = next(train_iter)

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = model(batch_x)

                loss = self.loss_fn(preds, batch_y)
                loss.backward()
                grads = extract_gradients_by_layer(model)
                optimizer.step()

                iteration_count += 1
                test_metrics = self.evaluate(model)
                model.train()

                if loss.item() < 1e-2:
                    print(f"Loss is too low ({loss.item():.4f}), stopping training.")
                    break

                log_line = (
                    f"Iter {iteration_count} | "
                    f"Dataset: {self.dataset_name} | "
                    f"Model type: {model_type} | "
                    f"Loss: {loss.item():.4f} | "
                    + " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
                )
                log_f.write(log_line + "\n")
                print(log_line)

                # ckpt_path = os.path.join(save_dir, f"{iteration_count}-model.pth")
                # torch.save(model.state_dict(), ckpt_path)

                self.save_model_data(model, batch_x, grads, save_dir, iteration_count)

    def save_model_data(self, model, input, grads, save_dir, iteration_count):
        params = dict(model.named_parameters())
        params_ranks = {}
        params_spectral = {}

        for name, param in params.items():
            if param.dim() == 2:  # Проверяем, что параметр является матрицей
                params_ranks[name] = get_rank(param)
            params_spectral[name] = compute_tensor_statistics(param)

        hess = compute_local_hessians_for_chunks(model, input)
        hess_spectral = {i: compute_tensor_statistics(hessian) for i, hessian in hess.items()}
        hess_eigs = {i: get_eigenvalues(hessian) for i, hessian in hess.items()}
        hess_eigs_spectral = {i: compute_tensor_statistics(eigs) for i, eigs in hess_eigs.items()}
        hess_ranks = {i: get_rank(hessian) for i, hessian in hess.items()}
        hess_eigs_conditionals = {i: compute_condition_number(eigs) for i, eigs in hess_eigs.items()}

        grad_spectral = {name: compute_tensor_statistics(grad) for name, grad in grads.items()}

        meta_data = {}
        layer_idx = 0
        for name, param in params.items():
            layer_identifier = f"layer.{layer_idx}"
            if name.endswith("weight"):
                meta_data[layer_identifier] = {
                    "weights": param.detach().cpu().tolist(),
                    "weights_rank": params_ranks[name].detach().cpu().tolist(),
                    "weights_spectral": params_spectral[name],
                    "hessian": hess[layer_idx].tolist(),
                    "hessian_spectral": hess_spectral[layer_idx],
                    "hessian_eigens": hess_eigs[layer_idx].tolist(),
                    "hessian_eigens_spectral": hess_eigs_spectral[layer_idx],
                    "hessian_rank": int(hess_ranks[layer_idx].detach().cpu().long()),
                    "hessian_condition": hess_eigs_conditionals[layer_idx],
                    "gradient": grads[name].detach().cpu().tolist(),
                    "gradient_spectral": grad_spectral[name]
                }
                layer_idx += 1
            elif name.endswith("bias"):
                past_identifier = f"layer.{layer_idx - 1}"
                meta_data[past_identifier]["bias"] = param.detach().cpu().tolist()
                meta_data[past_identifier]["bias_spectral"] = params_spectral[name]
                meta_data[past_identifier]["bias_gradient"] = grads[name].detach().cpu().tolist()
                meta_data[past_identifier]["bias_gradient_spectral"] = grad_spectral[name]

        pkl_path = os.path.join(save_dir, f"{iteration_count}-data.pkl")
        with open(pkl_path, "wb") as pkl_f:
            pickle.dump(meta_data, pkl_f)

    @abstractmethod
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """
        Оценивает модель на self.test_loader, считает метрики и возвращает словарь
        { "metric_name": value, ... }.
        Определяется в наследниках (ClassificationTrainer и RegressionTrainer).
        """
        pass


###############################################################################
#      Трейнер для классификации (Accuracy, Precision, Recall, ROC, AUC, F1)  #
###############################################################################
class ClassificationTrainer(Trainer):
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for bx, by in self.test_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                logits = model(bx)  # (batch, num_classes)
                # Предположим, что это много-классовая классификация или бинарная
                # Если много классов: pred = argmax dim=1
                # Если бинарная: то тоже argmax (num_classes=2)
                pred_labels = torch.argmax(logits, dim=1)
                all_preds.append(pred_labels.cpu().numpy())
                all_trues.append(by.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        # Метрики:
        acc = accuracy_score(all_trues, all_preds)
        prec = precision_score(all_trues, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_trues, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)

        # Для ROC и AUC нужно получать вероятности. 
        # Для упрощения если num_classes=2, возьмём логиты[:,1] как "score"
        # Если классов > 2, здесь уже сложнее (OneVsRest). 
        # Покажем упрощённый вариант для бинарного случая:
        if len(np.unique(all_trues)) == 2:
            # Выделяем score = logits[:,1] через повторный проход (не очень эффективно)
            # или можно было сохранить при первом проходе.
            all_scores = []
            idx_class1 = 1  # для бинарного
            with torch.no_grad():
                for bx, _ in self.test_loader:
                    bx = bx.to(self.device)
                    logits = model(bx)
                    scores = logits[:, idx_class1]  # (batch,)
                    all_scores.append(scores.cpu().numpy())
            all_scores = np.concatenate(all_scores)

            fpr, tpr, _ = roc_curve(all_trues, all_scores, pos_label=1)
            _auc = auc(fpr, tpr)
        else:
            # Много классов -> заглушка
            _auc = 0.0

        metrics_dict = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": _auc  # В много-классовом случае здесь формально 0, 
                         # реально нужно OneVsRest подход или другое усреднение
        }
        return metrics_dict


###############################################################################
#      Трейнер для регрессии (R2, MAE, RMSE)                                  #
###############################################################################
class RegressionTrainer(Trainer):
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for bx, by in self.test_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                preds = model(bx)  # (batch, 1)
                # Сгладим в вектор
                preds = preds.view(-1).cpu().numpy()
                all_preds.append(preds)

                true_vals = by.view(-1).cpu().numpy()
                all_trues.append(true_vals)

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        # Метрики регрессии
        r2 = r2_score(all_trues, all_preds)
        mae = mean_absolute_error(all_trues, all_preds)
        mse = mean_squared_error(all_trues, all_preds)
        rmse = np.sqrt(mse)

        metrics_dict = {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        }
        return metrics_dict


###############################################################################
#   Пример использования                                                      #
###############################################################################
if __name__ == "__main__":
    trainer = ClassificationTrainer(
        dataset_name="Make Biclusters",
        dataset_type="classification",
        batch_size=32
    )
    trainer.train_all_variants()

    reg_trainer = RegressionTrainer(
        dataset_name="Energy Efficiency",
        dataset_type="regression",
        batch_size=16
    )
    reg_trainer.train_all_variants()
