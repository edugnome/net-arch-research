import torch
import numpy as np
from datasets import DATASET_CONFIG
from models import DATASET_ARCHITECTURE_CONFIGS, create_model
import traceback

def dataset_autotester(X_train, X_test, y_train, y_test, n_classes=None):
    """
    Автоматический тестер для проверки датасетов.
    
    :param X_train: ndarray, тренировочные данные
    :param X_test: ndarray, тестовые данные
    :param y_train: ndarray, тренировочные метки
    :param y_test: ndarray, тестовые метки
    :param n_classes: int, количество классов (необязательно, если метки можно определить)
    :return: None
    """
    try:
        print("=== Автотест датасетов ===")

        assert X_train.shape[0] == y_train.shape[0], "Несоответствие размеров X_train и y_train"
        assert X_test.shape[0] == y_test.shape[0], "Несоответствие размеров X_test и y_test"
        print(f"✔ Размеры данных соответствуют: X_train {X_train.shape}, X_test {X_test.shape}")

        assert isinstance(X_train, np.ndarray), "X_train должен быть np.ndarray"
        assert isinstance(X_test, np.ndarray), "X_test должен быть np.ndarray"
        assert isinstance(y_train, np.ndarray), "y_train должен быть np.ndarray"
        assert isinstance(y_test, np.ndarray), "y_test должен быть np.ndarray"
        print("✔ Все данные имеют тип numpy.ndarray")

        y_min, y_max = y_train.min(), y_train.max()
        if n_classes is None:
            n_classes = len(np.unique(y_train))
        assert y_min >= 0, "Метки не должны быть отрицательными"
        assert y_max < n_classes, f"Метка {y_max} выходит за пределы [0, {n_classes-1}]"
        print(f"✔ Метки в допустимом диапазоне [0, {n_classes-1}]")

        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        assert set(test_classes).issubset(set(train_classes)), "Тестовые метки содержат классы, отсутствующие в тренировочных данных"
        print(f"✔ Уникальные классы меток: {train_classes}")

        assert X_train.size > 0, "Тренировочные данные пусты"
        assert X_test.size > 0, "Тестовые данные пусты"
        assert y_train.size > 0, "Тренировочные метки пусты"
        assert y_test.size > 0, "Тестовые метки пусты"
        print("✔ Данные не содержат пустых массивов")

        assert X_train.shape[1] == X_test.shape[1], "Количество фичей в X_train и X_test должно совпадать"
        print(f"✔ Количество фичей совпадает: {X_train.shape[1]}")

        assert not np.isnan(X_train).any(), "X_train содержит NaN значения"
        assert not np.isnan(X_test).any(), "X_test содержит NaN значения"
        assert not np.isinf(X_train).any(), "X_train содержит бесконечные значения"
        assert not np.isinf(X_test).any(), "X_test содержит бесконечные значения"
        print("✔ Данные не содержат NaN или бесконечных значений")

        print("=== Все тесты пройдены успешно ===")

    except AssertionError as e:
        print(f"Ошибка автотеста: {e}")


def validate_data_configs():
    for task_type, datasets_info in DATASET_CONFIG.items():
        for dataset_name in datasets_info.keys():
            print(f"Проверка размерности '{dataset_name}'", end="... ", flush=True)
            dataset = datasets_info[dataset_name]
            X_train, X_test, y_train, y_test = dataset['DatasetPreparer']()
            dataset_autotester(X_train, X_test, y_train, y_test)

            if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
                print(f"⭕ Размерности X и y не совпадают: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
                return False
            print("Проверка пройдена ✅")

    for task_type, datasets_info in DATASET_CONFIG.items():
        for dataset_name in datasets_info.keys():
            if dataset_name not in DATASET_ARCHITECTURE_CONFIGS:
                print(f"Параметры сети для '{dataset_name}' не найдены в DATASET_ARCHITECTURE_CONFIGS ⭕")
                return False
            else:
                print(f"'{dataset_name}' найден в DATASET_ARCHITECTURE_CONFIGS ✅")
    
    for task_type, datasets_info in DATASET_CONFIG.items():
        for dataset_name in datasets_info.keys():
            print(f"Проверка '{dataset_name}'", end="... ", flush=True)
            dataset = datasets_info[dataset_name]
            X_train, _, _, _ = dataset['DatasetPreparer']()

            model_data = DATASET_ARCHITECTURE_CONFIGS[dataset_name]

            if model_data["type"] != task_type:
                print(f"⭕ Тип задачи '{task_type}' не совпадает с типом задачи '{model_data['type']}' для '{dataset_name}'")
                return False

            try:
                with torch.no_grad():
                    model_no   = create_model(dataset_name, "no")
                    model_sure = create_model(dataset_name, "sure")
                    model_huge = create_model(dataset_name, "huge")

                    model_no = model_no.eval()
                    model_sure = model_sure.eval()
                    model_huge = model_huge.eval()

                    sample = torch.from_numpy(X_train[[0,1]]).float()
                    if len(sample.shape) == 1:
                        sample = sample.unsqueeze(0)

                    model_no(sample)
                    model_sure(sample)
                    model_huge(sample)
            except Exception as e:
                print(f"⭕ Ошибка: {str(e)}")
                print(traceback.format_exc())
                return False
            print("Проверка пройдена ✅")
    
    return True


if __name__ == '__main__':
    validate_data_configs()
