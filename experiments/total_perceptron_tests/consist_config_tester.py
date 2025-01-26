import torch
from datasets import DATASET_CONFIG
from models import DATASET_ARCHITECTURE_CONFIGS, create_model
import traceback


def validate_data_configs():
    for task_type, datasets_info in DATASET_CONFIG.items():
        for dataset_name in datasets_info.keys():
            print(f"Проверка размерности '{dataset_name}'", end="... ", flush=True)
            dataset = datasets_info[dataset_name]
            X_train, X_test, y_train, y_test = dataset['DatasetPreparer']()

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
