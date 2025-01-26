import datetime
import os
from consist_config_tester import validate_data_configs
from trainers import Trainer
from datasets import DATASET_CONFIG


# Получаем текущую дату и время
execution_date = datetime.datetime.now().isoformat()
save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
        )
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results",
                           f"{execution_date}")
results_file_path = os.path.join(results_dir, "common_results.txt")

# Функция для запуска обучения для всех датасетов
def run_all_datasets_training():
    with open(results_file_path, "w") as results_file:
        for dataset_type, datasets_info in DATASET_CONFIG.items():
            for dataset_name in datasets_info.keys():
                print(f"Запуск обучения для датасета '{dataset_name}' ({dataset_type})")
                trainer = Trainer(dataset_name=dataset_name, 
                                dataset_type=dataset_type, 
                                datetime_str=execution_date)
                result = trainer.train_all_variants()
                results_file.write(f"Dataset: {dataset_type} '{dataset_name}', Result: {result}\n")
                print(f"Dataset: {dataset_type} '{dataset_name}', Result: {result}")


if __name__ == "__main__":
    if validate_data_configs():
        run_all_datasets_training()
