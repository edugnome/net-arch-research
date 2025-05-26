import torch
import datetime
import os
from typing import TextIO
from consist_config_tester import validate_data_configs
from trainers import ClassificationTrainer, RegressionTrainer, Trainer
from datasets import DATASET_CONFIG

torch.cuda.set_per_process_memory_fraction(0.75, device=0)


# Функция для запуска обучения для всех датасетов
def run_all_datasets_training():
    # Получаем текущую дату и время
    execution_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
    results_dir = os.path.join(save_dir, f"{execution_date}")
    results_file_path = os.path.join(results_dir, "common_results.txt")

    os.makedirs(results_dir, exist_ok=True)

    with open(results_file_path, "w") as results_file:
        for dataset_type, datasets_info in DATASET_CONFIG.items():
            if dataset_type == "classification":
                trainer_cls = ClassificationTrainer
            elif dataset_type == "regression":
                trainer_cls = RegressionTrainer
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            for dataset_name in datasets_info.keys():
                start_new_dataset_log(results_file, dataset_type, dataset_name)

                trainer: Trainer = trainer_cls(
                    dataset_name=dataset_name, batch_size=4, datetime_str=execution_date
                )
                result = trainer.train_all_variants()

                dataset_evaluation_results_log(results_file, dataset_type, dataset_name, result)


def dataset_evaluation_results_log(
    results_file: TextIO, dataset_type: str, dataset_name: str, result: dict
) -> None:
    timestamp: str = get_current_timestamp()
    message = f"Dataset: {dataset_type} '{dataset_name}', Result: {result}"
    log_line = f"{timestamp:<26} - {message}"
    results_file.write(log_line + "\n")
    print(log_line)


def start_new_dataset_log(results_file: TextIO, dataset_type: str, dataset_name: str) -> None:
    timestamp: str = get_current_timestamp()
    message = f"Starting training for dataset '{dataset_name}' ({dataset_type})"
    log_line = f"{timestamp:<26} - {message}"
    results_file.write(log_line + "\n")
    print(log_line)


def get_current_timestamp() -> str:
    return datetime.datetime.now().isoformat()


if __name__ == "__main__":
    if validate_data_configs():
        run_all_datasets_training()
