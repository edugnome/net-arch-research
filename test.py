import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import multiprocessing
import os

os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

# Пример простой модели
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

# Функция для обучения модели
def train_model(model_id, model, data_loader, max_epochs=5):
    logger = TensorBoardLogger("tb_logs", name=f"model_{model_id}")
    trainer = Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(model, data_loader)
    print(f"Model {model_id} training completed.")

# Создание данных
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Создание моделей
models = [SimpleModel() for _ in range(4)]

# Создание процессов для параллельного обучения
processes = []
for i in range(4):
    p = multiprocessing.Process(target=train_model, args=(i, models[i], data_loader))
    processes.append(p)
    p.start()

# Ожидание завершения всех процессов
for p in processes:
    p.join()