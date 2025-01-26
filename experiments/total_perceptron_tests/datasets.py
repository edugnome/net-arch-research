#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Дополненный модуль с конфигурациями для множества датасетов (как реальных, так и
сгенерированных при помощи различных функций из sklearn), разделённых на задачи
классификации и регрессии. Каждый датасет имеет:
  1) Количество итераций обучения (iterations)
  2) Функцию потерь (loss_function) с параметрами
  3) Оптимизатор (optimizer) с параметрами
  4) Специальный хэндлер для подготовки данных (DatasetPreparer), который:
       - Генерирует/скачивает/загружает датасет
       - Делает необходимую предобработку (стандартизация/нормализация и т.п.)
       - Делит данные на train/test
       - Возвращает (X_train, y_train, X_test, y_test)
  5) Поле temp (резервное)
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Для реальных и классических датасетов из sklearn
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    load_diabetes,
    make_classification,
    make_regression,
    make_blobs,
    # Новые импортируемые функции:
    make_biclusters,
    make_checkerboard,
    make_circles,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_s_curve,
    make_sparse_spd_matrix,
    make_sparse_uncorrelated,
    make_spd_matrix,
    make_swiss_roll
)

# Для примеров из torchvision (MNIST, CIFAR и т.д.)
import torchvision
import torchvision.transforms as T

################################################################################
#                     ПОДГОТОВИТЕЛЬНЫЕ ФУНКЦИИ (DatasetPreparer)               #
################################################################################

# ---------------------- Примеры подготовки torchvision-датасетов -------------
def prepare_mnist_data():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               download=True,
                                               transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              download=True,
                                              transform=transform)

    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


def prepare_fashion_mnist_data():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                      train=True,
                                                      download=True,
                                                      transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                     train=False,
                                                     download=True,
                                                     transform=transform)

    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


def prepare_cifar10_data():
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


def prepare_cifar100_data():
    transform = T.Compose([
        T.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                  train=True,
                                                  download=True,
                                                  transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data',
                                                 train=False,
                                                 download=True,
                                                 transform=transform)
    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


def prepare_kmnist_data():
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1904,), (0.3475,))
    ])
    train_dataset = torchvision.datasets.KMNIST(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
    test_dataset = torchvision.datasets.KMNIST(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform)
    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


def prepare_emnist_data():
    transform = T.Compose([
        T.ToTensor(),
        # Можно добавить более точную нормализацию
    ])
    train_dataset = torchvision.datasets.EMNIST(root='./data',
                                                split='balanced',
                                                train=True,
                                                download=True,
                                                transform=transform)
    test_dataset = torchvision.datasets.EMNIST(root='./data',
                                               split='balanced',
                                               train=False,
                                               download=True,
                                               transform=transform)

    X_train_list, y_train_list = [], []
    for img, label in train_dataset:
        X_train_list.append(img.view(-1).numpy())
        y_train_list.append(label)
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int64)

    X_test_list, y_test_list = [], []
    for img, label in test_dataset:
        X_test_list.append(img.view(-1).numpy())
        y_test_list.append(label)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    return X_train, X_test, y_train, y_test


# ------------------ Классические датасеты из sklearn (classification) ---------
def prepare_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_wine_data():
    data = load_wine()
    X, y = data.data, data.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_digits_data():
    data = load_digits()
    X, y = data.data, data.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_spambase_data():
    # Предположим, CSV с признаком (последняя колонка - target) лежит в ./data/spambase.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/spambase.csv"))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# --------------------- Искусственные датасеты из sklearn ---------------------
def prepare_make_classification_data():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=5,
                               n_classes=2,
                               random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_blobs_data():
    X, y = make_blobs(n_samples=1000,
                      centers=3,
                      n_features=2,
                      random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------- Реальные датасеты (Titanic, Adult, Fraud) -----------------
def prepare_titanic_data():
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df = df.dropna(subset=["sex", "age", "fare", "class", "survived"])
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["class"] = df["class"].map({"Third": 0, "Second": 1, "First": 2})

    X = df[["sex", "age", "fare", "class"]].values
    y = df["survived"].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_adult_income_data():
    # Предполагаем, что CSV лежит в ./data/adult.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/adult.csv"))
    df = df.dropna()
    df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    X = df[["age", "educational-num", "gender", "hours-per-week"]].values
    y = df["income"].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_credit_card_fraud_data():
    # Предполагаем, что CSV лежит в ./data/creditcard.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/creditcard.csv"))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------- Классические датасеты (регрессия) -------------------
def prepare_diabetes_data():
    data = load_diabetes()
    X, y = data.data, data.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------ Искусственные регрессионные датасеты ---------------------
def prepare_make_regression_data():
    X, y = make_regression(n_samples=1000,
                           n_features=10,
                           n_informative=7,
                           noise=10.0,
                           random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------- Прочие регрессионные датасеты из UCI -------------------
def prepare_energy_efficiency_data():
    # Предполагаем, что CSV лежит в ./data/energy_efficiency.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/energy_efficiency.csv"))
    # Целевая переменная: 'Y1' (Heating Load), для примера
    X = df.drop(columns=["Y1"]).values
    y = df["Y1"].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_airfoil_data():
    # Предполагаем, что TSV лежит в ./data/airfoil_self_noise.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/airfoil_self_noise.csv"), sep="\t", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_concrete_data():
    # Предполагаем, что CSV лежит в ./data/concrete_data.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/concrete_data.csv"))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_house_prices_data():
    # Предполагаем, что CSV лежит в ./data/house_prices.csv
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/house_prices.csv"))
    df = df.dropna(subset=["SalePrice"])
    numeric_cols = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]
    df = df.dropna(subset=numeric_cols)
    X = df[numeric_cols].values
    y = df["SalePrice"].values
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------------------------------------------------------
#                 НОВЫЕ ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ (из sklearn.samples)         
# -----------------------------------------------------------------------------

########################
# 1) CLASSIFICATION
########################

def prepare_make_biclusters_data():
    """
    make_biclusters: обычно генерирует матрицу с biclusters и матрицы признаков rows, cols.
    Для демонстрации будем брать argmax по строкам в 'rows' как метку класса (упрощённо).
    """
    X, rows, cols = make_biclusters(shape=(200, 5), n_clusters=2, random_state=42)
    # X.shape = (200, 5)
    # rows.shape = (2, 200) -> признак принадлежности каждого из 200 образцов к 2 кластерам
    X = X.astype(np.float32)
    # Упрощённо делаем y = argmax по rows
    y = np.argmax(rows, axis=0).astype(np.int64)
    # Стандартизация
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_checkerboard_data():
    """
    make_checkerboard: аналогично biclusters, создаём "шахматную" структуру.
    """
    X, rows, cols = make_checkerboard(shape=(200, 5), n_clusters=2, random_state=42)
    X = X.astype(np.float32)
    # Упрощённо берём y = argmax(rows)
    y = np.argmax(rows, axis=0).astype(np.int64)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_circles_data():
    """
    make_circles: генерирует 2D круги для бинарной классификации (0/1).
    """
    X, y = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_gaussian_quantiles_data():
    """
    make_gaussian_quantiles: генерирует выборки, разбитые квантилями.
    """
    X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_hastie_10_2_data():
    """
    make_hastie_10_2: датасет с 10 признаками, y в {-1, +1}.
    Преобразуем метку в {0,1}.
    """
    X, y = make_hastie_10_2(n_samples=1000, random_state=42)
    # y \in {-1, +1} -> переводим в {0,1}
    y = ((y + 1) // 2).astype(np.int64)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_moons_data():
    """
    make_moons: двумерные "полулуния" для бинарной классификации.
    """
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_multilabel_classification_data():
    """
    make_multilabel_classification: создаёт мульти-меточные данные.
    Для простоты возвращаем y как есть, хотя для классического 'CrossEntropyLoss'
    обычно требуется иной подход (или BCEWithLogitsLoss).
    """
    X, Y = make_multilabel_classification(n_samples=500,
                                          n_features=10,
                                          n_classes=3,
                                          random_state=42)
    # Y.shape = (500, 3) (каждый пример может иметь несколько меток)
    # Упрощённо попытаемся сконвертировать в "одну метку" через argmax (не совсем корректно)
    # или можно оставить как есть и подобрать BCEWithLogitsLoss. 
    # Здесь для единообразия сделаем argmax:
    y = np.argmax(Y, axis=1).astype(np.int64)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


########################
# 2) REGRESSION
########################

def prepare_make_friedman1_data():
    """
    make_friedman1: классический фридмановский синтетический датасет для регрессии.
    """
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_friedman2_data():
    """
    make_friedman2: второй фридмановский синтетический датасет.
    """
    X, y = make_friedman2(n_samples=1000, noise=1.0, random_state=42)
    # X.shape = (1000, 4), y.shape = (1000,)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_friedman3_data():
    """
    make_friedman3: третий фридмановский датасет.
    """
    X, y = make_friedman3(n_samples=1000, noise=1.0, random_state=42)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_low_rank_matrix_data():
    """
    make_low_rank_matrix: генерирует матрицу (без меток). 
    Для демонстрации сделаем y = сумма по строке.
    """
    X = make_low_rank_matrix(n_samples=500, n_features=10, effective_rank=5, random_state=42)
    y = X.sum(axis=1)  # условная регрессия: сумма значений в строке
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_s_curve_data():
    """
    make_s_curve: генерирует 3D "S"-образную поверхность + t (как параметр).
    Возьмём y = t в качестве регрессионной цели.
    """
    X, t = make_s_curve(n_samples=1000, noise=0.1, random_state=42)
    # X.shape = (1000, 3), t.shape=(1000,)
    X = X.astype(np.float32)
    t = t.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, t, test_size=0.2, random_state=42)


def prepare_make_sparse_spd_matrix_data():
    """
    make_sparse_spd_matrix: возвращает матрицу (n x n). 
    Нет прямой целевой переменной, используем, к примеру, сумму всех элементов в качестве y.
    Затем "распрямляем" матрицу в вектор.
    """
    M = make_sparse_spd_matrix(n_dim=10, alpha=0.95, random_state=42)
    # M.shape = (10, 10)
    X = M.flatten().reshape(1, -1).astype(np.float32)  # (1, 100)
    y = np.array([M.sum()], dtype=np.float32)          # (1,)
    # У нас всего один пример? Это бессмысленно для train/test.
    # Для демо создадим несколько таких матриц в цикле:
    Xs, ys = [], []
    for i in range(100):
        M_i = make_sparse_spd_matrix(n_dim=10, alpha=0.95, random_state=42 + i)
        Xs.append(M_i.flatten())
        ys.append(M_i.sum())
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    scaler = StandardScaler().fit(Xs)
    Xs = scaler.transform(Xs)
    return train_test_split(Xs, ys, test_size=0.2, random_state=42)


def prepare_make_sparse_uncorrelated_data():
    """
    make_sparse_uncorrelated: создаёт регрессионную выборку с малым числом информативных признаков.
    """
    X, y = make_sparse_uncorrelated(n_samples=1000, n_features=10, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_make_spd_matrix_data():
    """
    make_spd_matrix: возвращает случайную SPD-матрицу (n x n).
    Аналогично make_sparse_spd_matrix, исскуственно делаем dataset.
    """
    # Создадим несколько матриц:
    Xs, ys = [], []
    for i in range(100):
        M = make_spd_matrix(n_dim=10, random_state=42 + i)
        Xs.append(M.flatten())
        ys.append(M.sum())
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    scaler = StandardScaler().fit(Xs)
    Xs = scaler.transform(Xs)
    return train_test_split(Xs, ys, test_size=0.2, random_state=42)


def prepare_make_swiss_roll_data():
    """
    make_swiss_roll: (n_samples, 3) + t. 
    Возьмём y = t как регрессию.
    """
    X, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    t = t.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return train_test_split(X, t, test_size=0.2, random_state=42)


################################################################################
#                                 DICT CONFIG                                  #
################################################################################

DATASET_CONFIG = {
    "classification": {
        # "MNIST": {
        #     "iterations": 10000,
        #     "loss_function": {
        #         "type": nn.CrossEntropyLoss,
        #         "args": {}
        #     },
        #     "optimizer": {
        #         "type": optim.SGD,
        #         "args": {
        #             "lr": 0.01,
        #             "momentum": 0.9
        #         }
        #     },
        #     "DatasetPreparer": prepare_mnist_data,
        #     "temp": None
        # },
        "Fashion-MNIST": {
            "iterations": 10000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_fashion_mnist_data,
            "temp": None
        },
        # "CIFAR-10": {
        #     "iterations": 15000,
        #     "loss_function": {
        #         "type": nn.CrossEntropyLoss,
        #         "args": {}
        #     },
        #     "optimizer": {
        #         "type": optim.SGD,
        #         "args": {
        #             "lr": 0.01,
        #             "momentum": 0.9
        #         }
        #     },
        #     "DatasetPreparer": prepare_cifar10_data,
        #     "temp": None
        # },
        "CIFAR-100": {
            "iterations": 20000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_cifar100_data,
            "temp": None
        },
        "KMNIST": {
            "iterations": 10000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_kmnist_data,
            "temp": None
        },
        "EMNIST": {
            "iterations": 10000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_emnist_data,
            "temp": None
        },
        "Iris": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_iris_data,
            "temp": None
        },
        "Wine": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_wine_data,
            "temp": None
        },
        "Breast Cancer Wisconsin": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_breast_cancer_data,
            "temp": None
        },
        "Digits": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_digits_data,
            "temp": None
        },
        "SpamBase": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_spambase_data,
            "temp": None
        },
        "Make Classification": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_classification_data,
            "temp": None
        },
        "Make Blobs": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_blobs_data,
            "temp": None
        },
        "Titanic Dataset": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_titanic_data,
            "temp": None
        },
        "Adult Income": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_adult_income_data,
            "temp": None
        },
        "Credit Card Fraud Detection": {
            "iterations": 3000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_credit_card_fraud_data,
            "temp": None
        },
        # Новые (sklearn.samples) - classification
        "Make Biclusters": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_biclusters_data,
            "temp": None
        },
        "Make Checkerboard": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_checkerboard_data,
            "temp": None
        },
        "Make Circles": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_circles_data,
            "temp": None
        },
        "Make Gaussian Quantiles": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_gaussian_quantiles_data,
            "temp": None
        },
        "Make Hastie 10 2": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_hastie_10_2_data,
            "temp": None
        },
        "Make Moons": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_moons_data,
            "temp": None
        },
        "Make Multilabel Classification": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.CrossEntropyLoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_multilabel_classification_data,
            "temp": None
        },
    },
    "regression": {
        "Diabetes": {
            "iterations": 2000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_diabetes_data,
            "temp": None
        },
        "Energy Efficiency": {
            "iterations": 3000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_energy_efficiency_data,
            "temp": None
        },
        "Airfoil Self-Noise": {
            "iterations": 3000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_airfoil_data,
            "temp": None
        },
        "Concrete Compressive Strength": {
            "iterations": 3000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_concrete_data,
            "temp": None
        },
        "Make Regression": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_regression_data,
            "temp": None
        },
        "House Prices Dataset": {
            "iterations": 3000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_house_prices_data,
            "temp": None
        },
        # Новые (sklearn.samples) - regression
        "Make Friedman1": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_friedman1_data,
            "temp": None
        },
        "Make Friedman2": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_friedman2_data,
            "temp": None
        },
        "Make Friedman3": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_friedman3_data,
            "temp": None
        },
        "Make Low Rank Matrix": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_low_rank_matrix_data,
            "temp": None
        },
        "Make S Curve": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_s_curve_data,
            "temp": None
        },
        "Make Sparse SPD Matrix": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_sparse_spd_matrix_data,
            "temp": None
        },
        "Make Sparse Uncorrelated": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_sparse_uncorrelated_data,
            "temp": None
        },
        "Make SPD Matrix": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_spd_matrix_data,
            "temp": None
        },
        "Make Swiss Roll": {
            "iterations": 1000,
            "loss_function": {
                "type": nn.MSELoss,
                "args": {}
            },
            "optimizer": {
                "type": optim.SGD,
                "args": {
                    "lr": 0.001,
                    "momentum": 0.9
                }
            },
            "DatasetPreparer": prepare_make_swiss_roll_data,
            "temp": None
        }
    }
}

if __name__ == "__main__":
    for key, conf in DATASET_CONFIG["classification"].items():
        print("Checking classification", key)
        X_train, y_train, X_test, y_test = conf["DatasetPreparer"]()

    for key, conf in DATASET_CONFIG["regression"].items():
        print("Checking regression", key)
        X_train, y_train, X_test, y_test = conf["DatasetPreparer"]()

    # Короткая проверка для одного из новых датасетов классификации:
    conf = DATASET_CONFIG["classification"]["Make Circles"]
    X_train, y_train, X_test, y_test = conf["DatasetPreparer"]()
    print("Make Circles shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # И для одного из новых датасетов регрессии:
    conf_reg = DATASET_CONFIG["regression"]["Make Friedman1"]
    X_tr, y_tr, X_te, y_te = conf_reg["DatasetPreparer"]()
    print("Make Friedman1 shapes:", X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
