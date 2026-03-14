import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_vector() -> np.ndarray:
    """
    Создает массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создает матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5, 5)


def reshape_vector(vec) -> np.ndarray:
    """
    Преобразует (10,) -> (2,5)

    Args:
        vec (numpy.ndarray): Входной массив формы (10,)

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    if vec.shape != (10,):
        raise ValueError("Вектор должен быть длинной (10,)")
    elif isinstance(vec, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    return vec.reshape(2,5)


def transpose_matrix(mat) -> np.ndarray:
    """
    Транспонирует матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    if isinstance(mat, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if len(mat.shape) != 2:
        raise ValueError("Вход должен быть 2D матрицей")
    return mat.T


def vector_add(a, b) -> np.ndarray:
    """
    Складывает вектора одинаковой длины.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы должны быть одинаковой формы")
    return a + b


def scalar_multiply(vec, scalar) -> np.ndarray:
    """
    Умножает вектор на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    if isinstance(vec, np.ndarray) == False:
        raise ValueError("vec должен быть numpy массивом")
    if not isinstance(scalar, (int, float)):
        raise ValueError("scalar должен быть числом")
    return vec * scalar


def elementwise_multiply(a, b) -> np.ndarray:
    """
    Поэлементно умножает.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы/матрицы должны быть одинаковой формы")
    return a * b


def dot_product(a, b) -> float:
    """
    Скалярное произведение.

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов
    """
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы должны быть одинаковой формы")
    return np.dot(a, b)


def matrix_multiply(a, b) -> np.ndarray:
    """
    Умножает матрицы.

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы")
    return a @ b


def matrix_determinant(a) -> float:
    """
    Определитель матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы
    """
    if isinstance(a, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    return np.linalg.det(a)


def matrix_inverse(a) -> np.ndarray:
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица
    """
    if isinstance(a, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    if np.linalg.det(a) == 0:
        raise ValueError("Матрица вырожденная, обратной не существует")
    return np.linalg.inv(a)


def solve_linear_system(a, b) -> np.ndarray:
    """
    Решает систему Ax = b

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x
    """
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")
    return np.linalg.solve(a, b)


def load_dataset(path="data/students_scores.csv") -> np.ndarray:
    """
    Загружает CSV и возвращает NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    # Подсказка: используйте pd.read_csv(path).to_numpy()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл по пути {path} не найден")
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data) -> dict:
    """
    Вычисляет статистические показатели: среднее, медиану, стандартное отклонение, минимум, максимум, 25-й и 75-й процентили.
    Args:
        data (numpy.ndarray): Одномерный массив данных

    Returns:
        dict: Словарь со статистическими показателями
    """
    return {"mean": np.mean(data), "median": np.median(data), "standart deviation": np.std(data),
            "min": np.min(data), "max": np.max(data),
            "25 percentiles": np.percentile(data, 25), "75 percentiles": np.percentile(data, 75)}


def normalize_data(data) -> np.ndarray:
    """
    Min-Max нормализация.

    Формула: (x - min) / (max - min)

    Args:
        data (numpy.ndarray): Входной массив данных

    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))




def plot_histogram(data):
    """
    Строит гистограмму распределения оценок по математике.
    """
    os.makedirs("lab_2/plots", exist_ok=True)

    plt.figure()
    plt.hist(data, bins=10)

    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценка")
    plt.ylabel("Количество студентов")

    plt.savefig("lab_2/plots/math_histogram.png")
    plt.close()


def plot_heatmap(matrix):
    """
    Строит тепловую карту корреляции предметов.
    """
    os.makedirs("lab_2/plots", exist_ok=True)

    plt.figure()
    sns.heatmap(matrix, annot=True, cmap="coolwarm")

    plt.title("Корреляция предметов")

    plt.savefig("lab_2/plots/correlation_heatmap.png")
    plt.close()


def plot_line(x, y):
    """
    Сроит график зависимости: студент -> оценка по математике.
    """
    os.makedirs("lab_2/plots", exist_ok=True)

    plt.figure()
    plt.plot(x, y, marker="o")

    plt.title("Оценки студентов по математике")
    plt.xlabel("Номер студента")
    plt.ylabel("Оценка")

    plt.savefig("lab_2/plots/math_line_plot.png")
    plt.close()


