import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_vector():
    """
    Создать массив от 0 до 9.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.arange.html

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    # Подсказка: используйте np.arange(10)
    return np.arange(10)


def create_matrix():
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Изучить:
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    # Подсказка: используйте np.random.rand(5,5)
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """
    Преобразовать (10,) -> (2,5)

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

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


def transpose_matrix(mat):
    """
    Транспонирование матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.transpose.html

    Args:
        mat (numpy.ndarray): Входная матрица

    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    # Подсказка: используйте mat.T или np.transpose(mat)
    if isinstance(mat, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if len(mat.shape) != 2:
        raise ValueError("Вход должен быть 2D матрицей")
    return mat.T


def vector_add(a, b):
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    # Подсказка: используйте оператор +
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы должны быть одинаковой формы")
    return a + b


def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число.

    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    # Подсказка: используйте оператор *
    if isinstance(vec, np.ndarray) == False:
        raise ValueError("vec должен быть numpy массивом")
    if not isinstance(scalar, (int, float)):
        raise ValueError("scalar должен быть числом")
    return vec * scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение.

    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица

    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    # Подсказка: используйте оператор *
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы/матрицы должны быть одинаковой формы")
    return a * b


def dot_product(a, b):
    """
    Скалярное произведение.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.dot.html

    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор

    Returns:
        float: Скалярное произведение векторов
    """
    # Подсказка: используйте np.dot(a, b)
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape != b.shape:
        raise ValueError("Векторы должны быть одинаковой формы")
    return a * b


def matrix_multiply(a, b):
    """
    Умножение матриц.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица

    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    # Подсказка: используйте a @ b или np.matmul(a, b)
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Число столбцов первой матрицы должно быть равно числу строк второй матрицы")
    return a @ b


def matrix_determinant(a):
    """
    Определитель матрицы.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        float: Определитель матрицы
    """
    # Подсказка: используйте np.linalg.det(a)
    if isinstance(a, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Обратная матрица.

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html

    Args:
        a (numpy.ndarray): Квадратная матрица

    Returns:
        numpy.ndarray: Обратная матрица
    """
    # Подсказка: используйте np.linalg.inv(a)
    if isinstance(a, np.ndarray) == False:
        raise ValueError("Вход должен быть numpy массивом")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    if np.linalg.det(a) == 0:
        raise ValueError("Матрица вырожденная, обратной не существует")
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решить систему Ax = b

    Изучить:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b

    Returns:
        numpy.ndarray: Решение системы x
    """
    # Подсказка: используйте np.linalg.solve(a, b)
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise ValueError("Оба входа должны быть numpy массивами")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")
    return np.linalg.solve(a, b)


def load_dataset(path="data/students_scores.csv"):
    """
    Загрузить CSV и вернуть NumPy массив.

    Args:
        path (str): Путь к CSV файлу

    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    # Подсказка: используйте pd.read_csv(path).to_numpy()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл по пути {path} не найден")
    return pd.read_csv(path).to_numpy()

