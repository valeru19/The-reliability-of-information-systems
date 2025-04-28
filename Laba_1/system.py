from scipy.stats import uniform, truncnorm, triang
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

# Численное дифференцирование методом центральных разностей
def numerical_derivative(func, x, dx=1e-5):
    """
    Вычисляет численное значение производной функции func в точке x методом центральных разностей.
    Аргументы:
        func: Функция для дифференцирования
        x: Точка, в которой вычисляется производная
        dx: Шаг дифференцирования
    Возвращает:
        Приблизительное значение производной
    """
    return (func(x + dx) - func(x - dx)) / (2 * dx)

# Функция надежности (survival function) для равномерного распределения
def uniform_sf(x, a=30, b=1500):
    """
    Вычисляет функцию надежности (1 - CDF) для равномерного распределения.
    Аргументы:
        x: Входное значение
        a: Нижняя граница (по умолчанию 30)
        b: Верхняя граница (по умолчанию 1500)
    Возвращает:
        Вероятность безотказной работы
    """
    return uniform.sf(x, loc=a, scale=b - a)

# Функция надежности для усеченного нормального распределения
def truncnorm_sf(x, mu=385, sigma=np.sqrt(8649)):
    """
    Вычисляет функцию надежности для усеченного нормального распределения.
    Аргументы:
        x: Входное значение
        mu: Среднее значение (по умолчанию 385)
        sigma: Среднеквадратическое отклонение (по умолчанию sqrt(8649))
    Возвращает:
        Вероятность безотказной работы
    """
    return truncnorm.sf(x, (0 - mu) / sigma, np.inf, loc=mu, scale=sigma)

# Функция надежности для треугольного распределения
def triang_sf(x, a=45, b=6000):
    """
    Вычисляет функцию надежности для треугольного распределения.
    Аргументы:
        x: Входное значение
        a: Нижняя граница (по умолчанию 45)
        b: Верхняя граница (по умолчанию 6000)
    Возвращает:
        Вероятность безотказной работы
    """
    return triang.sf(x, 0.5, loc=a, scale=b - a)

# Комбинированная функция надежности (произведение индивидуальных функций)
def combined_sf(x):
    """
    Вычисляет комбинированную функцию надежности как произведение функций для равномерного,
    усеченного нормального и треугольного распределений.
    Аргументы:
        x: Входное значение
    Возвращает:
        Комбинированная вероятность безотказной работы
    """
    S_uniform = uniform_sf(x)
    S_truncnorm = truncnorm_sf(x)
    S_triang = triang_sf(x)
    return S_uniform * S_truncnorm * S_triang

# Плотность вероятности (PDF) через численное дифференцирование
def combined_pdf(x, dx=1e-5):
    """
    Вычисляет плотность вероятности как производную от (1 - SF).
    Аргументы:
        x: Входное значение
        dx: Шаг для численного дифференцирования
    Возвращает:
        Значение плотности вероятности
    """
    return numerical_derivative(lambda t: 1 - combined_sf(t), x, dx)

# Интенсивность отказов (hazard rate)
def hazard_rate(x):
    """
    Вычисляет интенсивность отказов как pdf(x) / sf(x).
    Аргументы:
        x: Входное значение
    Возвращает:
        Интенсивность отказов (или np.inf, если sf(x) слишком мало)
    """
    pdf = combined_pdf(x)
    sf = combined_sf(x)
    return np.where(sf > 1e-10, pdf / sf, np.inf)

# Гамма-процентная наработка
def gamma(gamma, t_min=30, t_max=6000):
    """
    Находит время t, при котором вероятность отказа равна gamma/100.
    Аргументы:
        gamma: Процент (от 0 до 100)
        t_min: Минимальное время (по умолчанию 30)
        t_max: Максимальное время (по умолчанию 6000)
    Возвращает:
        Время t или np.nan, если решение не найдено
    """
    target = gamma / 100
    func = lambda t: (1 - combined_sf(t)) - target
    try:
        return brentq(func, t_min, t_max)
    except ValueError:
        return np.nan

# Построение графиков надежности
def plot_graphs():
    """
    Строит четыре графика: плотность вероятности, функция надежности,
    гамма-процентная наработка и интенсивность отказов.
    Также вычисляет и выводит математическое ожидание и дисперсию.
    """
    t_values = np.linspace(0, 6000, 100)
    f_values = [combined_pdf(t) for t in t_values]
    P_values = [combined_sf(t) for t in t_values]

    # Вычисляем математическое ожидание (среднее время безотказной работы)
    system_mean_time, _ = quad(combined_sf, 0, 6000)

    # Вычисляем дисперсию: Var(X) = E[X^2] - (E[X])^2
    # E[X^2] = 2 * integral(t * sf(t), 0, inf)
    moment2, _ = quad(lambda t: 2 * t * combined_sf(t), 0, 6000)
    system_variance = moment2 - system_mean_time**2

    lambda_values = [hazard_rate(t) for t in t_values]
    gamma_labels = list(range(0, 110, 10))
    gamma_values = [gamma(g) for g in gamma_labels]
    gamma_values[0] = 0

    print(f"Математическое ожидание: {system_mean_time:.2f}")
    print(f"Дисперсия: {system_variance:.2f}, среднеквадратическое отклонение: {np.sqrt(system_variance):.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # График 1: Плотность вероятности f(t)
    axs[0, 0].plot(t_values, f_values, label="f(t)", color="tab:blue")
    axs[0, 0].set_xlabel("Время t")
    axs[0, 0].set_ylabel("f(t)")
    axs[0, 0].set_title("Плотность распределения f(t)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # График 2: Вероятность безотказной работы P(t)
    axs[0, 1].plot(t_values, P_values, label="P(t)", color="tab:green")
    axs[0, 1].set_xlabel("Время t")
    axs[0, 1].set_ylabel("P(t)")
    axs[0, 1].set_title("Вероятность безотказной работы P(t)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # График 3: Гамма-процентная наработка T_gamma
    axs[1, 0].plot(gamma_labels, gamma_values[::-1], "ko-", label="γ")
    axs[1, 0].set_xlabel("γ, %")
    axs[1, 0].set_ylabel("Время γ")
    axs[1, 0].set_title("Гамма-процентная наработка")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # График 4: Интенсивность отказов λ(t)
    axs[1, 1].plot(t_values, lambda_values, label="λ(t)", color="tab:red")
    axs[1, 1].set_xlabel("Время t")
    axs[1, 1].set_ylabel("λ(t)")
    axs[1, 1].set_title("Интенсивность отказов λ(t)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

# Запуск функции построения графиков
if __name__ == "__main__":
    plot_graphs()