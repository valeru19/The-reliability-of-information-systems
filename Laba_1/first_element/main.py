import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Параметры равномерного распределения
a, b = 30, 1500

# 1. Функция надёжности: вероятность безотказной работы
def reliability_function(t):
    # Если t в интервале [a, b]: (b - t) / (b - a), если меньше a: 1, если больше b: 0
    return np.where((t >= a) & (t <= b), (b - t) / (b - a), np.where(t < a, 1, 0))

# Расчёт основных показателей:
mean_time = (a + b) / 2  # 2. Средняя наработка до отказа (матожидание)
variance = ((b - a) ** 2) / 12  # 3. Дисперсия времени безотказной работы
std_dev = np.sqrt(variance)  # 3. Среднеквадратическое отклонение
failure_rate = 1 / (b - a)  # 4. Интенсивность отказов

# Подготовка данных для графиков:
t_values = np.linspace(a - 100, b + 100, 1000)
reliability_values = reliability_function(t_values)
pdf_values = uniform.pdf(t_values, a, b - a)  # 5. Плотность распределения

# 6. Гамма-процентная наработка: определяем времена для γ = 0, 10, 20, …, 100%
gamma_values = np.arange(0, 101, 10)
gamma_life = a + (b - a) * gamma_values / 100

# Создаем фигуру с 6 субграфиками (3 строки x 2 столбца)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Вывод расчетных значений в консоль:
print(f'Средняя наработка до отказа: {mean_time:.2f}')
print(f'Дисперсия: {variance:.2f}, СКО: {std_dev:.2f}')
print(f'Интенсивность отказов: {failure_rate:.6f}')

# График 1: Функция надёжности
axs[0, 0].plot(t_values, reliability_values, color='green')
axs[0, 0].set_title('Функция надежности')
axs[0, 0].set_xlabel('Время')
axs[0, 0].set_ylabel('Вероятность безотказной работы')

# График 4: Интенсивность отказов (линейный график)
axs[0, 1].plot(t_values, np.full_like(t_values, failure_rate), color='red', linestyle='--')
axs[0, 1].set_title('Интенсивность отказов')
axs[0, 1].set_xlabel('Время')
axs[0, 1].set_ylabel('λ (отказов/ед. времени)')
axs[0, 1].set_ylim(0, failure_rate * 1.2)  # Чуть больше, чтобы была видна линия

# График 5: Плотность распределения времени до отказа (PDF)
axs[1, 0].plot(t_values, pdf_values, color='blue')
axs[1, 0].set_title('Плотность распределения (PDF)')
axs[1, 0].set_xlabel('Время')
axs[1, 0].set_ylabel('Плотность вероятности')

# График 6: Гамма-процентная наработка до отказа
axs[1, 1].plot(gamma_values, gamma_life, marker='o', linestyle='-', color='brown')
axs[1, 1].set_title('Гамма-процентная наработка до отказа')
axs[1, 1].set_xlabel('γ (%)')
axs[1, 1].set_ylabel('Время')

plt.tight_layout()
plt.show()