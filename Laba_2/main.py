import numpy as np
import matplotlib.pyplot as plt

# Исходные данные:
# n_values - количество отказов в каждом временном интервале
# delta_t - длина временного интервала (10 единиц времени)
# t_intervals - массив временных интервалов от 0 до 100 с шагом delta_t
# N - общее количество отказов (сумма всех n_values)

n_values = [78, 101, 14, 26, 138, 65, 8, 15, 73, 86]
delta_t = 10
t_intervals = np.arange(0, 100, delta_t)
N = np.sum(n_values)

# Расчет показателей надежности:

# N_values - количество оставшихся работоспособных объектов в начале каждого интервала
# N_1_2_values - среднее количество работоспособных объектов в интервале
# P_t - вероятность безотказной работы (вероятность, что объект проработает до начала интервала)
# f_t - плотность распределения времени до отказа
# lambda_t - интенсивность отказов

N_values = []
N_1_2_values = []
P_t = []
f_t = []
lambda_t = []

# Инициализация первых значений
N_values.append(N)  # В начале все объекты работоспособны
P_t.append(1)  # Вероятность безотказной работы в начале = 1
f_t.append(n_values[0] / (N*delta_t))  # Плотность отказов в первом интервале

# Расчет показателей для каждого интервала
for i in range(len(n_values) - 1):
    N_values.append(N - np.sum(n_values[0:i+1]))  # Оставшиеся объекты
    P_t.append(N_values[i+1] / N)  # Вероятность безотказной работы
    f_t.append(n_values[i+1] / (N*delta_t))  # Плотность отказов
    N_1_2_values.append((N_values[-2] + N_values[-1]) / 2)  # Среднее количество объектов в интервале
    lambda_t.append(n_values[i] / (N_1_2_values[i] * delta_t))  # Интенсивность отказов

# Добавляем последние значения
N_1_2_values.append(N_values[-1]/2)
lambda_t.append(n_values[-1] / (N_1_2_values[-1] * delta_t))

# Вывод промежуточных результатов
print("Количество оставшихся объектов в начале интервалов:", N_values)
print("Среднее количество объектов в интервалах:", N_1_2_values)

# Расчет средней наработки до отказа (T_mean)
print("\nРасчет средней наработки до отказа:")
for t,n in zip(t_intervals, n_values):
    print(f"Время: {t}, Отказы: {n}, Произведение: {t*n}")
total_time = sum(t * n for t, n in zip(t_intervals, n_values))
print("Суммарное время наработки:", total_time)
T_mean = total_time / N  # Средняя наработка до отказа

# Расчет дисперсии и среднеквадратического отклонения
print("\nРасчет дисперсии:")
for t,n in zip(t_intervals, n_values):
    print(f"Время: {t}, Отказы: {n}, Квадрат отклонения: {n * (t - T_mean)**2}")
D_T = sum(n * (t - T_mean)**2 for t, n in zip(t_intervals, n_values)) / (N - 1)  # Дисперсия
sigma_T = np.sqrt(D_T)  # Среднеквадратическое отклонение

# Вывод основных результатов
print(f"\nСредняя наработка до отказа: {T_mean:.2f}")
print(f"Дисперсия времени безотказной работы: {D_T:.2f}")
print(f"Среднеквадратическое отклонение: {sigma_T:.2f}")

# Функция для добавления подписей к столбцам на графиках
def add_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.annotate(f'{value:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

# Построение графиков
plt.figure(figsize=(18, 5))  # Создаем фигуру с тремя графиками в ряд

# 1. График вероятности безотказной работы
plt.subplot(1, 3, 1)
bars = plt.bar(t_intervals, P_t, width=delta_t, align='edge', color='skyblue', edgecolor='black')
add_labels(bars, P_t)
plt.title('Вероятность безотказной работы P(t)')
plt.xlabel('Время')
plt.ylabel('Вероятность')
plt.grid(True, linestyle='--', alpha=0.7)

# 2. График интенсивности отказов
plt.subplot(1, 3, 2)
bars = plt.bar(t_intervals, lambda_t, width=delta_t, align='edge', color='red', edgecolor='black')
add_labels(bars, lambda_t)
plt.title('Интенсивность отказов λ(t)')
plt.xlabel('Время')
plt.ylabel('Интенсивность отказов')
plt.grid(True, linestyle='--', alpha=0.7)

# 3. График плотности распределения отказов
plt.subplot(1, 3, 3)
bars = plt.bar(t_intervals, f_t, width=delta_t, align='edge', color='purple', edgecolor='black')
add_labels(bars, f_t)
plt.title('Плотность распределения отказов f(t)')
plt.xlabel('Время')
plt.ylabel('Плотность отказов')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()  # Автоматическая настройка расположения графиков
plt.show()