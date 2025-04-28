import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Параметры усеченного нормального распределения
mu = 385
sigma = 8649**(1/2)
a, b = (0 - mu) / sigma, np.inf

# Создание объекта усеченного нормального распределения
dist = truncnorm(a, b, loc=mu, scale=sigma)

# Генерация значений для графика на более широком интервале
x = np.linspace(0, 1000, 100)
pdf = dist.pdf(x) # Плотность распределения
df = dist.sf(x)  # Вероятность безотказной работы
cdf = np.ones(100) - df # Вероятность отказа
hazard = pdf / df  # Интенсивность отказов

# 1. Вероятность безотказной работы
reliability = df

# 2. Средняя наработка до отказа
mean_time_to_failure = dist.mean()

# 3. Среднее квадратическое отклонение и дисперсия времени безотказной работы
std_deviation = dist.std()
variance = std_deviation ** 2

# 4. Интенсивность отказов
intensity_of_failures = hazard

# 5. Плотность распределения времени до отказа
density_of_failure_time = pdf

# 6. Гамма-процентную наработку до отказа
gamma_values = np.arange(0, 1.01, 0.01)
quantiles = dist.ppf(gamma_values)
print(quantiles)

# Вывод результатов
print("Вероятность безотказной работы:")
print(reliability)

print("\nСредняя наработка до отказа:")
print(mean_time_to_failure)

print("\nСреднее квадратическое отклонение и дисперсия времени безотказной работы:")
print(f"СКО: {std_deviation}")
print(f"Дисперсия: {variance}")

print("\nИнтенсивность отказов:")
print(intensity_of_failures)

print("\nПлотность распределения времени до отказа:")
print(density_of_failure_time)

print("\nГамма-процентные наработки до отказа:")
for gamma, quantile in zip(gamma_values, quantiles):
    print(f"Гамма = {gamma * 100:.0f}%, t_γ = {quantile:.2f}")

# Построение графиков
plt.figure(figsize=(14, 10))

# График плотности распределения
plt.subplot(2, 3, 1)
plt.plot(x, pdf, 'r-', lw=2, label='PDF')
plt.axvline(mu, color='k', linestyle='--', label='μ')  # Добавляем вертикальную линию в точке μ
plt.axvline(mu + sigma, color='b', linestyle='--', label='μ + σ')  # Добавляем вертикальную линию в точке μ + σ
plt.axvline(mu - sigma, color='b', linestyle='--', label='μ - σ')  # Добавляем вертикальную линию в точке μ - σ
plt.title('Плотность распределения')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend()

# Вероятность работы
plt.subplot(2, 3, 2)
plt.plot(x, df, 'b-', lw=2, label='Вероятность работы ')
plt.title('Вероятность работы ')
plt.xlabel('t')
plt.ylabel('p(t)')
plt.legend()

# График кумулятивной функции распределения
plt.subplot(2, 3, 3)
plt.plot(x, cdf, 'b-', lw=2, label='Функция отказов')
plt.title('Функция отказов')
plt.xlabel('t')
plt.ylabel('F(t)')
plt.legend()

# График интенсивности отказов
plt.subplot(2, 3, 4)
plt.plot(x, hazard, 'g-', lw=2, label='Интенсивность отказов')
plt.title('Интенсивность отказов')
plt.xlabel('t')
plt.ylabel('λ(t)')
plt.legend()

list = []
for i in range(1, len(gamma_values) + 1):
    list.append(gamma_values[-i]*100)

# График гамма-процентных наработок до отказа
plt.subplot(2, 3, 5)
plt.plot(list, quantiles, 'm-', lw=2, label='Гамма-процентные наработка')
plt.title('Гамма-процентные наработки до отказа')
plt.xlabel('γ (%)')
plt.ylabel('t_γ')
plt.legend()

plt.tight_layout()
plt.show()