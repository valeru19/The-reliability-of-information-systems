import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#  ПАРАМЕТРЫ ТРЕУГОЛЬНОГО РАСПРЕДЕЛЕНИЯ
# --------------------------------------------------------------------------------
a, b = 45, 6000  # Границы интервала

# --------------------------------------------------------------------------------
#  1. ЧИСЛЕННЫЙ РАСЧЕТ M[T] (МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ)
# --------------------------------------------------------------------------------
def expected_value_simpson(a, b, steps=1000):
    """
    Вычисляет M[T] = ∫[a, b] t * f(t) dt
    для U(a, b): f(t)=1/(b-a) при t ∈ [a,b].
    Используется метод средних прямоугольников.
    """
    dt = (b - a) / steps
    integral_sum = 0.0
    for i in range(steps):
        t_mid = a + (i + 0.5) * dt  # середина подынтервала
        integral_sum += t_mid * pdf(t_mid)
    return integral_sum * dt

# --------------------------------------------------------------------------------
#  2. ЧИСЛЕННЫЙ РАСЧЕТ ВТОРОГО МОМЕНТА M[T^2]
# --------------------------------------------------------------------------------
def second_moment_simpson(a, b, steps=1000):
    """
    Вычисляет M[T^2] = ∫[a, b] t^2 * f(t) dt
    для U(a, b).
    """
    dt = (b - a) / steps
    integral_sum = 0.0
    for i in range(steps):
        t_mid = a + (i + 0.5) * dt
        integral_sum += (t_mid ** 2) * pdf(t_mid)
    return integral_sum * dt

# --------------------------------------------------------------------------------
#  3. ЧИСЛЕННЫЙ РАСЧЕТ PDF (ПЛОТНОСТИ)
# --------------------------------------------------------------------------------
def pdf(t):
    """Плотность для U(a,b)"""
    return 2.0 / (b - a) - 2.0 / (b - a)**2 * abs(a+b-2*t) if (a <= t <= b) else 0.0

# --------------------------------------------------------------------------------
#  4. ЧИСЛЕННЫЙ РАСЧЕТ F(t) (ФУНКЦИИ РАСПРЕДЕЛЕНИЯ)
# --------------------------------------------------------------------------------
def F_numerical(t, steps=1000):
    """
    Численно вычисляет F(t)=∫[a, t] f(u) du методом средних прямоугольников.
    - 0, если t <= a
    - 1, если t >= b
    """
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0

    dt = (t - a) / steps
    integral_sum = 0.0
    for i in range(steps):
        u_mid = a + (i + 0.5) * dt
        integral_sum += pdf(u_mid)
    return integral_sum * dt

# --------------------------------------------------------------------------------
#  5. ЧИСЛЕННЫЙ РАСЧЕТ ФУНКЦИИ НАДЕЖНОСТИ R(t) = 1 - F(t)
# --------------------------------------------------------------------------------
def R_numerical(t):
    return 1.0 - F_numerical(t)

# --------------------------------------------------------------------------------
#  6. ЧИСЛЕННЫЙ РАСЧЕТ ИНТЕНСИВНОСТИ ОТКАЗОВ (HAZARD RATE) λ(t) = f(t)/R(t)
# --------------------------------------------------------------------------------
def hazard_rate_numerical(t, h=1e-6):
    """
    Численно: f(t) ≈ [F(t+h) - F(t)] / h,
    R(t)=1-F(t).
    => λ(t)=f(t)/R(t).
    Возвращает np.nan, если t вне [a,b).
    """
    if t < a or t >= b:
        return np.nan

    f_t = pdf(t)
    R_t = R_numerical(t)

    if R_t <= 0:
        return np.nan
    return f_t / R_t


# --------------------------------------------------------------------------------
#  7. ЧИСЛЕННЫЙ РАСЧЕТ ГАММА-ПРОЦЕНТНОЙ НАРАБОТКИ
# --------------------------------------------------------------------------------
def gamma_life_integral(a, b, gamma, steps=1000):
    """
    Численный поиск Tγ из условия:
       ∫[a -> Tγ] f(t) dt = gamma/100.
    Используется метод средних прямоугольников.
    Увеличено число steps для плавности.
    """
    # При gamma=0 => Tγ=a, при gamma=100 => Tγ=b
    if gamma == 0:
        return a
    if gamma == 100:
        return b

    target_prob = gamma / 100
    dt = (b - a) / steps
    integral_sum = 0.0
    t_gamma = a

    for i in range(steps):
        # Середина i-го подынтервала
        t_mid = a + (i + 0.5) * dt
        # Прибавляем площадь маленького прямоугольника
        integral_sum += pdf(t_mid) * dt
        if integral_sum >= target_prob:
            t_gamma = t_mid
            break

    return t_gamma

# --------------------------------------------------------------------------------
#  ВЫЧИСЛЯЕМ ОСНОВНЫЕ ПОКАЗАТЕЛИ
# --------------------------------------------------------------------------------
mean_time = expected_value_simpson(a, b)
mean_square_time = second_moment_simpson(a, b)
variance = mean_square_time - mean_time**2
std_dev = variance**0.5

# --------------------------------------------------------------------------------
#  ВЫВОД ЗНАЧЕНИЙ В КОНСОЛЬ
# --------------------------------------------------------------------------------
print(f'Средняя наработка (MTTF, численно) : {mean_time:.4f}')
print(f'M[T^2] (численно)                  : {mean_square_time:.4f}')
print(f'Дисперсия (численно)               : {variance:.4f}')
print(f'СКО (численно)                     : {std_dev:.4f}')

# Выводим интенсивность отказов в нескольких точках:
test_points = [a, (a + b)/2, b - 1]
for tp in test_points:
    hr_val = hazard_rate_numerical(tp)
    print(f'λ(t={tp:.1f}) = {hr_val:.6f}')

# --------------------------------------------------------------------------------
#  ПОДГОТОВКА ДАННЫХ ДЛЯ ГРАФИКОВ
# --------------------------------------------------------------------------------
t_values_full = np.linspace(a - 100, b + 100, 100)

# Численная функция надёжности (для всего диапазона)
R_values = np.array([R_numerical(t) for t in t_values_full])

# Плотность (PDF)
pdf_values = [pdf(t) for t in t_values_full]

# Интенсивность отказов (hazard) — только на [a,b]
t_values_hazard = np.linspace(a, b, 1450)
hazard_values = np.array([hazard_rate_numerical(t) for t in t_values_hazard])

# Гамма-процентная наработка
gamma_values = np.arange(0, 101, 10)
gamma_life_numeric = [gamma_life_integral(a, b, g) for g in gamma_values]

# --------------------------------------------------------------------------------
#  ПОСТРОЕНИЕ ГРАФИКОВ
# --------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# (1) Функция надежности
axs[0, 0].plot(t_values_full, R_values, color='green')
axs[0, 0].set_title('Функция надежности (численно)')
axs[0, 0].set_xlabel('Время')
axs[0, 0].set_ylabel('R(t)')

# (2) Интенсивность отказов
axs[0, 1].plot(t_values_hazard, hazard_values, color='red', linestyle='--')
axs[0, 1].set_title('Интенсивность отказов (численно)')
axs[0, 1].set_xlabel('Время')
axs[0, 1].set_ylabel('λ(t)')
# Попробуйте логарифмическую шкалу, если хотите «сгладить» скачок:
# axs[0, 1].set_yscale('log')

# Установим ограничение по оси Y, чтобы лучше видеть график
# (Обрезаем верхнюю часть, т.к. около t=b интенсивность уходит к бесконечности)
max_hazard = np.nanmax(hazard_values)
axs[0, 1].set_ylim(0, max_hazard * 1.2 if not np.isnan(max_hazard) else 1)

# (3) Плотность распределения (PDF)
axs[1, 0].plot(t_values_full, pdf_values, color='blue')
axs[1, 0].set_title('Плотность распределения (PDF)')
axs[1, 0].set_xlabel('Время')
axs[1, 0].set_ylabel('f(t)')

# (4) Гамма-процентная наработка
axs[1, 1].plot(gamma_values, gamma_life_numeric[::-1], marker='o', color='brown')
axs[1, 1].set_title('Гамма-процентная наработка (численно)')
axs[1, 1].set_xlabel('γ (%)')
axs[1, 1].set_ylabel('Tγ')

plt.tight_layout()
plt.show()