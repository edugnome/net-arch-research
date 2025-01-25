import sympy as sp

# Объявляем символы
x, a = sp.symbols('x a', real=True)
target = 0  # Таргет равен нулю

# Определяем функцию модели: y = a * sin(exp(x))
y = a * sp.sin(sp.exp(x))

# Определяем функцию потерь: loss = (y - target)^2
loss = (y - target)**2

# Вычисляем первую производную функции потерь по x
# d(loss)/dx = 2 * y * dy/dx
dy_dx = sp.diff(y, x)
loss_dx = 2 * y * dy_dx

# Вычисляем вторую производную функции потерь по x
# d^2(loss)/dx^2 = 2 * (dy/dx)^2 + 2 * y * d^2y/dx^2
d2y_dx2 = sp.diff(dy_dx, x)
loss_dxx = 2 * (dy_dx**2) + 2 * y * d2y_dx2

# Вычисляем первую производную функции потерь по a
# d(loss)/da = 2 * y * dy/da
dy_da = sp.diff(y, a)
loss_da = 2 * y * dy_da

# Вычисляем вторую производную функции потерь по a
# d^2(loss)/da^2 = 2 * (dy/da)^2 + 2 * y * d^2y/da^2
d2y_da2 = sp.diff(dy_da, a)
loss_daa = 2 * (dy_da**2) + 2 * y * d2y_da2

# Подставляем численные значения
subs_dict = {x: 0.5, a: 1.0}

# Вычисляем численные значения
loss_value = loss.evalf(subs=subs_dict)
loss_dx_value = loss_dx.evalf(subs=subs_dict)
loss_dxx_value = loss_dxx.evalf(subs=subs_dict)
loss_da_value = loss_da.evalf(subs=subs_dict)
loss_daa_value = loss_daa.evalf(subs=subs_dict)

# Выводим результаты
print(f"Значение функции потерь: {loss_value}")
print(f"Первая производная потерь по x: {loss_dx_value}")
print(f"Вторая производная потерь по x: {loss_dxx_value}")
print(f"Первая производная потерь по a: {loss_da_value}")
print(f"Вторая производная потерь по a: {loss_daa_value}")
