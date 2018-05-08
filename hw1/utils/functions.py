import numpy as np
from scipy import signal
x_range = (0, 1)

def sine_wave(length, heigth, x):
    x = x * np.pi * 2
    y = np.sin(x / length) * heigth
    return y

def square_wave(length, heigth, x):
    x = x * np.pi * 2
    y = signal.square(x / length) * heigth
    return y

def saw_wave(length, heigth, x):
    x = x * np.pi * 2
    y = signal.sawtooth(x / length) * heigth
    return y

def polynomial(coef, x):
    power = len(coef)
    y = 0
    for i in range(0, power):
        y = y + np.power(x, i) * coef[i]
    return y