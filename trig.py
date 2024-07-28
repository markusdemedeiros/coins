#!/usr/bin/env python3

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm
import matplotlib.pyplot as plt


# Return a sample from the random variable (bernoulli (cos(x)))
# Taylor series drawn from cos(sqrt(x^2))
# Good for 0 <= x <= 1
def bern_cos_1(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x / (float(2 * k)), size=1).sum() == 1 and
           bernoulli.rvs(x / (float(2 * k - 1)), size=1).sum() == 1):
        k += 1
    return (k % 2 == 1)

# Return a sample from the random variable (bernoulli (cos(x)))
# Taylor series drawn from cos(sqrt(x^2))
# Good for 0 <= x <= sqrt(2)
def bern_cos_sq2(x):
    if (x > np.sqrt(2)): return 0
    k = 1
    while (bernoulli.rvs(x * x / (float(2 * k * (2 * k - 1))), size=1).sum() == 1):
        k += 1
    return (k % 2 == 1)


# Return a sample from the random variable (bernoulli (cos(x)))
# Drawn from the first 3 terms of the Taylor Series for cos (good up to O(x^11))
# Good for 0 <= x <= 1
def bern_cos_1_approx(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x / float(k), size=1).sum() == 1):
        k += 1
    return ((k == 1) or (k == 2) or (k == 5) or (k == 6) or (k == 9) or (k == 10))


# Return a sample from the random variable (bernoulli (cos(x)))
# Taylor series drawn directly from cos(x)
# Good for 0 <= x <= 1
def bern_cos_1_direct(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x / float(k), size=1).sum() == 1):
        k += 1
    return (((k % 4) == 1) or ((k % 4) == 2))


# Return a sample from the random variable (bernoulli (sin(x)))
# Taylor series drawn directly from sin (x)
# Good for 0 <= x <= 1
def bern_sin_1_direct(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x / float(k), size=1).sum() == 1):
        k += 1
    return (((k % 4) == 2) or ((k % 4) == 3))

def negate_bern(x):
    return 1 - x


# Return a sample from the random variable (bernoulli (cos(x)))
# Taylor series drawn indirectly from sin and cos samplers
# Requires evaluating (pi/2 - x), can only be computed lazily
# Good for 0 <= x < pi/2
def bern_cos_indirect(x):
    if (x < 1):
        return bern_cos_1_direct(x)
    else:
        if (np.pi / 2.0 - x) < 0: return 0
        return bern_sin_1_direct(np.pi / 2.0 - x)


# Return a sample from the random variable (bernoulli (sin(x)))
# Taylor series drawn indirectly from sin and cos samplers
# Requires evaluating (pi/2 - x), can only be computed lazily
# Good for 0 <= x < pi/2
def bern_sin_indirect(x):
    if (x < 1):
        return bern_sin_1_direct(x)
    else:
        if (np.pi / 2.0 - x) < 0: return 1
        return bern_cos_1_direct(np.pi / 2.0 - x)


# Return a sample from the random variable (bernoulli (arctan(x)))
# Taylor series drawn directly from arctan(x)
# Good for 0 <= x <= 1
def bern_arctan_1_direct(x):
    if (x > 1): return 0
    k = 1
    Dk = 1
    while (bernoulli.rvs(Dk, size=1).sum() == 1):
        k += 1
        Dk = float(k * x) / float(k - 1)
    return (((k % 4) == 1) or ((k % 4) == 2))





# Number of samples
N = 1000

# Plot grid
n = 100
M = np.pi / 2.0
delta = M / float (n)

def sample_data(f, xs):
    values = []
    for x in tqdm(xs):
        values += [np.array([f(x) for _ in range(N)]).mean()]
    return np.array(values)


def compare_cos(xs):
    ideal_cos = np.cos(xs)
    values_cos_1 = sample_data(bern_cos_1, xs)
    values_cos_sq2 = sample_data(bern_cos_sq2, xs)
    values_cos_1_approx = sample_data(bern_cos_1_approx, xs)
    values_cos_1_direct = sample_data(bern_cos_1_direct, xs)
    values_cos_indirect = sample_data(bern_cos_indirect, xs)

    colours = plt.cm.rainbow(np.linspace(0, 1, 5))

    plt.figure(figsize=(12,8))
    plt.title("cos samplers, N={}".format(N))
    plt.plot(xs, values_cos_1, color=colours[0], label="E[bern_cos_1(x)]")
    plt.plot(xs, values_cos_1_approx, color=colours[1], label="E[bern_cos_approx(x)]")
    plt.plot(xs, values_cos_sq2, color=colours[2], label="E[bern_cos_sq2(x)]")
    plt.plot(xs, values_cos_1_direct, color=colours[3], label="E[bern_cos_1_direct(x)]")
    plt.plot(xs, values_cos_indirect, color=colours[4], label="E[bern_cos_indirect(x)]")
    plt.plot(xs, ideal_cos, color="black", linestyle="--", label="cos(x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("cos.pdf")


def compare_sin(xs):
    ideal_sin = np.sin(xs)
    values_sin_1_direct = sample_data(bern_sin_1_direct, xs)
    values_sin_indirect = sample_data(bern_sin_indirect, xs)

    colours = plt.cm.rainbow(np.linspace(0, 1, 2))

    plt.figure(figsize=(12,8))
    plt.title("sin samplers, N={}".format(N))
    plt.plot(xs, values_sin_1_direct, color=colours[0], label="E[bern_sin_1_direct(x)]")
    plt.plot(xs, values_sin_indirect, color=colours[1], label="E[bern_sin_indirect(x)]")
    plt.plot(xs, ideal_sin, color="black", linestyle="--", label="sin(x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("sin.pdf")


if __name__ == "__main__":
    xs = np.array([i * delta for i in range (n + 1)])
    compare_cos(xs)
    compare_sin(xs)
