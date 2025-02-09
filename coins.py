#!/usr/bin/env python3

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

##
## Samplers
##

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
    Dk = 1.0
    while (bernoulli.rvs(Dk * x, size=1).sum() == 1):
        k += 1
        Dk = float(k - 1) / float(k)
    return (((k % 4) == 2) or ((k % 4) == 3))



# Return a sample from the random variable (bernoulli e^x)
# Taylor series drawn directly from e^(-x)
# Good for 0 <= x <= 1
def bern_exp_neg_1_direct(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x / float(k), size=1).sum() == 1):
        k += 1
    return ((k % 2) == 1)


# Return a sample from the random variable (bernoulli 1/(1+x))
# Taylor series drawn directly from 1/(1+x)
# Good for 0 <= x <= 1
def bern_inv_one_plus_direct(x):
    if (x > 1): return 0
    k = 1
    while (bernoulli.rvs(x, size=1).sum() == 1):
        k += 1
    return ((k % 2) == 1)


# Return a sample from the random variable (bernoulli 1/(1+x))
# Taylor series drawn directly from 1/(1+x)
# Good for 0 <= x <= 1
def bern_log_one_plus_direct(x):
    if (x > 1): return 0
    k = 1
    Dk = 1.0
    while (bernoulli.rvs(Dk * x, size=1).sum() == 1):
        k += 1
        Dk = float(k - 1) / float(k)
    return ((k % 2) == 0)


##
## Tests and plots
##




def sample_data(f, xs, N):
    values = []
    for x in tqdm(xs):
        values += [np.array([f(x) for _ in range(N)]).mean()]
    return np.array(values)

def sample_value_single(f, x, N):
    success = 0
    for _ in tqdm(range(N)):
        success += f(x)
    return (float(success) / float(N))


def compare_cos(xs, N):
    print("Sampling: cos(x)")
    ideal = np.cos(xs)
    values_cos_1 = sample_data(bern_cos_1, xs, N)
    values_cos_sq2 = sample_data(bern_cos_sq2, xs, N)
    values_cos_1_approx = sample_data(bern_cos_1_approx, xs, N)
    values_cos_1_direct = sample_data(bern_cos_1_direct, xs, N)
    values_cos_indirect = sample_data(bern_cos_indirect, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 5))

    plt.figure(figsize=(12,8))
    plt.title("cos samplers, N={}".format(N))
    plt.plot(xs, values_cos_1, color=colours[0], label="E[bern_cos_1(x)]")
    plt.plot(xs, values_cos_1_approx, color=colours[1], label="E[bern_cos_approx(x)]")
    plt.plot(xs, values_cos_sq2, color=colours[2], label="E[bern_cos_sq2(x)]")
    plt.plot(xs, values_cos_1_direct, color=colours[3], label="E[bern_cos_1_direct(x)]")
    plt.plot(xs, values_cos_indirect, color=colours[4], label="E[bern_cos_indirect(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="cos(x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/cos.pdf")


def compare_sin(xs, N):
    print("Sampling: sin(x)")
    ideal = np.sin(xs)
    values_sin_1_direct = sample_data(bern_sin_1_direct, xs, N)
    values_sin_indirect = sample_data(bern_sin_indirect, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 2))

    plt.figure(figsize=(12,8))
    plt.title("sin samplers, N={}".format(N))
    plt.plot(xs, values_sin_1_direct, color=colours[0], label="E[bern_sin_1_direct(x)]")
    plt.plot(xs, values_sin_indirect, color=colours[1], label="E[bern_sin_indirect(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="sin(x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/sin.pdf")


def compare_arctan(xs, N):
    print("Sampling: arctan(x)")
    ideal = np.arctan(xs)
    values_arctan_1_direct = sample_data(bern_arctan_1_direct, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 1))

    plt.figure(figsize=(12,8))
    plt.title("arctan samplers, N={}".format(N))
    plt.plot(xs, values_arctan_1_direct, color=colours[0], label="E[bern_arctan_direct(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="arctan(x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/arctan.pdf")


def compare_exp_neg(xs, N):
    print("Sampling: e^(-x)")
    ideal = np.exp(-xs)
    values_exp_1_direct = sample_data(bern_exp_neg_1_direct, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 1))

    plt.figure(figsize=(12,8))
    plt.title("exp(-x) samplers, N={}".format(N))
    plt.plot(xs, values_exp_1_direct, color=colours[0], label="E[bern_exp_neg_direct(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="exp(-x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/exp.pdf")


def compare_inv_one_plus(xs, N):
    print("Sampling: 1/(1+x)")

    ideal = 1.0 / (1.0 + xs)
    values_inv_one_plus_direct = sample_data(bern_inv_one_plus_direct, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 1))

    plt.figure(figsize=(12,8))
    plt.title("1/(1+x) samplers, N={}".format(N))
    plt.plot(xs, values_inv_one_plus_direct, color=colours[0], label="E[bern_inv_one_plus_direct(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="exp(-x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/inv_one_plus.pdf")


def compare_log_one_plus(xs, N):
    print("Sampling: log(1+x)")

    ideal = np.log(1.0 + xs)
    values_log_one_plus_direct = sample_data(bern_log_one_plus_direct, xs, N)

    colours = plt.cm.rainbow(np.linspace(0, 1, 1))

    plt.figure(figsize=(12,8))
    plt.title("log(1+x) samplers, N={}".format(N))
    plt.plot(xs, values_log_one_plus_direct, color=colours[0], label="E[bern_log_one_plus_direct(x)]")
    plt.plot(xs, ideal, color="black", linestyle="--", label="log(1+x)")
    plt.xlabel("x")
    plt.legend(loc = "best")
    plt.savefig("figures/log_one_plus.pdf")


def pi_estimator(N):
    print("Estimating pi:")
    pi_div_four = sample_value_single(bern_arctan_1_direct, 1.0, N)
    print("Estimate: {}".format(4*pi_div_four))


if __name__ == "__main__":

    if not os.path.exists("./figures"):
        os.makedirs("./figures")

    # Plot grid
    n = 100
    delta =  np.pi / float (2 * n)

    # Range from [0, 1)
    xs_one = np.array([float(i) / float(n) for i in range (n)])
    xs_pi_div_four = np.array([i * delta for i in range (n + 1)])

    # compare_cos(xs_pi_div_four, 5000)
    # compare_sin(xs_pi_div_four, 5000)
    # compare_arctan(xs_one, 5000)
    # compare_exp_neg(xs_one, 5000)
    # compare_inv_one_plus(xs_one, 5000)
    # compare_log_one_plus(xs_one, 5000)

    # pi_estimator(1000000) # Estimate: 3.141524
