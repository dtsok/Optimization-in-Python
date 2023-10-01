import sys
import numpy as np

""" Forecasters:
    1) Simple moving average - SAS
    2) Linear exponential smoothing - LES
    3) Simple exponential smoothing - SES

    Below functions:
    Given period (day) t returns the prediction for the next period (t+1)
"""


def SAS(data: np.array, xi: int, t: int) -> float:
    if t > 1 and t <= len(data):
        if t > xi:
            return np.sum(data[t - xi : t]) / xi
        else:
            return np.sum(data[0 : t - 1]) / (t - 1)
    else:
        return data[0]


def SES(data: np.array, a: float, t: int) -> float:
    prediction = data[0]
    for i in range(1, t):
        prediction = a * data[i] + (1 - a) * prediction
    return prediction


def LES(data: np.array, a: float, b: float, t: int) -> float:
    level = data[0]
    trend = data[1] - data[0]
    for i in range(1, t):
        prev_lvl = level
        level = a * data[i] + (1 - a) * (level + trend)
        trend = b * (level - prev_lvl) + (1 - b) * trend
    prediction = level + trend
    return prediction


""" New prediction function combining forecasters using weights
    y_new_t = Sum_k(w_k*y_forecaster_k(t))
    where:
    y_forecaster_k(t) is the prediction given by forecaster k for period t
    w_k the weight for forecaster k

    Objective (I): Find optimal weight vector w based on the last m values/observations
    Objective function: mean square error
    Objective (II): Minimize MSE w.r.t. w

    E(w) = (1/m)*Sum_{N-m+1}^{N}[y_new_t-y_real]^2
"""


# Objective function: f
def f(w: np.array, data: np.array, forecasters_pred: np.array, m: int, N: int) -> float:
    y = 0
    t = 0
    for l in range(N - m, N):
        prod = 0
        for k in range(len(w)):
            prod = prod + w[k] * forecasters_pred[k][t]
        t += 1
        y = y + (prod - data[l]) ** 2
    return y / m


# partial derivative of f
def df_wi(
    w: np.array, data: np.array, forecasters_pred: np.array, m: int, N: int, i: int
) -> float:
    y = 0
    t = 0
    for l in range(N - m, N):
        prod = 0
        for k in range(len(w)):
            prod = prod + w[k] * forecasters_pred[k][t]
        y = y + (prod - data[l]) * forecasters_pred[i][t]
        t += 1
    return (2 / m) * y


# gradient of f
def gradf(
    w: np.array, data: np.array, forecasters_pred: np.array, m: int, N: int
) -> np.array:
    return np.array([df_wi(w, data, forecasters_pred, m, N, i) for i in range(len(w))])


# mix partial derivative of f
def df_wij(forecasters_pred: np.array, m: int, N: int, i: int, j: int) -> float:
    y = 0
    t = 0
    for l in range(N - m, N):
        y = y + forecasters_pred[i][t] * forecasters_pred[j][t]
        t += 1
    return 2 * y / m


# hessian matrix of f
def hessianf(w: np.array, forecasters_pred: np.array, m: int, N: int) -> np.array:
    hes = np.array(
        [
            np.array([df_wij(forecasters_pred, m, N, i, j) for j in range(len(w))])
            for i in range(len(w))
        ]
    )
    return hes


def readData() -> np.array:
    try:
        return np.loadtxt("EURUSD.dat", dtype=np.float64)
    except IOError as err:
        print("Error @ loading data")
        exit()


# initialize/fill array with the predictions from the forecasters
def initializePredictions(data: np.array, f_pred: np.array, k: int, m: int):
    N = len(data)
    # SAS: xi = 1
    f1 = lambda t: SAS(data, 1, t)
    # SAS: xi = 5
    f2 = lambda t: SAS(data, 5, t)
    # SES: a = 0.2
    f3 = lambda t: SES(data, 0.2, t)
    # SES: a = 0.8
    f4 = lambda t: SES(data, 0.8, t)
    # LES: a = 0.3, b = 0.1
    f5 = lambda t: LES(data, 0.3, 0.1, t)
    # LES: a = 0.3, b = 0.5
    f6 = lambda t: LES(data, 0.3, 0.5, t)
    index = 0
    for t in range(N - m, N):
        f_pred[0][index] = f1(t)
        f_pred[1][index] = f2(t)
        f_pred[2][index] = f3(t)
        f_pred[3][index] = f4(t)
        f_pred[4][index] = f5(t)
        f_pred[5][index] = f6(t)
        index += 1

# Line search with Wolfe conditions - search (sub)optimal step size a
def LineSearch(fun, f_prime, w: np.array, p: np.array, aMax: float) -> float:
    # Wolfe condition constants
    c1 = 1e-4
    c2 = 0.9
    a = 0
    a_next = aMax
    i = 1
    while True:
        fa_next = fun(w + a_next * p)
        f_zero = fun(w)
        f_zero_prime = f_prime(w).dot(p)
        if fa_next > f_zero + c1 * a_next * f_zero_prime or (
            fa_next >= fun(w + a * p) and i > 1
        ):
            return zoom(fun, f_prime, w, p, a, a_next, c1, c2)

        fa_next_prime = f_prime(w + a_next * p).dot(p)
        if abs(fa_next_prime) <= -c2 * f_zero_prime:
            return a_next
        elif fa_next_prime >= 0:
            return zoom(fun, f_prime, w, p, a_next, a, c1, c2, p)
        else:
            temp = a_next
            a_next = (a + a_next) / 2.0
            a = temp
            i += 1


# Zoom subroutine - findind ai in a subset (low, high)
def zoom(
    fun,
    f_prime,
    w: np.array,
    p: np.array,
    low: float,
    high: float,
    c1: float,
    c2: float,
) -> float:
    while True:
        a = (low + high) / 2.0  # bisection - interpolation
        fa = fun(w + a * p)
        f_zero_prime = f_prime(w).dot(p)
        if fa > fun(w) + c1 * a * f_zero_prime or fa >= fun(w + low * p):
            high = a
        else:
            fa_prime = f_prime(w + a * p).dot(p)
            if abs(fa_prime) <= -c2 * f_zero_prime:
                return a
            if fa_prime * (high - low) >= 0:
                high = low
            low = a


def main(argv: list):
    data = readData()  # data - 180 days total
    k = 6  # (fixed) number of forecasters
    m = int(argv[1])  # predict best last m days
    N = len(data)
    f_predictions = np.zeros((k, m))  # predictions from forecasters
    initializePredictions(data, f_predictions, k, m)

    # simplify the calls to the objective function
    fun = lambda w: f(w, data, f_predictions, m, N)
    dfun = lambda w: gradf(w, data, f_predictions, m, N)

    # initial values for variables
    alpha = 1
    iterations = 0
    q = 1
    # w = np.random.rand(6)
    w = np.zeros(6)  # initial state
    while q > 1e-4 and iterations < 1e4:
        p = -dfun(w)
        alpha = LineSearch(fun, dfun, w, p, alpha)
        w = w + alpha * p
        q = np.linalg.norm(dfun(w))
        print(q)
        iterations += 1
    print(w)


if __name__ == "__main__":
    main(sys.argv)
