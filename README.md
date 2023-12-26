# Optimization-in-Python
Contains two mathematical optimization projects implemented in Python 

## Project 1:
### Optimization of the objective function using Gradient-based methods
In this project we are given a dataset that contains the exchange rate of the Euro against the US Dollar for a period of 180 days and our task is to predict as accurately as possible the exchange rate for the last $\mu$ days. One way to accomplish this is by using methods based on *moving averages forecasting* such as simple moving average (SAS), simple exponential smoothing (SES) and linear exponential smoothing (LES). Since every forecaster has its own pros and cons we will combine them using weights and the final prediction will be a linear combination of those weights with the corresponding prediction from every forecaster. Hence, our goal is to find the optimal weight vector. In order to find this vector we will minimize a specific function, the *objective function*, using gradient based optimization methods.

#### Mathematical interpretation:
Let $Y$ be a set of $N$ observations of a time series:
$$Y = \lbrace y_{1}, y_{2}, \dots, y_{N}\rbrace,$$
and $\mathcal{M}$ a set of $K$ forecasters:
$$\mathcal{M} = \lbrace M_{1}, M_{2}, \dots,M_{k} \rbrace.$$
For every observation $y_{t},\ t\in T = \lbrace 1,2,\dots, N\rbrace$, every forecaster has a prediction:
$$\hat{y_{t}}^{(k)},\ k = 1,2,\dots, K$$  
A model that combines all $K$ forecasters with weights, will yield a prediction:
$$\hat{y_{t}} = \sum_{k=1}^{K}w_{k}\hat{y}_{t}^{(k)}, t\in T.$$

**Finding the weights:**  
As optimality criterion we will use the minimization of the mean square error of the predicted values for the last $\mu$ days. *(The days that we want to predict as accurately as possible)*.  
Let $`w`$ be the weight vector:  
```math
\begin{equation}
w = [w_{1}, w_{2}, \dots, w]^{K},\ \in [-1,1]^{K}
\end{equation}
```
then the objective function is the following:  
```math
$$ E(w) = \frac{1}{\mu}\sum_{\lambda = N-\mu+1}^{N}(\hat{y}_{\lambda}(w)-y_{\lambda})^2 $$
```
with the minimizers of the above function being the (sub) optimal weights used in forecasting the desired times.


#### Gradient based methods used:
####  Line Search satisfying Wolfe conditions using:
1. Steepest descent
2. Newton
3. BFGS
#### Trust Region using:
1. BFGS


#### Forecasters used:
Predict the last $\mu$ days of the Euros to US Dollars exchange rate combining various well-known time-series forecasters. The forecasters used:
1. $M_{1}$: simple moving average (SAS) with $\xi = 1$
2. $M_{2}$: simple moving average (SAS) with $\xi = 5$
3. $M_{3}$: simple exponential smoothing (SES) with $\alpha = 0.2$
4. $M_{4}$: simple exponential smoothing (SES) with $\alpha = 0.8$
5. $M_{5}$: linear exponential smoothing (LES) with $\alpha = 0.3, \beta = 0.1$
6. $M_{6}$: linear exponential smoothing (LES) with $\alpha = 0.3, \beta = 0.5$


#### Prerequisites:
Python v. $\geq 3.9.9$  
Numpy v. $\geq 1.21.3$


#### Run:
`python3 project_1.py <int> <int> <string>`  
First `<int>`: number of days to predict $\mu=[1,179]$  
Second `<int>`: descent direction, integer in range $[0,2]$

0. Steepest descent  
1. Newton
2. BFGS

Third `<string>`: method - `LS` for Line Search / `TR` for Trust Region

ex. Predicting the last 15 days using Newton descent direction with Wolfe conditions Line search command:

`python3 project_1.py 15 1 LS`


---

## Project 2:
Optimization of the objective function using Derivative-free methods.
