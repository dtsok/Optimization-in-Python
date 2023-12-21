# Python
Contains various projects in Python 

## Project 1:
Optimization of the objective function using Gradient-based methods.
####  Line Search satisfying Wolfe conditions using:
1. Steepest descent
2. Newton
3. BFGS
#### Trust Region using:
1. BFGS


#### Goal:
Predict the last $\mu$ days of the Euros to US Dollars exchange rate combining various well-known time-series forecasters. The forecasters used:
1. $M_{1}$: simple moving average (SAS) with $\xi = 1$
2. $M_{2}$: simple moving average (SAS) with $\xi = 5$
3. $M_{3}$: simple exponential smoothing (SES) with $\alpha = 0.2$
4. $M_{4}$: simple exponential smoothing (SES) with $\alpha = 0.8$
5. $M_{5}$: linear exponential smoothing (LES) with $\alpha = 0.3, \beta = 0.1$
6. $M_{6}$: linear exponential smoothing (LES) with $\alpha = 0.3, \beta = 0.5$

#### Mathematical intepretation:
The file EURUSD.dat contains the Euros to US Dollars exchange rate for a period of 180 days. Using this file we want to predict the last $\mu$ days for different values of $\mu$.

...

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

ex. Predicting the last 15 days using Newton as the descent direction with Wolfe conditions Line search command:

`python3 project_1.py 15 1 LS`


---

## Project 2:
Optimization of the objective function using Derivative-free methods.
