import numpy as np
from statistics import mean
def bestfit(x, y):
    m = (mean(x*y)-(mean(x)*mean(y)))/(mean(x**2)-(mean(x)**2))
    b = mean(y) - m * mean(x)
    return m, b
c = list()
l = list()

with open("trainingdata.txt") as file:
    for i in file:
        fld = i.split(',')
        if float(fld[0]) <= 4.00:
            c.append(float(fld[0]))
            l.append(float(fld[1][: 4]))
c= np.array(c)
l= np.array(l)
m, b = bestfit(c,l)
n = float(input())
if n > 4.00:
    print(8.00)
else:
    print(m * n + b)