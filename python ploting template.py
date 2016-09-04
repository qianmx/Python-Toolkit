from random import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from numpy import *
import math

#12
import sys
fancy = False
trials = 40

if len(sys.argv) > 1:
    trials = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == '-fancy':
        fancy = True

# 1
prices = []
f = open("prices.txt")
for line in f:
    v = float(line.strip())
    prices.append(v)


# 2
def sample(data):
    lst = []
    for num in range(len(data)):
        lst.append(data[randrange(0, len(data))])
    return lst

# 3
X_ = []
for i in range(trials):
    samplelst = sample(prices)
    X_.append(np.mean(samplelst))

# 4
boundary_left = int(trials * 0.025)
boundary_right = int(trials * 0.975)
inside = sorted(X_)[boundary_left:boundary_right]

# 5
print inside[0], inside[-1]

# 6,7,8
mean = np.mean(prices)   # what is the mean and stddev here?
stddev = np.std(prices)
std = stddev/sqrt(len(prices))

x = np.arange(1.05, 1.25, 0.001)
y = scipy.stats.norm.pdf(x, mean, std)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis([1.10, 1.201, 0, 30])
plt.plot(x, y, color='red')
plt.plot(inside[0], 0, marker='D', color='b')
plt.plot(inside[-1], 0, marker='D', color='b')

# 7 --extra
def normpdf(x, mu, sigma):
    y = []
    for i in x:
        pi = 3.1415926
        var = float(sigma)**2
        part1 = 1/((2*pi*var)**0.5)
        part2 = math.exp(-(float(i)-float(mu))**2/(2*var))
        y.append(part1*part2)
    return y

# 8, 9
left = mean - (1.96 * std)  # what is the mean and std here?
right = mean + (1.96 * std)

print left, right

# Redraw normal but only shade in 95% CI
ci_x = np.arange(left, right, 0.001)
ci_y = normpdf(ci_x, mean, std)

plt.fill_between(ci_x, ci_y, color='#F8ECE0')

#10,11
plt.text(.02, .95, '$TRIALS = %d$' % trials, transform=ax.transAxes)
plt.text(.02, .9, '$mean(prices)$ = %f' % np.mean(prices), transform = ax.transAxes)
plt.text(.02, .85, '$mean(\\overline{X})$ = %f' % np.mean(X_), transform = ax.transAxes)
plt.text(.02, .80, '$stddev(\\overline{X})$ = %f' % np.std(X_,ddof=1), transform = ax.transAxes)
plt.text(.02, .75, '95%% CI = $%1.2f \\pm 1.96*%1.3f$' % (mean, std), transform = ax.transAxes)
plt.text(.02, .70, '95%% CI = ($%1.2f,\\ %1.2f$)' % (mean-1.96*std, mean+1.96*mean), transform=ax.transAxes)

plt.text(1.135, 11.5, "Expected", fontsize=16)
plt.text(1.135, 10, "95% CI $\\mu \\pm 1.96\\sigma$", fontsize=16)
plt.title("95% Confidence Intervals: $\\mu \\pm 1.96\\sigma$", fontsize=16)

ax.annotate('Empirical 95% CI', xy=(inside[0], .3), xycoords="data", xytext=(1.13, 4),
            textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontsize=16)

# 12, 13
plt.savefig('bootstrap-'+str(trials)+('-basic' if not fancy else '')+'.pdf',format="pdf")   # should be bootstrap-500.pdf or conf-500.pdf
plt.show()
