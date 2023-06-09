#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%%

n = 100
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x)*x

G = np.ones((n, 2))
G[:, 0] = x

std = np.hstack((np.ones(int(n/2))*2, np.ones(int(n/2))*5))
Cd = np.diag(std**2)
Cdinv = scipy.linalg.lstsq(Cd, np.eye(n))[0]

Cpm = scipy.linalg.lstsq(G.T.dot(Cdinv).dot(G), np.eye(2))[0]
Cpd = G.dot(Cpm).dot(G.T)

m = Cpm.dot(G.T.dot(Cdinv).dot(y))

yy = G.dot(m)

plt.figure(figsize=(10, 10))
plt.plot(x, y, label='data')
plt.fill_between(x, y-std, y+std, alpha=0.3, color='tab:blue', label='data variance')
plt.plot(x, yy, label='model predictions')
plt.fill_between(x, yy-np.sqrt(np.diag(Cpd)), yy+np.sqrt(np.diag(Cpd)), alpha=0.3, color='tab:orange', label='projected variance')

#%% Lompe
s = 100

Cd = np.diag((std+s)**2)
Cdinv = scipy.linalg.lstsq(Cd, np.eye(n))[0]

Cpm = scipy.linalg.lstsq(G.T.dot(Cdinv).dot(G), np.eye(2))[0]
Cpd = G.dot(Cpm).dot(G.T)

m = Cpm.dot(G.T.dot(Cdinv).dot(y))

yy = G.dot(m)

plt.plot(x, yy, label='with Lompe scaling')
plt.fill_between(x, yy-np.sqrt(np.diag(Cpd)), yy+np.sqrt(np.diag(Cpd)), alpha=0.3, color='tab:green', label='Projected variance')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')


#%% Attempt to find correct scaling method

s = 100

Cd = np.diag((std/s)**2)
Cdinv = scipy.linalg.lstsq(Cd, np.eye(n))[0]

Cpm = scipy.linalg.lstsq(G.T.dot(Cdinv).dot(G), np.eye(2))[0]
Cpd = G.dot(Cpm).dot(G.T)

m = Cpm.dot(G.T.dot(Cdinv).dot(y/s))

yy = s * (G.dot(m))

plt.plot(x, yy, label='with scaling')
plt.fill_between(x, yy-s*np.sqrt(np.diag(Cpd)), yy+s*np.sqrt(np.diag(Cpd)), alpha=0.3, color='tab:green', label='Projected variance')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')



































#%%


Cdsinv = scipy.linalg.lstsq(Cds, np.eye(n))[0]
ms = scipy.linalg.lstsq(G.T.dot(Cdsinv).dot(G), G.T.dot(Cdsinv).dot(y))[0]
yys = G.dot(ms)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.plot(x, yys)
plt.plot(x, yys*s)



#%%
s = 100
ys = y/s
Gs = G/s
Cds = np.eye(n) * (5/s)**2
Cdsinv = scipy.linalg.lstsq(Cds, np.eye(n))[0]
ms = scipy.linalg.lstsq(Gs.T.dot(Cdsinv).dot(Gs), Gs.T.dot(Cdsinv).dot(ys))[0]
yys = Gs.dot(ms)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.plot(x, yys)
plt.plot(x, yys*s)

#%%
s = 100
Gs = G/s
ms = scipy.linalg.lstsq(Gs.T.dot(Cdinv).dot(Gs), Gs.T.dot(Cdinv).dot(y))[0]
yys = Gs.dot(ms)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.plot(x, yys)
plt.plot(x, yys*s)


#%%
s = 100
Cds = np.eye(n) * (5 + s)**2
Cdsinv = scipy.linalg.lstsq(Cds, np.eye(n))[0]
ms = scipy.linalg.lstsq(G.T.dot(Cdsinv).dot(G), G.T.dot(Cdsinv).dot(y))[0]
yys = G.dot(ms)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.plot(x, yys)
plt.plot(x, yys*s)