# ---
# jupyter:
#   '@webio':
#     lastCommId: c429af11def541468ab1929d72e8e9d6
#     lastKernelId: 79706546-268b-4aa7-a766-c4a53988de82
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# # Liquid and illiquid savings

# +
import sys, os

sys.path.insert(0, "/home/pkofod/HARK/")
# -

from illiquid import IlliquidSaver

illsaver = IlliquidSaver()

illsaver.updateLast()

illsaver.solution_terminal.V_T

import HARK.interpolation as itp

help(itp.BilinearInterp)

illsaver.solve()

import numpy
A=numpy.zeros((3,6))+1
B=numpy.zeros((3,6))+2
from HARK.interpolation import calcLogSumChoiceProbs

calcLogSumChoiceProbs(numpy.stack((A, B)),0.0)

# +
import numpy
import matplotlib.pyplot as plt


from mpl_toolkits import mplot3d


# -

illsaver.solution[0].C.shape

plt.plot(illsaver.grids.m,illsaver.solution[0].BFunc(illsaver.grids.m, 1))

illsaver.grids.m+0.1-illsaver.utility.adjcost

plt.plot(illsaver.grids.m, illsaver.solution[0].BFunc(illsaver.grids.m, 3.0)-3.0)

plt.plot(illsaver.solution[0][0].ravel(), illsaver.solution[0][1].ravel())

illsaver.solution[0].P[0]

# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[0:,0:]
Y = illsaver.grids.N[0:,0:]
Z = Y-illsaver.solution[0].BFunc(X,Y)
Z1 = Z.copy()
Z2 = Z.copy()
Z1[Z<0.0] = numpy.nan
Z2[Z>=0.0] = numpy.nan
ax.plot_surface(X, Y, Z1)
ax.plot_surface(X, Y, Z2)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(45, -175)
# -

plt.plot(illsaver.solution[1].V_T[0])

# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[450:1700,450:1700]
Y = illsaver.grids.N[450:1700,450:1700]
Z = illsaver.solution[2].V_TFunc(X,Y)
Z1 = numpy.divide(-1.0, Z.copy())
ax.plot_surface(X, Y, Z)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(20, 175)


# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[0:,0:]
Y = illsaver.grids.N[0:,0:]
Z = illsaver.solution[-1].CFunc(X,Y)
Z1 = Z.copy()
ax.plot_surface(X, Y, Z1)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(45, -135)

# -

illsaver.solution[0].P

# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[0:,100:]
Y = illsaver.grids.N[0:,100:]
Z = illsaver.solution[2].V_TFunc(X,Y)
Z1 = Z.copy()
ax.plot_surface(X, Y, Z1)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(45, -40)

# -

numpy.isnan(Z3).any()

Z

# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[100:,100:]
Y = illsaver.grids.N[100:,100:]
Z = illsaver.solution[1].BFunc(X,Y)-Y
Z1 = Z.copy()
Z2 = Z.copy()
Z1[Z<-0.04] = numpy.nan
Z2[Z>0.04] = numpy.nan

ax.plot_surface(X, Y, Z2)
ax.plot_surface(X, Y, Z1)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(90, 95)


# +
fig = plt.figure()
ax = plt.axes(projection='3d')


ax.plot_surface(illsaver.solution[0][0][200:,200:], illsaver.grids.N[200:,200:], numpy.divide(-1.0, illsaver.solution[0][3][200:,200:]))

# rotate the axes and update
ax.view_init(30, -90)


# +
fig = plt.figure()
ax = plt.axes(projection='3d')


X, Y = numpy.meshgrid(illsaver.grids.m[40:], illsaver.grids.n[40:])
f = illsaver.solution[0][5].V_TFunc
Z = numpy.divide(-1,f(X, Y))

ax.plot_surface(X, Y, Z)



# +
fig = plt.figure()
ax = plt.axes(projection='3d')


X, Y = numpy.meshgrid(illsaver.grids.m[40:], illsaver.grids.n[40:])
f = illsaver.solution_terminal.V_TFunc
Z = numpy.divide(-1,f(X, Y))

ax.plot_surface(X, Y, Z)



# +
fig = plt.figure()
ax = plt.axes(projection='3d')


X, Y = numpy.meshgrid(illsaver.grids.m[200:], illsaver.grids.n[100:])
f = illsaver.solution[0]
Z = f(X, Y)

ax.plot_surface(X, Y, Z)



# -


