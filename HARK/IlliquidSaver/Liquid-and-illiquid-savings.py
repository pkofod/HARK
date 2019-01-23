# ---
# jupyter:
#   '@webio':
#     lastCommId: 31eca40f82114de68bfb3c6d0f542f77
#     lastKernelId: e127a39a-6a50-4a9f-abbc-88e25c5de934
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

from illiquid import IlliquidSaver

illsaver = IlliquidSaver()

illsaver.updateLast()

illsaver.solution_terminal.V_T

illsaver.solve()

# +
import numpy
import matplotlib.pyplot as plt


from mpl_toolkits import mplot3d


# -

illsaver.solution[0][1].shape

illsaver.solution[0][5](illsaver.grids.m)

illsaver.grids.m+0.1-illsaver.utility.adjcost

plt.plot(illsaver.grids.m, illsaver.solution[0][5](illsaver.grids.m+5.01-illsaver.utility.adjcost)-5.01)

plt.plot(illsaver.solution[0][0].ravel(), illsaver.solution[0][1].ravel())

illsaver.solution[0]

# +
fig = plt.figure()
ax = plt.axes(projection='3d')

X = illsaver.grids.M[200:500,200:500]
Y = illsaver.grids.N[200:500,200:500]
Z = illsaver.solution[0][5](X+Y-illsaver.utility.adjcost)-Y
ax.plot_surface(X, Y, Z)
ax.set_xlabel("m")
ax.set_ylabel("n")

# rotate the axes and update
ax.view_init(30, -135)


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


