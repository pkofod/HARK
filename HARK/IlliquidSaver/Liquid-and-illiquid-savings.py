# ---
# jupyter:
#   '@webio':
#     lastCommId: 493edb500e834cfab9155cc952e200e4
#     lastKernelId: 24e0f69d-c4cd-48b7-811b-0d0807abe02a
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
#     version: 3.6.4
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


