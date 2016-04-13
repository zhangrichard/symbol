#!/usr/bin/env python

from __future__ import print_function # python 2 and 3 compatibility
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Where graphtype is 2 for a 2D plot, or 3 for a 3D plot.
# Example: bivar(3, [4 3], [1 0 ; 0 1], [2 2], [1 0; 0 1])
def bivar(graphtype=2, mubass=[4, 3], sigmabass=[[1, 0.75], [0.75, 1]], musalmon=[2, 2], sigmasalmon=[[1, 0], [0, 1]], interactive=False):

  # Compute probability of (x,y) for a given mu and sigma
  def getprob(x, y, mu, sigma):
    vec = np.matrix([x, y])
    mua = np.matrix(mu)
    E = 2.0 * np.pi * np.sqrt(np.linalg.det(sigma))
    P = (1/E) * np.exp(-1 * ((vec-mua) * np.linalg.inv(sigma) * (vec.T-mua.T) / 2.0))
    return float(P)

  # initialise default parameters
  Nbass = 100
  Nsalmon = 100

  # Generate distribution surface data
  bass = multivariate_normal.rvs(mubass,sigmabass,Nbass);
  salmon = multivariate_normal.rvs(musalmon,sigmasalmon,Nsalmon);
  x = np.arange(0, 6.1, 0.1)
  y = np.arange(0, 6.1, 0.1)

  Pbass = np.zeros((len(y), len(x)))
  Psalmon = np.zeros((len(y), len(x)))
  for i in range(len(x)):
    for j in range(len(y)):
      Pbass[j,i] = getprob(x[i], y[j], mubass, sigmabass)
      Psalmon[j,i] = getprob(x[i], y[j], musalmon, sigmasalmon)

  Pb = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigmabass  )))) * np.exp(-1/2.0)
  Ps = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigmasalmon)))) * np.exp(-1/2.0)

  LR = np.divide(Pbass, Psalmon)

  # Do the plotting
  if interactive:
    plt.ion()
  if graphtype == 2:
    # 2D scatter plots and standard deviation contours
    fig2D = plt.figure()
    ax = fig2D.add_subplot( 111 )
    ax.grid(True)
    ax.scatter(bass[:,0], bass[:,1], color='k',)
    ax.scatter(salmon[:,0], salmon[:,1], color='r')
    ax.contour(x, y, Pbass, [Pb], colors='k')
    ax.contour(x, y, Psalmon, [Ps], colors='r')
    ax.contour(x, y, LR, [1], colors='g')
  elif graphtype == 32:
    # 2D projection of 3D surfaces + standard deviation contours
    fig = plt.figure()
    ax = fig.add_subplot(111)

    vmin=np.min(np.nanmin(Pbass), np.nanmin(Psalmon))
    vmax=np.max(np.nanmax(Pbass), np.nanmax(Psalmon))

    Pbs = np.maximum(Pbass, Psalmon)
    ax.contourf(x, y, Pbs, vmin=vmin, vmax=vmax)
    # ax.contourf(x, y, Psalmon, vmin=vmin, vmax=vmax, alpha=.6)
    # ax.contourf(x, y, Pbass, vmin=vmin, vmax=vmax,  alpha=.4)

    ax.contour(x, y, Pbass, [Pb], colors='k', linewidths=3)
    ax.contour(x, y, Psalmon, [Ps], colors='r', linewidths=3)
    ax.contour(x, y, LR, [1], colors='g', linewidths=3)
  elif graphtype == 3:
    # 3D surfaces and standard deviation contours
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    Pbs = np.maximum(Pbass, Psalmon)
    vmin=np.nanmin(Pbs)
    vmax=np.nanmax(Pbs)

    xl, yl= np.meshgrid(x, y)
    ax.plot_surface(xl, yl, Pbs,\
      vmin=vmin, vmax=vmax,\
      rstride=1, cstride=1, cmap=plt.cm.jet, alpha=0.3, linewidth=0, antialiased=False)

    ax.contour(xl, yl, Pbass, [Pb], colors='k', linewidths=5)
    ax.contour(xl, yl, Psalmon, [Ps], colors='r', linewidths=5)
    ax.contour(xl, yl, LR, [1], offset=0, colors='g', linewidths=5)

    ax.set_zlim(0, 1.5*vmax)
    ax.view_init(elev=90, azim=-90)
  else:
    print("Unknown plotting option: %d" % graphtype)
  if interactive:
    plt.draw()
  else:
    plt.show()

if __name__ == "__main__":
  while True:
    try:
      choice = int(input("Do you want to plot this in: \n(2) 2D,\n(32) 2D projection of 3D plot, or\n(3) 3D?\n> "))
    except ValueError:
      print("Please input a number: 2 or 3")
      continue
    if choice == 2 or choice == 3 or choice == 32:
      break
    else:
      print("That is neither 2, 32, nor 3! Try again:")
  bivar(choice)