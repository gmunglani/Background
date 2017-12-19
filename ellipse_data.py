import numpy as np
from numpy.linalg import eig, inv
import csv
from pylab import *
from scipy.spatial import distance

def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    print(np.dot(inv(S), C))
    print(E.shape,V.shape)
    n = np.argmax(np.abs(E))
    a = V[:, n]
    print(a)
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    print(np.array([x0, y0]))
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    print(np.array([res1, res2]))
    return np.array([res1, res2])

def ellipse_angle_of_rotation2(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

def fitness(coord):
    coord = np.genfromtxt('test_output2.csv', delimiter=",")

    x = coord[:,0]
    y = coord[:,1]

    a = fitEllipse(x,y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)
    print(phi)
    
    #a, b = axes
    #R = np.arange(0,np.pi, 0.01)
    #xx = center[0] + b*np.cos(R)*np.cos(phi) - a*np.sin(R)*np.sin(phi)
    #yy = center[1] + b*np.cos(R)*np.sin(phi) + a*np.sin(R)*np.cos(phi)

    xe1 = center[0] + axes[0]*np.sin(np.pi*phi)
    ye1 = center[1] - axes[0]*np.cos(np.pi*phi)

    xe2 = center[0] - axes[0]*np.sin(np.pi*phi)
    ye2 = center[1] + axes[0]*np.cos(np.pi*phi)

    diste1 = sum(distance.cdist(np.vstack((xe1,ye1)).T,coord,'euclidean'))
    diste2 = sum(distance.cdist(np.vstack((xe2,ye2)).T,coord,'euclidean'))

    majore = np.empty([2])

    if (diste1 < diste2):
        majore = np.vstack((xe1,ye1))
    else:
        majore = np.vstack((xe2,ye2))

    #print(majore)

    #plot(x,y)
    #plot(xx,yy, color = 'red')
    #plot(center[0],center[1], 'bo')
    #plot(xe1,ye1, 'bo')
    #plot(xe2,ye2, 'bo')

    #plt.show()
    return majore