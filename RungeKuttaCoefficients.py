# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 00:36:28 2017

@author: Alastair Wiseman
"""
import numpy as np
import scipy.optimize as sop

ERK1FEMatrix = np.array([[0.0]])
ERK1FEWeights = np.array([1.0])
ERK1FENodes = np.array([0.0])
ERK1FEOrder = 1

ERK1FE = [ERK1FEMatrix, ERK1FEWeights, ERK1FENodes, ERK1FEOrder]


ERK2MatrixMP = np.array([[0.0, 0.0],
                         [0.5, 0.0]])
ERK2WeightsMP = np.array([0.0, 1.0])
ERK2NodesMP = np.array([0.0, 0.5])
ERK2OrderMP = 2

ERK2MP = [ERK2MatrixMP, ERK2WeightsMP, ERK2NodesMP, ERK2OrderMP]


ERK2MatrixH = np.array([[0.0, 0.0],
                        [1.0, 0.0]])
ERK2WeightsH = np.array([0.5, 0.5])
ERK2NodesH = np.array([0.0, 1.0])
ERK2OrderH = 2

ERK2H = [ERK2MatrixH, ERK2WeightsH, ERK2NodesH, ERK2OrderH]


ERK3KMatrix = np.array([[0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [-1.0, 2.0, 0.0]])
             
ERK3KNodes = np.array([0.0, 0.5, 1.0])
ERK3KWeights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
ERK3Order = 3

ERK3K = [ERK3KMatrix, ERK3KWeights, ERK3KNodes, ERK3Order]


ERK4CMatrix = np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]])              
ERK4CNodes = np.array([0.0, 0.5, 0.5, 1.0])
ERK4CWeights = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
ERK4COrder = 4

ERK4C = [ERK4CMatrix, ERK4CWeights, ERK4CNodes, ERK4COrder]


ERK4TEMatrix = np.array([[0.0, 0.0, 0.0, 0.0],
                         [1.0 / 3.0, 0.0, 0.0, 0.0],
                         [-1.0 / 3.0, 1.0, 0.0, 0.0],
                         [1.0, -1.0, 1.0, 0.0]])
ERK4TEWeights = np.array([1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0,1.0 / 8.0])
ERK4TENodes = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
ERK4TEOrder = 4

ERK4TE = [ERK4TEMatrix, ERK4TEWeights, ERK4TENodes, ERK4TEOrder]

ERK5KNMatrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [4.0 / 25.0, 6.0 / 25.0, 0.0, 0.0, 0.0, 0.0],
                         [1.0 / 4.0, -3.0, 15.0 / 4.0, 0.0, 0.0, 0.0],
                         [2.0 / 27.0, 10.0 / 9.0, -50.0 / 81.0, 8.0 / 81.0,
                          0.0,0.0],
                         [2.0 / 25.0, 12.0 / 25.0, 2.0 / 15.0, 8.0 / 75.0,
                          0.0, 0.0]])
ERK5KNWeights = np.array([23.0 / 192.0, 0.0, 125.0 / 192.0, 0.0, -27.0 / 64.0,
                          125.0 / 192.0])
ERK5KNNodes = np.array([0.0, 1.0 / 3.0, 2.0 / 5.0, 1.0, 2.0 / 3.0, 4.0 / 5.0])
ERK5KNOrder = 5

ERK5KN = [ERK5KNMatrix, ERK5KNWeights, ERK5KNNodes, ERK5KNOrder]

ERK5BMatrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0 / 8.0, 1.0 / 8.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
                        [3.0 / 16.0, -3.0 / 8.0, 3.0 / 8.0, 9.0 / 16.0, 0.0,
                         0.0],
                        [-3.0 / 7.0, 8.0 / 7.0, 6.0 / 7.0, -12.0 / 7.0,
                         8.0 / 7.0, 0.0]])
ERK5BWeights = np.array([7.0 / 90.0, 0.0, 16.0 / 45.0, 2.0 / 15.0, 
                         16.0 / 45.0, 7.0 / 90.0])
ERK5BNodes = np.array([0.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0])
ERK5BOrder = 5

ERK5B = [ERK5BMatrix, ERK5BWeights, ERK5BNodes, ERK5BOrder]

ERK6BMatrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0 / 12.0, 1.0 / 3.0, -1.0 / 12.0, 0.0, 0.0, 0.0, 
                         0.0],
                        [25.0 / 48.0, -55.0 / 24.0, 35.0 / 48.0, 15.0 / 8.0,
                         0.0, 0.0 ,0.0],
                        [3.0 / 20.0, -11.0 / 24.0, -1.0 / 8.0, 1.0 / 2.0,
                         1.0 / 10.0, 0.0, 0.0],
                        [-261.0 / 260.0, 33.0 / 13.0, 43.0 / 156.0,
                         -118.0 / 39.0, 32.0 / 195.0, 80.0 / 39.0, 0.0]])
ERK6BWeights = np.array([13.0 / 200.0, 0.0, 11.0 / 40.0, 11.0 / 40.0,
                         4.0 / 25.0, 4.0 / 25.0, 13.0 / 200.0])
ERK6BNodes = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0,
                       1.0 / 6.0, 1.0])
ERK6BOrder = 6

ERK6B = [ERK6BMatrix, ERK6BWeights, ERK6BNodes, ERK6BOrder]


IRK1BEMatrix = np.array([[1.0]])
IRK1BEWeights = np.array([1.0])
IRK1BENodes = np.array([1.0])
IRK1BEOrder = 1

IRK1BE = [IRK1BEMatrix, IRK1BEWeights, IRK1BENodes, IRK1BEOrder] 


IRK1IMMatrix = np.array([[0.5]])
IRK1IMWeights = np.array([1.0])
IRK1IMNodes = np.array([0.5])
IRK1IMOrder = 2

IRK1IM = [IRK1IMMatrix, IRK1IMWeights, IRK1IMNodes, IRK1IMOrder]


IRK2LAMatrix = np.array([[0.0, 0.0],
                         [0.5, 0.5]])
IRK2LAWeights = np.array([0.5, 0.5])
IRK2LANodes = np.array([0.0, 1.0])
IRK2LAOrder = 2

IRK2LA = [IRK2LAMatrix, IRK2LAWeights, IRK2LANodes, IRK2LAOrder]


IRK4LAMatrix = np.array([[0.0, 0.0, 0.0],
                         [5.0 / 24.0, 1.0 / 3.0, -1.0 / 24.0],
                         [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]])
IRK4LAWeights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
IRK4LANodes = np.array([0.0 , 1.0 / 2.0 , 1.0])
IRK4LAOrder = 4

IRK4LA = [IRK4LAMatrix, IRK4LAWeights, IRK4LANodes, IRK4LAOrder]


IRK2LBMatrix = np.array([[0.5, 0.0],
                         [0.5, 0.0]])
IRK2LBWeights = np.array([0.5, 0.5])
IRK2LBNodes = np.array([0.0, 1.0])
IRK2LBOrder = 2

IRK2LB = [IRK2LBMatrix, IRK2LBWeights, IRK2LBNodes, IRK2LBOrder]


IRK4LBMatrix = np.array([[1.0 / 6.0, -1.0 / 6.0, 0.0],
                         [1.0 / 6.0, 1.0 / 3.0, 0.0],
                         [1.0 / 6.0, 5.0 / 6.0, 0.0]])
IRK4LBWeights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
IRK4LBNodes = np.array([0.0 , 1.0 / 2.0 , 1.0])
IRK4LBOrder = 4

IRK4LB = [IRK4LBMatrix, IRK4LBWeights, IRK4LBNodes, IRK4LBOrder]


IRK2LCMatrix = np.array([[0.5, -0.5],
                         [0.5, 0.5]])
IRK2LCWeights = np.array([0.5, 0.5])
IRK2LCNodes = np.array([0.0, 1.0])
IRK2LCOrder = 2

IRK2LC = [IRK2LCMatrix, IRK2LCWeights, IRK2LCNodes, IRK2LCOrder]


IRK4LCMatrix = np.array([[1.0 / 6.0, -1.0 / 3.0, 1.0 / 6.0],
                         [1.0 / 6.0, 5.0 / 12.0, -1.0 / 12.0],
                         [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]])
IRK4LCWeights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
IRK4LCNodes = np.array([0.0 , 1.0 / 2.0 , 1.0])
IRK4LCOrder = 4

IRK4LC = [IRK4LCMatrix, IRK4LCWeights, IRK4LCNodes, IRK4LCOrder]

IRK2GLMatrix = np.array([[1.0 / 4.0, 1.0 / 4.0 - np.sqrt(3.0) / 6.0],
                         [1.0 / 4.0 + np.sqrt(3.0) / 6.0, 1.0 / 4.0]])
IRK2GLWeights = np.array([1.0 / 2.0, 1.0 / 2.0])
IRK2GLNodes = np.array([1.0 / 2.0 - np.sqrt(3.0) / 6.0,
                        1.0 / 2.0 + np.sqrt(3.0) / 6.0])
IRK2GLOrder = 4

IRK2GL = [IRK2GLMatrix, IRK2GLWeights, IRK2GLNodes, IRK2GLOrder]


IRK3GLMatrix = np.array([[5.0 / 36.0, 2.0 / 9.0 - np.sqrt(15.0) / 15.0,
                         5.0 / 36.0 - np.sqrt(15.0) / 30.0],
                        [5.0 / 36.0 + np.sqrt(15.0) / 24.0, 2.0 / 9.0,
                         5.0 / 36.0 - np.sqrt(15.0) / 24.0],
                        [5.0 / 36.0 + np.sqrt(15.0) / 30.0, 
                         2.0 / 9.0 + np.sqrt(15.0) / 15.0, 5.0 / 36.0]])
IRK3GLWeights = np.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
IRK3GLNodes = np.array([1.0 / 2.0 - np.sqrt(15.0) / 10.0, 1.0 / 2.0,
                        1.0 / 2.0 + np.sqrt(15.0) / 10.0])
IRK3GLOrder = 6

IRK3GL = [IRK3GLMatrix, IRK3GLWeights, IRK3GLNodes, IRK3GLOrder]


DIRK3CMatrix = np.array([[1.0 / 2.0 + 1.0 / (2.0 * np.sqrt(3)), 0.0],
                         [- 1.0 / np.sqrt(3), 
                          1.0 / 2.0 + 1.0 / (2.0 * np.sqrt(3))]])
DIRK3CWeights = np.array([0.5, 0.5])
DIRK3CNodes = np.array([1.0 / 2.0 + 1.0 / (2.0 * np.sqrt(3)),
                        1.0 / 2.0 - 1.0 / (2.0 * np.sqrt(3))])
DIRK3COrder = 3

DIRK3C = [DIRK3CMatrix, DIRK3CWeights, DIRK3CNodes, DIRK3COrder]


a = 2 * np.cos(np.pi / 18.0) / np.sqrt(3)
DIRK4CMatrix = np.array([[(1.0 + a) / 2.0, 0.0, 0.0],
                         [- a / 2.0, (1.0 + a) / 2.0, 0.0],
                         [1.0 + a, -(1.0 + 2.0 * a), (1.0 + a) / 2.0]])
DIRK4CWeights = np.array([1.0 / (6.0 * a **2), 1.0 - 1.0 / (3.0 * a **2),
                          1.0 / (6.0 * a **2)])
DIRK4CNodes = np.array([(1.0 + a) / 2.0, 1.0 / 2.0, (1.0 - a) / 2.0])
DIRK4COrder = 4

DIRK4C = [DIRK4CMatrix, DIRK4CWeights, DIRK4CNodes, DIRK4COrder]

#DIRK5NASAMatrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                            [1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                             0.0],
#                            [1748874742213.0 / 5795261096931.0,
#                             1748874742213.0 / 5795261096931.0, 1.0 / 4.0,
#                             0.0, 0.0, 0.0, 0.0, 0.0],
#                            [2426486750897.0 / 12677310711630.0,
#                             2426486750897.0 / 12677310711630.0,
#                             - 783385356511.0 / 7619901499812, 1.0 / 4.0,
#                             0.0, 0.0, 0.0, 0.0],
#                            [1616209367427.0 / 5722977998639.0,
#                             1616209367427.0 / 5722977998639.0,
#                             - 211896077633.0 / 5134769641545.0,
#                             464248917192.0 / 17550087120101.0, 1.0 / 4.0,
#                             0.0, 0.0, 0.0],
#                            [1860464898611.0 / 7805430689312.0,
#                             1825204367749.0 / 7149715425471.0,
#                             - 1289376786583.0 / 6598860380111.0,
#                             55566826943.0 / 2961051076052.0,
#                             1548994872005.0 / 13709222415197.0, 1.0 / 4.0,
#                             0.0, 0.0],
#                            [1783640092711.0 / 14417713428467.0,
#                             - 5781183663275.0 / 18946039887294.0,
#                             57847255876685.0 / 10564937217081.0,
#                             29339178902168.0 / 9787613280015.0,
#                             122011506936853.0 / 12523522131766.0,
#                             - 60418758964762.0 / 9539790648093.0, 1.0 / 4.0,
#                             0.0],
#                            [3148564786223.0 / 23549948766475.0,
#                             - 4152366519273.0 / 20368318839251.0,
#                             - 143958253112335.0 / 33767350176582.0,
#                             16929685656751.0 / 6821330976083.0,
#                             37330861322165.0 / 4907624269821.0,
#                             - 103974720808012.0 / 20856851060343.0,
#                             - 93596557767.0 / 4675692258479.0, 1.0 / 4.0]])
#DIRK5NASAWeights = np.array([3148564786223.0 / 23549948766475.0,
#                             - 4152366519273.0 / 20368318839251.0,
#                             - 143958253112335.0 / 33767350176582.0,
#                             16929685656751.0 / 6821330976083.0,
#                             37330861322165.0 / 4907624269821.0,
#                             103974720808012.0 / 20856851060343.0,
#                             - 93596557767.0 / 4675692258479.0, 1.0 / 4.0])
#DIRK5NASANodes = np.array([0.0, 1.0 / 2.0, (2.0 + np.sqrt(2)) / 4.0,
#                           53.0 / 100.0, 4.0 / 5.0, 17.0 / 25.0, 1.0, 1.0])
#
#DIRK5NASA = [DIRK5NASAMatrix, DIRK5NASAWeights, DIRK5NASANodes]

SDIRK2Matrix = np.array([[1 + np.sqrt(2) / 2.0, 0.0],
                         [- np.sqrt(2) / 2.0, 1 + np.sqrt(2) / 2.0]])
SDIRK2Weights = np.array([- np.sqrt(2) / 2.0, 1 + np.sqrt(2) / 2.0])
SDIRK2Nodes = np.array([1 + np.sqrt(2) / 2.0, 1.0])
SDIRK2Order = 2

SDIRK2 = [SDIRK2Matrix, SDIRK2Weights, SDIRK2Nodes, SDIRK2Order]

def f(x):
    return (x ** 3) - 3.0 * (x ** 2) + (3.0 / 2.0) * x - (1.0 / 6.0)

alpha = sop.newton(f, 0.3)
tau = (1.0 + alpha) / 2.0
b1 = - (6.0 * (alpha ** 2) - 16.0 * alpha + 1.0) / 4.0
b2 = (6 * (alpha **2) - 20.0 * alpha + 5.0) / 4.0

SDIRK3Matrix = np.array([[alpha, 0.0, 0.0],
                         [tau - alpha, alpha, 0.0],
                         [b1, b2, alpha]])
SDIRK3Weights = np.array([b1, b2, alpha])
SDIRK3Nodes = np.array([alpha, tau, 1.0])
SDIRK3Order = 3

SDIRK3 = [SDIRK3Matrix, SDIRK3Weights, SDIRK3Nodes, SDIRK3Order] 
