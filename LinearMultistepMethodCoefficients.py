# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:25:23 2017

@author: Alastair Wiseman
"""
import numpy as np

AB1A = np.array([-1.0 , 1.0])
AB1B = np.array([1.0, 0.0])
AB1Order = 1

AB1 = [AB1A, AB1B, AB1Order]

AB2A = np.array([0.0, -1.0 , 1.0])
AB2B = np.array([-0.5 , 1.5, 0.0])
AB2Order = 2

AB2 = [AB2A, AB2B, AB2Order]

AB3A = np.array([0.0, 0.0, -1.0 , 1.0])
AB3B = np.array([5.0 / 12.0, - 4.0 / 3.0, 23.0 / 12.0, 0.0])
AB3Order = 3

AB3 = [AB3A, AB3B, AB3Order]

AB4A = np.array([0.0, 0.0, 0.0,-1.0, 1.0])
AB4B = np.array([- 3.0 / 8.0, 37.0 / 24.0, -59.0 / 24.0, 55.0 / 24, 0.0])
AB4Order = 4

AB4 = [AB4A, AB4B, AB4Order]

AB5A = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
AB5B = np.array([251.0 / 720.0, -637.0 / 360.0, 109.0 / 30.0, 
                 -1387.0 / 360.0, 1901.0 / 720.0, 0.0])
AB5Order = 5

AB5 = [AB5A, AB5B, AB5Order]

AM1AA = np.array([-1.0, 1.0])
AM1AB = np.array([0.0, 1.0])
AM1AOrder = 1

AM1A = [AM1AA, AM1AB, AM1AOrder]

AM1BA = np.array([-1.0, 1.0])
AM1BB = np.array([0.5, 0.5])
AM1BOrder = 2

AM1B = [AM1BA, AM1BB, AM1BOrder]

AM2A = np.array([0.0, -1.0, 1.0])
AM2B = np.array([-1.0 / 12.0, 2.0 / 3.0, 5.0 / 12.0])
AM2Order = 3

AM2 = [AM2A, AM2B, AM2Order]

AM3A = np.array([0.0, 0.0, -1.0, 1.0])
AM3B = np.array([1.0 / 24.0, -5.0 / 24.0, 19.0 / 24.0, 3.0 / 8.0])
AM3Order = 4

AM3 = [AM3A, AM3B, AM3Order]

AM4A = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
AM4B = np.array([-19.0 / 720.0, 106.0 / 720, -264.0 / 720.0, 646.0 / 720.0,
                 251.0 / 720])
AM4Order = 5

AM4 = [AM4A, AM4B, AM4Order]

BDF1A = np.array([-1.0, 1.0])
BDF1B = np.array([0.0, 1.0])
BDF1Order = 1

BDF1 = [BDF1A, BDF1B, BDF1Order]

BDF2A = np.array([1.0 / 3.0, -4.0 / 3.0, 1.0])
BDF2B = np.array([0.0, 0.0, 2.0 / 3.0])
BDF2Order = 2

BDF2 = [BDF2A, BDF2B, BDF2Order]

BDF3A = np.array([-2.0 / 11.0, 9.0 / 11.0, -18.0 / 11.0, 1.0])
BDF3B = np.array([0.0, 0.0, 0.0, 6.0 / 11.0])
BDF3Order = 3

BDF3 = [BDF3A, BDF3B, BDF3Order]

BDF4A = np.array([3.0 / 25.0, -16.0 / 25.0, 36.0 / 25.0, -48.0 / 25.0, 1.0])
BDF4B = np.array([0.0, 0.0, 0.0, 0.0, 12.0 / 25.0])
BDF4Order = 4

BDF4 = [BDF4A, BDF4B, BDF4Order]

BDF5A = np.array([-12.0 / 137.0, 75.0 / 137.0, -200.0 / 137.0, 300 / 137.0,
                  -300.0 / 137.0, 1.0])
BDF5B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 60.0 / 137.0])
BDF5Order = 5

BDF5 = [BDF5A, BDF5B, BDF5Order]

BDF6A = np.array([10.0 / 147.0, -72.0 / 147.0, 225.0 / 147.0, -400.0 / 147.0,
                  450.0 / 147.0, -360.0 / 147.0, 1.0])
BDF6B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0 / 147.0])
BDF6Order = 6

BDF6 = [BDF6A, BDF6B, BDF6Order]