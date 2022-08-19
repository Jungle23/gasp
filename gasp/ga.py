'''
@File          :  ga.py 
@Author        :  Jiang
@Date          :  2022/8/15 下午5:11 
@Description   :  
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import math
import seaborn as sns
import time
import os
import datetime
import imageio
import multiprocessing

today = str(datetime.date.today())[5:]  # 日期戳
# 设置字体，仿宋：FangSong，黑体：SimHei，宋体：SimSun，微软雅黑：Microsoft YaHei
matplotlib.rc("font", family='Microsoft YaHei ')
matplotlib.rcParams['xtick.direction'] = 'in'  # 刻度标签方向：in, out
matplotlib.rcParams['ytick.direction'] = 'in'


class Individual(object):

    def __init__(self):

        return

class Pop(object):

    def __init__(self):

        return


class Ga(object):

    def __init__(self, objects='1'):
        return

