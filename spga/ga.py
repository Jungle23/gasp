import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
from .facility import Plans, Facility

today = str(datetime.date.today())[5:]  # date

class Population:
    def __init__(self, plans: Plans):
        self.plans = plans
    @property
    def dnas(self):
        return self.plans.dnas
    @property
    def objects(self):
        return self.plans.objects
    @property
    def size(self):
        return self.plans.size

def cro_ordered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    swap_index = np.arange(a, b+1, 1)
    head_index = np.arange(0, a, 1)
    bot_index = np.arange(b+1, size, 1)
    swap1, swap2 = ind1[swap_index], ind2[swap_index]
    rep1 = np.concatenate((ind1[bot_index], ind1[head_index], ind1[swap_index]))
    rep2 = np.concatenate((ind2[bot_index], ind2[head_index], ind2[swap_index]))
    # 去重
    pure1 = np.setdiff1d(rep1, swap2, assume_unique=True)
    pure2 = np.setdiff1d(rep2, swap1, assume_unique=True)
    off1 = np.concatenate((pure2[size-swap_index[-1]-1:size-swap_index[-1]-1 + a],
                           swap1, pure2[:size-swap_index[-1]-1]))
    off2 = np.concatenate((pure1[size-swap_index[-1]-1:size-swap_index[-1]-1 + a],
                           swap2, pure1[:size-swap_index[-1]-1]))
    return off1, off2

def mut_twopointexhcange(ind1):
    size = len(ind1)
    a, b = random.sample(range(size), 2)
    ind1[a], ind1[b] = ind1[b], ind1[a]
    return ind1

def mut_scramble(ind1):
    size = len(ind1)
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a
    mut_idx = np.arange(a, b+1, 1)
    mut_part = ind1[mut_idx]
    np.random.shuffle(mut_part)
    ind1[mut_idx] = mut_part
    return ind1

def asf_direct(in_plans: Plans):  # assign fitness to its object function values
    return np.exp(-10 * in_plans.objects[0])

def asf_rank(in_plans: Plans, selection_pressure):  # assign fitness according to its rank
    values = 1 / in_plans.objects[0]  # for a minimizing problem
    pos = np.zeros(in_plans.size)
    for i in range(0, in_plans.size):
        pos[np.argsort(values)[i]] = i + 1
    fitness_pos = 2 - selection_pressure + (2 * (selection_pressure - 1) * (pos - 1)) / in_plans.size
    return fitness_pos

def sel_roulette(in_plans: Plans, fitness):
    idx = np.random.choice(np.arange(in_plans.size), size=in_plans.size, replace=True,
                           p=fitness / fitness.sum())
    return idx, in_plans.dnas[idx]

def sel_tournament(in_plans: Plans):

    return

class Sga:
    def __init__(self, pop: Plans, cross_rate, mutate_rate):
        self.pop = pop
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

    def make_new_pop(self):
        return

    def evolve(self):
        # 赋予适应度
        fitness = asf_rank(in_plans=self.pop, selection_pressure=1.7)
        # 选择操作
        idx, offspring_dnas = sel_roulette(self.pop, fitness=fitness)
        self.pop.dnas = offspring_dnas
        # 交叉操作
        for i, j in zip(np.arange(0, self.pop.size, 2), np.arange(1, self.pop.size, 2)):
            if random.random() < self.cross_rate:
                ch_dna1, ch_dna2 = cro_ordered(offspring_dnas[i], offspring_dnas[j])
                # 将交叉操作后的子代替换父代
                offspring_dnas[i], offspring_dnas[j] = ch_dna1, ch_dna2
        # 变异操作
        for i in range(self.pop.size):
            if random.random() < self.mutate_rate:
                ch_dna = mut_scramble(offspring_dnas[i])
                offspring_dnas[i] = ch_dna
        # 更新种群  获得新的dnas：offspring_dnas
        self.pop.dnas = offspring_dnas  # 此处仍然需要使用到facility的相关信息
        # new_size = self.pop.size()
        # new_objects = self.pop.objects()
        return

class Nsga2:
    def __init__(self, pop: Plans, cross_rate, mutate_rate):
        self.pop = pop
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate

    def make_new_pop(self):
        # 选择：二元锦标赛 (有待完善)
        # 跳过选择阶段，进行交叉和变异
        copy_dnas = self.pop.dnas.copy()
        # 交叉操作
        for i, j in zip(np.arange(0, self.pop.size, 2), np.arange(1, self.pop.size, 2)):
            if random.random() < self.cross_rate:
                ch_dna1, ch_dna2 = cro_ordered(copy_dnas[i], copy_dnas[j])
                # 替换
                copy_dnas[i], copy_dnas[j] = ch_dna1, ch_dna2
        # 变异操作
        for i in range(self.pop.size):
            if random.random() < self.mutate_rate:
                ch_dna = mut_scramble(copy_dnas[i])
                copy_dnas[i] = ch_dna
        return copy_dnas

    def non_dominated_sort(self):
        # 计算合并种群的目标函数值
        objects_value = self.pop.objects
        objects_sort = np.zeros_like(objects_value)
        size_sort = self.pop.size
        # 对目标函数值进行排序
        for i in range(size_sort):
            # 对dur从小到大排序
            objects_sort[0][np.argsort(objects_value[0])[i]] = i + 1
            # 对avgd从大到小排序
            objects_sort[1][np.argsort(objects_value[1])[::-1][i]] = i + 1
        # 开始非支配排序
        dominate_map = np.zeros([size_sort, size_sort])
        # 针对两个目标函数时
        for i in range(size_sort):
            # DUR
            temp_dominate_u = np.where(objects_sort[0] > objects_sort[0][i], 1, 0)
            # avgD
            temp_dominate_d = np.where(objects_sort[1] > objects_sort[1][i], 1, 0)
            # 找出个体i所支配的个体
            dominate_map[i] = (temp_dominate_u * temp_dominate_d).astype(int)
        dominate_map_df = pd.DataFrame(dominate_map).astype(int)
        non_dom_front = []
        # 循环，找出各个非支配前沿的索引
        temp_map = dominate_map_df
        for rank in range(size_sort):
            temp_list = temp_map[(temp_map == 0).all(axis=0)].index.to_list()
            non_dom_front.append(temp_list)
            temp_map.drop(columns=temp_list, index=temp_list, inplace=True)
            if len(temp_map) == 0:
                break
        return non_dom_front

    def cal_crowding_distance(self, front_idx):  # 计算临界层的聚集距离
        combine_objects = self.pop.objects
        # 将需要被计算聚集距离的个体索引、目标函数值堆叠起来
        ff = np.vstack([front_idx, combine_objects[:, front_idx]])
        # 按照第一个目标函数值进行排序
        ff_sorted = ff[:, ff[1].argsort()]
        # 计算聚集距离
        c_distance = np.zeros(len(front_idx))
        c_distance[0] = 999999
        c_distance[-1] = 999999
        # 两个目标函数
        for i in range(1, len(front_idx)-1):
            c_distance[i] = abs(ff_sorted[1, i-1] - ff_sorted[1, i+1])+abs(ff_sorted[2, i-1] - ff_sorted[2, i+1])
        # 统计所有的目标函数（三个目标函数时，需要确认此种计算方法的正确性）
        # for i in range(1, len(front_idx) - 1):
        #     temp_distance = 0
        #     for obj in range(0, len(combine_objects)):
        #         temp_distance = temp_distance + abs(ff_sorted[1+obj, i-1] - ff_sorted[1+obj, i+1])
        #     c_distance[i] = temp_distance
        # 合并起来，获得所有个体的目标函数值以及聚集距离
        fd = np.vstack([ff_sorted, c_distance])
        fd_sorted = fd[:, fd[-1].argsort()[::-1]]
        return fd_sorted[0].astype(int)

    def evolve(self):
        # make new pop
        new_pop_dnas = self.make_new_pop()
        combine_pop_dnas = np.vstack([self.pop.dnas, new_pop_dnas])
        # 更新，得到合并后的种群dna
        self.pop.dnas = combine_pop_dnas
        # 获取size和objects属性
        combine_size = self.pop.size  # 2 pop size
        combine_objects = self.pop.objects
        # 对合并后的种群进行非支配排序，获得排序结果的索引，通过索引来填充新的个体到新一代种群中
        non_dom_idx = self.non_dominated_sort()
        # 从非支配排序的索引中选出一定数量的个体，来填充进下一代
        merged_pop_idx = []
        rank = 0
        # 填充个体
        while len(merged_pop_idx) <= int(combine_size/2):
            merged_pop_idx = merged_pop_idx + non_dom_idx[rank]
            rank = rank + 1
        # 此时 merged_pop_list 元素数目会超出种群大小 self.pop_size
        # 需要把最后一次加上的 元素 去除，再将它们进行 密度计算，并排序，最后取需要的部分，填充进 merged_pop_list
        # 去除 nd_front[rank] 元素
        rank = rank - 1
        # 去除临界层个体
        merged_pop_idx = merged_pop_idx[:-len(non_dom_idx[rank])]
        # 需要通过聚集密度，从临界层中选出所需解
        # 个数：
        num = int(combine_size/2) - len(merged_pop_idx)
        # 对临界层进行密度排序，返回聚集距离将序排列的序号
        distance_sort_idx = self.cal_crowding_distance(front_idx=non_dom_idx[rank])
        merged_pop_idx = merged_pop_idx + list(distance_sort_idx[:num])
        # 更新种群
        offspring_dnas = combine_pop_dnas[merged_pop_idx]
        self.pop.dnas = offspring_dnas
        # new_size = self.pop.size
        # new_objects = self.pop.objects
        return


if __name__ == '__main__':
    f1 = Facility(facility_info='../hengde.json')
    rp1 = f1.random_plan(num=10)
    rp2 = f1.random_plan(num=5)
    rp0 = f1.random_plan(num=200)

    p0 = Plans(facility_ob=f1, dnas=rp0)

    # sga:
    sga1 = Sga(pop=p0, cross_rate=0.8, mutate_rate=0.1)
    n1 = Nsga2(pop=p0, cross_rate=0.8, mutate_rate=0.1)

    best_dur = []
    N_GEN = 200
    # SGA的测试
    # for gen in range(0, N_GEN):
    #     dur_here = sga1.pop.objects[0]
    #     best_dur.append(dur_here.min())
    #     print("Gen:", gen, "Best DUR:", best_dur[-1])
    #     if gen % 20 == 0:
    #         plt.plot(list(np.arange(gen+1)), best_dur)
    #         plt.xlabel("Generation")
    #         plt.ylabel("DUR")
    #         plt.grid()
    #         plt.show()
    #     sga1.evolve()
    # NSGA2的测试
    begin_time = time.time()
    for gen in range(0, N_GEN):
        objects_here = n1.pop.objects
        print("Gen:", gen, " time:", time.time()-begin_time)
        if gen % 5 == 0:
            plt.scatter(objects_here[0], objects_here[1])
            plt.xlabel("DUR")
            plt.ylabel("AvgD")
            plt.title("Solutions on generation "+str(gen))
            plt.grid()
            plt.show()
        n1.evolve()

    print("ok")
