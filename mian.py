import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import datetime
import spga.facility as sf
import spga.ga as sg


plan_sga_1 = [137, 68, 89, 108, 40, 174, 129, 121, 156, 208, 91, 79, 90, 36, 128, 146, 173, 157, 125, 103, 119, 75,
              134, 100, 113, 106, 15, 141, 138, 187, 101, 142, 191, 95, 74, 81, 102, 80, 132]  # DUR=1.100
plan_sga_2 = [124, 80, 64, 62, 196, 140, 158, 32, 142, 151, 86, 63, 76, 40, 130, 163, 119, 138, 125, 94, 154, 147,
              120, 67, 105, 88, 38, 156, 220, 128, 78, 146, 218, 91, 68, 83, 84, 79, 90]  # DUR=1.00418
plan_sga_3 = [60, 127, 171, 145, 228, 113, 106, 28, 73, 74, 138, 154, 166, 7, 65, 83, 101, 75, 104, 151, 96, 153,
              76, 141, 139, 112, 91, 78, 197, 207, 157, 66, 43, 143, 158, 168, 152, 155, 130]  # DUR=1.00424

# 1. define a facility by .json file to be optimized
f1 = sf.Facility(facility_info='hengde.json')
# 2. generate random plans as the initial population for genetic algorithms
rp = f1.random_plan(num=200)
# 3. use a Plans class to manage the plans
p0 = sf.Plans(facility_ob=f1, dnas=rp)
# 4. define a genetic algorithm class
s1 = sg.Sga(pop=p0, cross_rate=0.8, mutate_rate=0.1)
n1 = sg.Nsga2(pop=p0, cross_rate=0.8, mutate_rate=0.1)
# 5. start the optimization
N_GEN = 200
# 5.1 single object optimization with Simple Genetic Algorithm
# best_dur = []  # record DUR of the best plan on generations
# begin_time = time.time()
# for gen in range(0, N_GEN):
#     best_dur.append(s1.pop.objects[0].min())
#     print("Gen:", gen, "Best DUR:", best_dur[-1], " time:", time.time()-begin_time)
#     if gen % 20 == 0:
#         plt.plot(list(np.arange(gen+1)), best_dur)
#         plt.xlabel("Generation")
#         plt.ylabel("DUR")
#         plt.grid()
#         plt.show()
#     s1.evolve()
# 5.2 multi-objects optimization with NSGA2
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

