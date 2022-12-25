import argparse
import random
#import random
import re
import threading

import numpy as np
from random import *
import time
import sys
import getopt
import copy
import multiprocessing
class CARP:
    def __init__(self):
        self.map_matrix = np.array([0])
        self.locking = threading.Lock()
        self.sorted_dists = []
        self.current_opt_q = 2147483647
        self.current_opt_s = []
        self.task_list = []
        self.demand_matrix = np.array([0])
        self.map = []
        self.adj = []
        self.shortest_path = np.array([0])
        self.vertices_num = 0
        self.depot = 0
        self.requ_edge_num = 0
        self.non_requ_edge_num = 0
        self.vehicle_num = 0
        self.capacity = 0
        self.totalCost_require_edge = 0
        self.genetic_min_q = 2147483647
        self.genetic_opt_s = []

    def read_args(self):
        parser = argparse.ArgumentParser(description='Personal information')
        parser.add_argument('-t', dest='demand_time', type=int, help='maximum time run')
        parser.add_argument('-s', dest='seed', type=int, help='random seed')
        parser.add_argument(dest='path', type=str, help='Age of the candidate')
        return parser.parse_args()

    def read_sample_file(self,path):
        with open(path, 'r') as f:
            text = f.read()
        text_line = text.split('\n')
        for i in range(8):
            if text_line[i].split(' : ')[0]=='VERTICES':
                self.vertices_num = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='DEPOT':
                self.depot = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='REQUIRED EDGES':
                self.requ_edge_num = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='NONE-REQUIRED EDGES':
                self.non_requ_edge_num = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='VEHICLES':
                self.vehicle_num = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='CAPACITY':
                self.capacity = int(text_line[i].split(' : ')[1])
            if text_line[i].split(' : ')[0]=='TOTAL COST OF REQUIRED EDGES':
                self.totalCost_require_edge = int(text_line[i].split(' : ')[1])
        for i in range(self.vertices_num+1):
            self.map.append([])

        self.map_matrix = np.zeros((self.vertices_num + 1, self.vertices_num + 1), dtype=int)
        self.shortest_path = np.ones((self.vertices_num + 1, self.vertices_num + 1), dtype=int)
        self.demand_matrix = np.zeros((self.vertices_num+1, self.vertices_num+1), dtype=int)

        for i in range(9, len(text_line)-1):
            #n1, n2, cost, demand = text_line[i].split(' ')
            tmp = re.split(r'[ ]+', text_line[i])
            n1 = int(tmp[0])
            n2 = int(tmp[1])
            cost = int(tmp[2])
            demand = int(tmp[3])
            self.map_matrix[n1][n2] = cost
            self.map_matrix[n2][n1] = cost
            self.demand_matrix[n1][n2] = demand
            self.demand_matrix[n2][n1] = demand
            self.map[n1].append(list([n2, cost, demand]))
            self.map[n2].append(list([n1, cost, demand]))
            #print(tmp)
        for i in range(self.vertices_num+1):
            for j in range(self.vertices_num+1):
                if self.map_matrix[i][j] == 0:
                    self.shortest_path[i][j] = 2147483647
                else:
                    self.shortest_path[i][j] = self.map_matrix[i][j]
        for i in range(self.vertices_num+1):
            self.shortest_path[i][i] = 0
        #print(self.map)
        self.floyd()
        self.arrange_sorted_dists()


    def floyd(self):
        #print([[self.map_matrix[i][j] for j in range(1, self.vertices_num+1)] for i in range(1, self.vertices_num+1)])
        #print(self.shortest_path)
        for k in range(1, self.vertices_num+1):
            for i in range(1, self.vertices_num+1):
                for j in range(1, self.vertices_num+1):
                    if self.shortest_path[i][j]-self.shortest_path[k][j]> self.shortest_path[i][k]:
                        self.shortest_path[i][j] = self.shortest_path[i][k]+self.shortest_path[k][j]
        #self.shortest_path = [[self.shortest_path[i][j] for j in range(1, self.vertices_num+1)] for i in range(1, self.vertices_num+1)]
        #print([[self.shortest_path[i][j] for j in range(1, self.vertices_num+1)] for i in range(1, self.vertices_num+1)])

    def arrange_sorted_dists(self):
        #print([[self.shortest_path[i][j] for j in range(1, self.vertices_num+1)] for i in range(1, self.vertices_num+1)])
        #print(self.shortest_path)
        self.sorted_dists.append([])
        for i in range(1, self.vertices_num+1):
            ids = range(1, self.vertices_num+1)
            dist = self.shortest_path[i][1:]
            d = dict(zip(ids, dist))
            sorted_dist = sorted(d.items(), key=lambda d: d[1])
            self.sorted_dists.append(sorted_dist)

        #print(self.sorted_dists)


    def multi_solve(self):
        start_time = time.time()
        while time.time() - start_time + 10 < float(demand_time):
            self.locking.acquire()
            self.greedy()
            #print(threading.currentThread())
            self.locking.release()
            time.sleep(0)

        #self.generate_ans()

    def greedy(self):
        demand_matrix_cp = self.demand_matrix.copy()
        demand_sum = np.sum(self.demand_matrix)//2
        demand_sum_cur = 0
        self.task_list = []
        while demand_sum_cur<demand_sum:
            task = []
            load = 0
            cur_position = self.depot
            perm = [i for i in range(1,self.vertices_num+1)]
            while load<self.capacity:
                sorted_dist = self.sorted_dists[cur_position]
                if load>0 and cur_position==self.depot:
                    break
                for nodes in sorted_dist:
                    n = nodes[0]
                    shuffle(perm)
                    for dest in perm:
                        demand = demand_matrix_cp[n][dest]
                        if demand>0 and load<=self.capacity-demand:
                            demand_sum_cur += self.demand_matrix[n][dest]
                            demand_matrix_cp[n][dest] = demand_matrix_cp[dest][n] = 0
                            task.append((n, dest))
                            cur_position = dest
                            load+= self.demand_matrix[n][dest]
                            break
                    else:
                        continue
                    break
                else:
                    break
            self.task_list.append(task)
        #print(self.task_list)
        self.count_q()

        #print(demand_sum)

    def count_q(self):
        q = 0
        for task in self.task_list:
            cost = 0
            cur_pos = self.depot
            for t in task:
                cost += self.shortest_path[cur_pos][t[0]]
                cost += self.map_matrix[t[0]][t[1]]
                cur_pos = t[1]
            cost += self.shortest_path[cur_pos][self.depot]
            q += cost
        #print(q)
        if q < self.current_opt_q:
            self.current_opt_q = q
            self.current_opt_s = self.task_list
        #self.generate_ans(q)
        #return q

    def generate_ans(self):
        res = 's '
        for task in self.current_opt_s:
            res += '0,'
            for t in task:
                res += '('+str(t[0])+','+str(t[1])+')'
                res+= ','
            res+= '0,'
        res = res[0:len(res)-1]
        print(res)
        print('q '+str(self.current_opt_q))

    def multithread(self):
        l = []
        thread1 = threading.Thread(target=self.multi_solve)
        thread2 = threading.Thread(target=self.multi_solve)
        thread3 = threading.Thread(target=self.multi_solve)
        thread4 = threading.Thread(target=self.multi_solve)
        thread5 = threading.Thread(target=self.multi_solve)
        thread6 = threading.Thread(target=self.multi_solve)
        thread7 = threading.Thread(target=self.multi_solve)
        thread8 = threading.Thread(target=self.genetics)
        l.append(thread1)
        l.append(thread2)
        l.append(thread3)
        l.append(thread4)
        l.append(thread5)
        l.append(thread6)
        l.append(thread7)
        l.append(thread8)
        for t in l:
            t.start()
        for t in l:
            t.join()

    def genetics(self):
        time.sleep(10)
        origin = copy.deepcopy(self.current_opt_s) #original generation
        chorosome_num = len(origin)
        sort_chorosome = sorted(origin, key=lambda o: self.calc_fitness(o))
        new_generation = sort_chorosome
        mutate_rate = 0.1
        min_q = 2147483647
        opt_s = []
        for li in range(100):
            generation = sorted(new_generation, key=lambda o: self.calc_fitness(o))
            if chorosome_num>=2:
                a = generation[0]
                b = generation[1]
                new_a, new_b = self.cross(a,b)
                a_or_b = choice([0, 1])
                if a_or_b == 0:
                    new_a = self.mutation(mutate_rate, new_a)
                else:
                    new_b = self.mutation(mutate_rate, new_b)
                if self.isValid(new_a) and self.isValid(new_b):
                    a = new_a
                    b = new_b
                new_generation = []
                new_generation.append(a)
                new_generation.append(b)
                for i in range(2, len(generation)):
                    new_generation.append(generation[i])
            q = 0
            for task in new_generation:
                cost = 0
                cur_pos = self.depot
                for t in task:
                    cost += self.shortest_path[cur_pos][t[0]]
                    cost += self.map_matrix[t[0]][t[1]]
                    cur_pos = t[1]
                cost += self.shortest_path[cur_pos][self.depot]
                q += cost
            if q < min_q:
                min_q =q
                opt_s = copy.deepcopy(new_generation)
        self.genetic_min_q = min_q
        self.genetic_opt_s = opt_s



    def cross(self, a, b):
        a_cross_idx = randint(0, len(a)-1)
        b_cross_idx = randint(0, len(b)-1)
        new_a = []
        new_b = []
        a_to_b = a[a_cross_idx:]
        b_to_a = b[b_cross_idx:]
        for i in range(0, a_cross_idx):
            new_a.append(a[i])
        for i in b_to_a:
            new_a.append(i)
        for j in range(0, b_cross_idx):
            new_b.append(b[j])
        for j in a_to_b:
            new_b.append(j)
        return new_a, new_b

    def mutation(self, rate, cho):
        ran_rate = uniform(0, 1)
        cho_cp = []
        if ran_rate <= rate:
            ran_idx = randint(0, len(cho)-1)
            selected = cho[ran_idx]
            selected_0 = selected[0]
            selected_1 = selected[1]
            after_mutation = (selected_1, selected_0)
            for i in range(len(cho)):
                if i!=ran_idx:
                    cho_cp.append(cho[i])
                else:
                    cho_cp.append(after_mutation)
        else:
            cho_cp = cho
        #print(cho_cp)
        return cho_cp


    def isValid(self, cho):
        demand = 0
        for i in cho:
            demand += self.demand_matrix[i[0]][i[1]]
        return demand <= self.capacity


    def calc_fitness(self, chorosome):
        cost = 0
        cur_pos = self.depot
        for t in chorosome:
            cost += self.shortest_path[cur_pos][t[0]]
            cost += self.map_matrix[t[0]][t[1]]
            cur_pos = t[1]
        cost += self.shortest_path[cur_pos][self.depot]
        return cost


if __name__ == '__main__':
    carp = CARP()
    args = carp.read_args()
    sample_path = args.path
    demand_time = args.demand_time
    seed = args.seed
    carp.read_sample_file(sample_path)
    carp.multithread()
    #print(carp.genetic_min_q, carp.current_opt_q)
    if carp.genetic_min_q < carp.current_opt_q:
        carp.current_opt_q = carp.genetic_min_q
        carp.current_opt_s = carp.genetic_opt_s
    carp.generate_ans()
    #print( sample_path, time, seed)
