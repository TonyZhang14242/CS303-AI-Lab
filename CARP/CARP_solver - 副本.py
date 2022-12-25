import argparse
#import random
import re
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
        self.sorted_dists = []
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

    def read_args(self):
        parser = argparse.ArgumentParser(description='Personal information')
        parser.add_argument('-t', dest='time', type=int, help='maximum time run')
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
        self.greedy()

    def greedy(self):
        demand_matrix_cp = self.demand_matrix.copy()
        demand_sum = np.sum(self.demand_matrix)//2
        while demand_sum>0:
            task = []
            load = 0
            cur_position = self.depot
            perm = list(range(1, self.vertices_num+1))
            while load<self.capacity:
                sorted_dist = self.sorted_dists[cur_position]
                if load>0 and cur_position==self.depot:
                    break
                for nodes in sorted_dist:
                    n = nodes[0]
                    shuffle(perm)
                    for dest in perm:
                        demand = demand_matrix_cp[n][dest]
                        if demand>0 and load+demand<=self.capacity:
                            demand_sum -= self.demand_matrix[n][dest]
                            load+= self.demand_matrix[n][dest]
                            demand_matrix_cp[n][dest] = demand_matrix_cp[dest][n] = 0
                            task.append((n, dest))
                            cur_position = dest
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
        self.generate_ans(q)
        #return q

    def generate_ans(self, q):
        res = 's '
        for task in self.task_list:
            res += '0,'
            for t in task:
                res += '('+str(t[0])+','+str(t[1])+')'
                res+= ','
            res+= '0,'
        res = res[0:len(res)-1]
        print(res)
        print('q '+str(q))



if __name__ == '__main__':
    carp = CARP()
    args = carp.read_args()
    sample_path = args.path
    time = args.time
    seed = args.seed
    carp.read_sample_file(sample_path)
    #print( sample_path, time, seed)
