import numpy as np
import random
import time
import math

infinity = math.inf
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.directions = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def go(self, chessboard):
        self.candidate_list = self.find_valid_pos(chessboard, self.color)
        step = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] != 0:
                    step += 1
        if step > 57:
            bb = self.alpha_beta_search(chessboard, 8, 0, -infinity, + infinity, self.color)
            if bb is not None:
                self.candidate_list.append(bb)
        elif step > 47:
            bb = self.alpha_beta_search(chessboard, 4, 0, -infinity, + infinity, self.color)
            if bb is not None:
                self.candidate_list.append(bb)
        else:
            aa = self.alpha_beta_search(chessboard, 4, 0, -infinity, +infinity, self.color)
            if aa is not None:
                self.candidate_list.append(aa)

        # return self.candidate_list

    def find_valid_pos(self, chessboard, assign_color):
        valid_candidates = []
        # self.candidate_list.clear()
        # ==========================
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        opposite_color = 1
        if assign_color == 1:
            opposite_color = -1
        else:
            opposite_color = 1
        for pos in idx:
            pos_x, pos_y = pos
            for dire in self.directions:
                if 0 <= pos_x + dire[0] < self.chessboard_size and 0 <= pos_y + dire[1] < self.chessboard_size and \
                        chessboard[pos_x + dire[0], pos_y + dire[1]] != opposite_color:
                    continue
                coe = 1
                inner_opposite = 0
                while 0 <= pos_x + coe * dire[0] < self.chessboard_size and 0 <= pos_y + coe * dire[
                    1] < self.chessboard_size:
                    if chessboard[pos_x + coe * dire[0], pos_y + coe * dire[1]] == 0:
                        break
                    elif chessboard[pos_x + coe * dire[0], pos_y + coe * dire[1]] == assign_color:
                        if inner_opposite == 0:
                            break
                        else:
                            if valid_candidates.__contains__(pos) == False:
                                valid_candidates.append(pos)
                            break
                    else:
                        inner_opposite += 1
                        coe += 1
        # print(valid_candidates)

        return valid_candidates

    def alpha_beta_search(self, chessboard, depth_limit, depth_current, alpha, beta, assign_color):
        if depth_current == depth_limit:
            return self.evaluate_func(chessboard, assign_color)
        else:
            start = time.time()
            if depth_current % 2 == 0:  # max
                valid_positions = self.find_valid_pos(chessboard, assign_color)
                v, choice = -infinity, 0
                for idx in range(len(valid_positions)):
                    tmp_chessboard = self.execute_move(chessboard, valid_positions[idx], assign_color)
                    oppo_color = COLOR_BLACK
                    if assign_color == COLOR_BLACK:
                        oppo_color = COLOR_WHITE
                    v2 = self.alpha_beta_search(tmp_chessboard, depth_limit, depth_current + 1, alpha, beta, oppo_color)
                    if v2 is not None:
                        if v2 > v:
                            v, choice = v2, idx
                        if v >= beta:
                            if depth_current == 0:
                                return valid_positions[idx]
                            else:
                                return v
                        alpha = max(v, alpha)
                        
                if depth_current == 0:  # fail to prone
                    if len(valid_positions) != 0:
                        return valid_positions[choice]
                    return None
                return v
            else:  # min
                valid_positions = self.find_valid_pos(chessboard, assign_color)
                v, choice = +infinity, 0
                for idx in range(len(valid_positions)):
                    tmp_chessboard = self.execute_move(chessboard, valid_positions[idx], assign_color)
                    oppo_color = COLOR_BLACK
                    if assign_color == COLOR_BLACK:
                        oppo_color = COLOR_WHITE
                    v2 = self.alpha_beta_search(tmp_chessboard, depth_limit, depth_current + 1, alpha, beta, oppo_color)
                    if v2 is not None:
                        if v2 < v:
                            v, choice = v2, idx
                        if v <= alpha:
                            if depth_current == 0:
                                return valid_positions[idx]
                            else:
                                return v
                        beta = min(v, beta)
                        
                if depth_current == 0:  # fail to prone
                    if len(valid_positions) != 0:
                        return valid_positions[choice]
                    return None
                return v

    def execute_move(self, chessboard, choice, assign_color):
        chessboard_tmp = chessboard.copy()
        choice_x, choice_y = choice[0], choice[1]
        # print(choice)
        chessboard_tmp[choice_x][choice_y] = assign_color
        for dire in self.directions:
            coe = 1
            inner_opposite = 0
            while 0 <= choice_x + coe * dire[0] < self.chessboard_size and 0 <= choice_y + coe * dire[
                1] < self.chessboard_size:
                if chessboard_tmp[choice_x + coe * dire[0], choice_y + coe * dire[1]] == 0:
                    break
                elif chessboard_tmp[choice_x + coe * dire[0], choice_y + coe * dire[1]] == assign_color:
                    if inner_opposite == 0:
                        break
                    else:
                        for i in range(coe):
                            chessboard_tmp[choice_x + i * dire[0], choice_y + i * dire[1]] = assign_color
                        break

                else:
                    inner_opposite += 1
                    coe += 1

        # print(chessboard_tmp)
        return chessboard_tmp

    def evaluate_func(self, chessboard, color):
        # a = random.Random()
        step = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] != 0:
                    step += 1
        weight_sum = self.calc_weight_sum(chessboard, color)
        mobility_eval = self.mobility(chessboard, color) - self.mobility(chessboard, -color)
        disc_num_diff = self.count_assigned_number(chessboard, -color) - self.count_assigned_number(chessboard, color)
        stable_cnt_diff = self.stable_cnt(chessboard, -color) - self.stable_cnt(chessboard, color)
        if step < 15:
            return weight_sum * 30 + mobility_eval * 3 + disc_num_diff * 2 + stable_cnt_diff * 5
        elif step < 45:
            return weight_sum * 12 + mobility_eval * 100 + disc_num_diff * 20 #+ stable_cnt_diff * 5
        elif step <= 55:
            return weight_sum * 7 + mobility_eval * 4 + disc_num_diff * 50
        else:
            return weight_sum * 2 + disc_num_diff * 500 #+ stable_cnt_diff * 10

    def calc_weight_sum(self, chessboard, color):
        weight_matrix = [[-1000, 500, -8, 6, 6, -8, 500, -1000],
                         [500, -8, -15, 3, 3, -15, -8, 500],
                         [-8, -15, 4, 4, 4, 4, -15, -8],
                         [6,   3,  4, 0, 0, 4, 3, 6],
                         [6,   3,  4, 0, 0, 4, 3, 6],
                         [-8, -15, 4, 4, 4, 4, -15, -8],
                         [500, -8, -15, 3, 3, -15, -8, 500],
                         [-1000, 500, -8, 6, 6, -8, 500, -1000]]
        weight_sum = 0
        if color == 1:
            for i in range(self.chessboard_size):
                for j in range(self.chessboard_size):
                    weight_sum += weight_matrix[i][j] * chessboard[i][j]
        else:
            for i in range(self.chessboard_size):
                for j in range(self.chessboard_size):
                    weight_sum += weight_matrix[i][j] * chessboard[i][j]
            weight_sum = -weight_sum

        return weight_sum

    def mobility(self, chessboard, color):
        cnt = len(self.find_valid_pos(chessboard, color))
        return cnt

    def count_assigned_number(self, chessboard, color):
        num = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] == color:
                    num += 1
        return num

    def stable_cnt(self, chessboard, color):
        corner = [(0, 0), (0, 7), (7, 0), (7, 7)]
        dire = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        stable_discs = 0
        for i in range(4):
            stable_discs += self.stable_from_corner(chessboard, color, corner[i], dire[i][0], dire[i][1])
        print(stable_discs)
        return stable_discs

    def stable_from_corner(self, chessboard, color, corner, direcX, direcY):
        stable_discs = 0
        i = corner[0]
        while 8 > i > 0:
            if chessboard[i][corner[1]] == color:
                j = corner[1]
                while 8 > j > 0:
                    if chessboard[i][j] == color:
                        stable_discs += 1
                        #print(i,j)
                    else:
                        break
                    j += direcY
            else:
                break
            i += direcX

        return stable_discs


if __name__ == '__main__':
    a = AI(8, 1, 5)
    a.go(np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [1, -1, -1, -1, -1, -1, -1, 0], [-1, 1, 1, 1, 1, 0, 0, 0],
         [-1, -1, 1, 1, 1, 1, 1, 0], [-1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]))
    #a.stable_cnt([[0, 0, 0, 0, 0, 0, 0, 0],
                 # [0, 0, 0, 0, 0, 0, 0, 0],
                  #[0, 0, 0, 0, 0, 0, 0, 0],
                 # [0, 0, 0, 0, 0, 0, 0, 0],
                 # [0, 0, 0, 0, 0, 0, 0, 0],
                 # [0, 0, 0, 0, 0, 0, 1, 0],
                 # [0, 0, 0, 0, 0, 1, 1, 1],
                 # [1, 1, 1, 1, 1, 1, 1, 1]], 1)
    # print(a.evaluate_func())
