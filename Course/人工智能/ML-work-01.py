'''
功能：
- 随机建立一个有5~50个顶点的联通图
- 每个顶点的连接数为2~6个
- 顶点之间的距离与路径权值成比例
- 求出两点之间的最短路径
覃子轩
2019-05-19
'''
from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import random, math
import cv2
import numpy as np
from collections import defaultdict
from heapq import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def dijkstra_raw(self, edges, from_node, to_node):
        g = defaultdict(list)
        for l,r,c in edges:
            g[l].append((c,r))
        q, seen = [(0,from_node,())], set()
        while q:
            (cost,v1,path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    return cost,path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost+c, v2, path))
        return float("inf"),[]
 
    def dijkstra(self, edges, from_node, to_node):
        len_shortest_path = -1
        ret_path=[]
        length,path_queue = self.dijkstra_raw(edges, from_node, to_node)
        if len(path_queue)>0:
            len_shortest_path = length      
            left = path_queue[0]
            ret_path.append(left)      
            right = path_queue[1]
            while len(right)>0:
                left = right[0]
                ret_path.append(left)   
                right = right[1]
            ret_path.reverse()  
        return len_shortest_path,ret_path


    def drawGraph(self, start_index, end_index):
        Max = 99999
        range_int = [50, 50]
        cat_line = [2, 6]
        node_nums = random.randint(range_int[0], range_int[1])
        node_point = list(np.arange(0, node_nums))
        # 构建邻接全零数组
        node_G = np.zeros((node_nums, node_nums), np.int32)
        node_G += Max
        # 构建连通图
        node_flag = 1
        cat_flag = 0
        for G_line in node_G[:-1]:
            top_value = node_nums-node_flag
            if top_value > 6:
                top_value = 6
            cat_num = random.randint(2, top_value) if top_value >= 2 else 1
            while 1:
                index = random.randint(node_flag, node_nums-1)
                if G_line[index] == Max:
                    one_flag, two_flag = 0, 0
                    for one_item in G_line[:]:
                        if one_item == 1:
                            one_flag += 1
                    for two_item in node_G[index][:index]:
                        if two_item == 1:
                            two_flag +=1
                    if two_flag < 6 and one_flag < 6:
                        G_line[index] = 1
                        node_G[index][node_flag-1] = 1
                    cat_flag += 1
                if cat_flag == cat_num:
                    cat_flag = 0
                    break
            node_flag += 1


        bg_img = np.zeros((600,900,3),np.uint8)
        bg_img = 255-bg_img
        # 将点随机画到bg_img中
        r = 10
        min_dis = 30
        node_local = []
        for point_index in range(node_nums):
            # 确保不重合
            while 1:
                is_have = 0
                random_x = random.randint(r, bg_img.shape[1]-r)
                random_y = random.randint(r,bg_img.shape[0]-r)
                for node_local_item in node_local:
                    if abs(node_local_item[0]-random_x)<min_dis and abs(node_local_item[1]-random_y)<min_dis :
                        is_have = 1
                if is_have:
                    continue
                node_local.append((random_x,random_y))
                break

            cv2.circle(bg_img, (random_x,random_y), r, (0,0,255), -1)

        # 将点按照邻接矩阵连接
        for index, node_G_line in enumerate(node_G):
            for line_index, line_item in enumerate(node_G_line[:]):
                if line_item != Max:
                    start_point, end_point = (node_local[index][0],node_local[index][1]), (node_local[line_index][0],node_local[line_index][1])
                    cv2.line(bg_img,start_point, end_point, (128,0,255),1)
                    # 将距离存入邻接矩阵
                    dis = int((abs(start_point[0]-end_point[0])**2 + abs(start_point[1]-end_point[1])**2)**0.5)
                    node_G_line[line_index] = dis
        print(node_G)
        # 避免文字被覆盖
        for point_index, node_local_item in enumerate(node_local):
            # 个位数-5，十位数-10
            x_dis = r if point_index-10>=0 else r//2
            cv2.putText(bg_img, str(point_index), (node_local_item[0]-x_dis,node_local_item[1]+r//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # 给定起始点和终点找到最短路径并标蓝
        edges = []  
        for i in range(len(node_G)):
            for j in range(len(node_G[0])):
                if i!=j and node_G[i][j]!=Max:
                    edges.append((i,j,node_G[i][j]))
        print("=== Dijkstra ===")
        start_index = start_index if start_index < node_nums else 0
        end_index = end_index if end_index < node_nums else node_nums-1
        length,shortest_path = self.dijkstra(edges, start_index, end_index)
        print('length = ',length)
        print('The shortest path is ',shortest_path)
        for item in shortest_path[1:]:
            cv2.line(bg_img, (node_local[start_index][0],node_local[start_index][1]), 
                             (node_local[item][0],node_local[item][1]), (255,0,0),1)
            start_index = item


        cv2.imshow('bg',bg_img)
        cv2.waitKey()


    def createWidgets(self):
        self.start_input = Entry(self)
        self.start_input.pack()
        self.end_intput = Entry(self)
        self.end_intput.pack()

        self.alertButton = Button(self, text='确定', command=self.hello)
        self.alertButton.pack()

    def hello(self):
        start_index = int(self.start_input.get())
        end_index = int(self.end_intput.get())
        self.drawGraph(start_index,end_index)



app = Application()
app.master.title('输入窗口')
app.mainloop()