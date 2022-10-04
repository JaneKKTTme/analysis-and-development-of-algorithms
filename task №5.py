import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp

VERTICES = 100

def generate_adj_matrix(edges):
    adj_matrix = np.zeros(VERTICES*VERTICES, dtype=int).reshape(VERTICES, VERTICES)
    connections = []
    while edges:
        i = random.randint(0, VERTICES-1)
        j = random.randint(0, VERTICES-1)
        while i == j:
            j = random.randint(0, VERTICES-1)
        adj_matrix[i][j], adj_matrix[j][i] = 1, 1
        connections.append([i+1, j+1])
        edges -= 1
    return connections, adj_matrix

def transfer_to_adj_list(adj_matrix):
    adj_list = {}
    for index, line in enumerate(adj_matrix):
        vertices = set()
        for node in range(len(line)):
            if line[node] == 1:
                vertices.add(node+1)
        adj_list[index+1] = vertices
    return adj_list

def visualize(graph):
    G = nx.Graph()
    print(graph)
    for edge in graph:
        G.add_edge(*edge)
    nx.draw_kamada_kawai(G, with_labels=True)
    plt.show()

def do_depth_first_search(graph, start):
    '''
    # recursive version
    if visited is None:
        visited = []
    visited.append(start)

    for next in graph[start]:
        do_depth_first_search(graph, next, visited)
    '''
    visited, stack = set(), [start]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
        stack.extend(graph[vertex] - visited)

    return visited

def do_breadth_first_search(graph, start):
    visited = [False for i in range(len(graph))]
    queue = []

    queue.append(start)
    visited[start] = True

    while queue:
        vertex = queue.pop(0)
        print(vertex, end=" ")
        for node in graph[vertex]:
            if visited[node-1] is False:
                queue.append(node)
                visited[node-1] = True

    return 0

def find_shortest_path(graph, start, end):
    visited = set()
    queue = [(start, [start])]

    while queue:
        vertex, path = queue.pop(0)
        visited.add(vertex)
        for node in graph[vertex]:
            if node == end:
                return path + [end]
            else:
                if node not in visited:
                    visited.add(node)
                    queue.append((node, path + [node]))

    return queue

if __name__ == '__main__':
    connections, graph = generate_adj_matrix(edges=200)
    print(graph[:5])
    graph = transfer_to_adj_list(graph)
    print(graph)

    visualize(connections)

    print(do_depth_first_search(graph, 1))

    print(do_breadth_first_search(graph, 1))

    start = random.randint(min(graph.keys()), max(graph.keys()))
    end = random.randint(min(graph.keys()), max(graph.keys()))
    print('Shortest way between', start, 'and', end, ':', find_shortest_path(graph, start, end))

