"""
CSE 331 SS22 (Onsay)
Graph Project Part 1
Name: Chirag Rudrangi
"""

import math
import queue
import time
import csv
from typing import TypeVar, Tuple, List, Set, Dict

import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """
    Class representing a Vertex object within a Graph.
    """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, id_init: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex.
        :param id_init: [str] A unique string identifier used for hashing the vertex.
        :param x: [float] The x coordinate of this vertex (used in a_star).
        :param y: [float] The y coordinate of this vertex (used in a_star).
        :return: None.
        """
        self.id = id_init
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY.
        Equality operator for Graph Vertex class.
        :param other: [Vertex] vertex to compare.
        :return: [bool] True if vertices are equal, else False.
        """
        if self.id != other.id:
            return False
        if self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        if self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        if self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        if set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Constructs string representation of Vertex object.
        :return: [str] string representation of Vertex object.
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]
        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    __str__ = __repr__

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set. Used in unit tests.
        :return: [int] Hash value of Vertex.
        """
        return hash(self.id)

#============== Modify Vertex Methods Below ==============#

    def deg(self) -> int:
        """
        Returns the number of outgoing edges from this vertex;
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return len(self.adj) # number of outgoing edges

    def get_outgoing_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples representing outgoing edges from this vertex
        Edges are represented as tuples (other_id, weight) where
            - other_id is the unique string id of the destination vertex
            - weight is the weight of the edge connecting this vertex to the other vertex
        Returns an empty set if this vertex has no outgoing edges
        Time Complexity: O(degV)
        Space Complexity: O(degV)
        """
        return set(self.adj.items()) #set of outgoing edges

    def euclidean_dist(self, other: Vertex) -> float:
        """
        Returns the euclidean distance [based on two-dimensional coordinates]
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2) #euclidean distance

    def taxicab_dist(self, other: Vertex) -> float:
        """
        Returns the taxicab distance
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return abs(self.x - other.x) + abs(self.y - other.y) #taxicab distance


class Graph:
    """
    Class implementing the Graph ADT using an Adjacency Map structure.
    """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csvf: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance.
        :param plt_show: [bool] If true, render plot when plot() is called; else, ignore plot().
        :param matrix: [Matrix] Optional matrix parameter used for fast construction.
        :param csvf: [str] Optional filepath to a csv containing a matrix.
        :return: None.
        """
        matrix = matrix if matrix else np.loadtxt(csvf, delimiter=',', dtype=str).tolist()\
            if csvf else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class.
        :param other: [Graph] Another graph to compare.
        :return: [bool] True if graphs are equal, else False.
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        for vertex_id, vertex in self.vertices.items():
            other_vertex = other.get_vertex_by_id(vertex_id)
            if other_vertex is None:
                print(f"Vertices not equal: '{vertex_id}' not in other graph")
                return False

            adj_set = set(vertex.adj.items())
            other_adj_set = set(other_vertex.adj.items())

            if not adj_set == other_adj_set:
                print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                print(f"Adjacency symmetric difference = "
                      f"{str(adj_set.symmetric_difference(other_adj_set))}")
                return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Constructs string representation of graph.
        :return: [str] String representation of graph.
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    __str__ = __repr__

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib.
        :return: None.
        """
        if self.plot_show:
            import matplotlib.cm as cm
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_all_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_all_edges())
            max_weight = max([edge[2] for edge in self.get_all_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_all_edges()):
                origin = self.get_vertex_by_id(edge[0])
                destination = self.get_vertex_by_id(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_all_vertices()])
            y = np.array([vertex.y for vertex in self.get_all_vertices()])
            labels = np.array([vertex.id for vertex in self.get_all_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_all_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, begin_id: str, end_id: str = None, weight: float = 1) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param begin_id: [str] unique string id of starting vertex
        :param end_id: [str] unique string id of ending vertex
        :param weight: [float] weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(begin_id) is None:
            self.vertices[begin_id] = Vertex(begin_id)
            self.size += 1
        if end_id is not None:
            if self.vertices.get(end_id) is None:
                self.vertices[end_id] = Vertex(end_id)
                self.size += 1
            self.vertices.get(begin_id).adj[end_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: [Matrix] an n x n square matrix (list of lists) representing Graph
        :return: None
        """
        for i in range(1, len(matrix)):         # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):         # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        Given a graph, creates an adjacency matrix of the type described in construct_from_matrix.
        :return: [Matrix] representing graph.
        """
        matrix = [[None] + list(self.vertices)]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        Given a (non-empty) graph, creates a csv file containing data necessary to reconstruct.
        :param filepath: [str] location to save CSV.
        :return: None.
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

#============== Modify Graph Methods Below ==============#

    def unvisit_vertices(self) -> None:
        """
        Resets visited flags to False of all vertices within the graph
        Used in unit tests to reset graph between tests
        Time Complexity: O(V)
        Space Complexity: O(V)
        """

        for v_id, vertex in self.vertices.items(): #reset all vertices
            vertex.visited = False #reset all vertices

    def get_vertex_by_id(self, v_id: str) -> Vertex:
        """
        Returns the Vertex object with id v_id if it exists in the graph
        Returns None if no vertex with unique id v_id exists
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self.vertices.get(v_id) # return vertex with id v_id

    def get_all_vertices(self) -> Set[Vertex]:
        """
        Returns a set of all Vertex objects held in the graph
        Returns an empty set if no vertices are held in the graph
        Time Complexity: O(V)
        Space Complexity: O(V)
        """
        return set(self.vertices.values()) # return all vertices

    def get_edge_by_ids(self, begin_id: str, end_id: str) -> Tuple[str, str, float]:
        """
        Returns the edge connecting the vertex with id begin_id to the vertex
        If the edge or either of the associated vertices does not exist in the graph, returns None
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self.vertices.get(begin_id) is not None and \
                self.vertices.get(begin_id).adj.get(end_id) is not None:
            return begin_id, end_id, self.vertices.get(begin_id).adj.get(end_id) #return edge
        return None #return None if edge doesn't exist

    def get_all_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Returns a set of tuples representing all edges within the graph
        Edges are represented as tuples (begin_id, end_id, weight) where
        begin_id is the unique string id of the starting vertex
        end_id is the unique string id of the destination vertex
        weight is the weight of the edge connecting the starting vertex to the destination vertex
        Returns an empty set if the graph is empty
        Time Complexity: O(V+E)
        Space Complexity: O(E)
        """
        edges = set() #create empty set
        for v_id, vertex in self.vertices.items(): #go through all the vertices
            for adj_id, weight in vertex.adj.items(): #go through all adjacent vrtices
                edges.add((v_id, adj_id, weight)) #add edge to the set
        return edges #return the set of edges

    def build_path(self, back_edges: Dict[str, str],
                   begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Given a dictionary of back-edges (a mapping of vertex id to predecessor vertex id)
        Helper function must be called.
        Returns tuple of the form ([path], distance)
            - [path] is a list of vertex id strings beginning with begin_id, terminating with end_id
            - Distance is the sum of the weights of the edges along the [path] traveled
        Time Complexity: O(V)
        Space Complexity: O(V)
        """
        if begin_id not in self.vertices or end_id not in self.vertices: # if vertex N/A
            return None, None
        path = [end_id] # create empty list
        distance = 0
        while path[-1] != begin_id:
            path.append(back_edges[path[-1]]) # create empty list
            distance += self.vertices.get(path[-1]).adj.get(path[-2]) # add edge weight to distance
        return path[::-1], distance # return reversed path and distance

    def bfs(self, begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Perform a breadth-first search beginning at vertex with id begin_id and ending at end_id
        Call build_path
        Iterate over neighbors using vertex.adj (not vertex.get_edges())
        Returns tuple of the form ([path], distance) where
            path is a list of vertix strings beginning with begin_id and ending with end_id
            distance is the sum of the weights of the edges traveled.
        begin_id != end_id
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        if begin_id == end_id:
            return [begin_id], 0
        if begin_id not in self.vertices or end_id not in self.vertices:
            return [], 0
        parent = {begin_id: None}
        q = queue.SimpleQueue()
        q.put(begin_id)
        while not q.empty():
            v_id = q.get()
            for adj_id, weight in self.vertices.get(v_id).adj.items():
                if adj_id not in parent:
                    parent[adj_id] = v_id
                    q.put(adj_id)
        if end_id not in parent:
            return [], 0
        return self.build_path(parent, begin_id, end_id)



    def dfs(self, begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        This function makes it simpler for client code to call for a dfs
        construct the path in reverse order in dfs_inner
        reverse the path in function
        call dfs_inner with current_id and begin_id
        _Time Complexity: O(V+E) (including calls to dfs_inner)_
        _Space Complexity: O(V) (including calls to dfs_inner)_
        """
        def dfs_inner(current_id: str, end_id: str,
                      path: List[str] = []) -> Tuple[List[str], float]:
            """
            MUST BE RECURSIVE
            iterate over neighbors using vertex.adj (not vertex.get_edges())
            Returns tuple of the form ([path], distance) where
                [path] is a list of vertex id begin_id and ends with end_id
                distance is the sum of the weights of the edges along the [path] traveled
            if no path exists, return tuple ([], 0)
            Guaranteed that begin_id != end_id
            Time Complexity: O(V+E)
            Space Complexity: O(V)
            """
            if current_id != end_id:
                self.get_vertex_by_id(current_id).visited = True
                for i in self.get_vertex_by_id(current_id).adj:
                    if not self.get_vertex_by_id(i).visited:
                        path, distance = dfs_inner(i, end_id, path)
                        if path:
                            path.append(current_id)
                            return path, distance + self.get_vertex_by_id(current_id).adj[i]
                self.get_vertex_by_id(current_id).visited = False
                return [], 0
            else:
                path.append(current_id)
                return path, 0

        empty = []
        path = []
        if begin_id not in self.vertices or end_id not in self.vertices:
            return empty, 0
        final = dfs_inner(begin_id, end_id, path)
        if final is None:
            return empty, 0
        return final[0][::-1], final[1]

    def topological_sort(self) -> List[str]:

        """
        Performs topological sort on the graph, returning a ordered vertex ids.
        Recall that there can be multiple correct orderings following topological sort
        _Time Complexity: O(V+E) (including calls to topological_sort_inner)_
        _Space Complexity: O(V) (including calls to topological_sort_inner)_

        """
        final = []  # list of vertices that have no remaining constraints
        stack = []  # a list of vertices placed in topological order
        counter = {item: 0 for item in self.vertices} #dict of vertices and constraints
        for item in self.vertices:
            for adj in self.vertices[item].adj:
                counter[adj] += 1
        for item in self.vertices:
            if counter[item] == 0:
                final.append(item)
        while final:
            current = final.pop()
            stack.append(current)
            for adj in self.vertices[current].adj:
                counter[adj] -= 1
                if counter[adj] == 0:
                    final.append(adj)
        return stack



    def boss_order_validity(self) -> bool:
        """
        Returns True if the game be beaten otherwise return False
        Be careful with time and space complexity, they are easy to violate here.
        The testcases should give a good idea of when the game can be beaten or not.
        Guaranteed that the graph is connected.
        Consider an empty graph (no bosses) as already trivially beatable (return True).
        Hint: You may find it useful to use your topological_sort function to solve this problem
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        order = self.topological_sort()
        if len(self.vertices) == 0:
            return True
        if len(order) == 0 or self.get_edge_by_ids(order[-1], order[0]) is not None:
            return False
        if len(order) == len(self.vertices):
            return True