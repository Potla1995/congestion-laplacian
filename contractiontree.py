# This file contains class and function definitions related to the contraction tree of a tensor network
import networkx
from matplotlib import pyplot as plt


class ContractionTree:
    def __init__(self, base_graph: networkx.Graph, nested_tuple_repr: tuple):
        """
        A class with useful functions related to the binary contraction tree of graphs
        :param base_graph: the base graph whose nodes are the leaves of the contraction tree
        :param nested_tuple_repr: a nested tuple representation of the tree. For example, ((1,2),(3,4)) represents the
        complete binary tree with 4 leaves. It is required that the leaves are nodes of base_graph.
        """
        if base_graph.number_of_nodes() < 2:
            raise RuntimeError("Provided base graph must have at least 2 nodes!")
        for v in list(base_graph):
            if str(v).startswith("B"):
                raise ValueError("Provided base graph has a vertex starting with the letter \'B\'. Please rename all"
                                 "nodes, since the letter \'B\' is reserved for the nodes of the binary tree nodes.")
            if isinstance(v, tuple):
                raise ValueError("Provided base graph has a vertex which is a tuple. Please avoid this as it will"
                                 "conflict with the nested tuple representation used in this class.")
        self.base_graph = base_graph
        # Class variables
        self.binary_tree = networkx.Graph()  # the contraction tree, rooted at the node "B1"
        self.num_contractions = 1  # total number of non-leaf nodes in the tree
        self.dict_descendant_leaves = dict({})  # for every B-node, will contain list of nodes under it
        # Generate self.binary_tree
        self.binary_tree_root = self.__unwrap_tuple(nested_tuple_repr)
        # Calculate the dictionary of descendants
        self.__calculate_dict_descendant_leaves()

    # CLASS UTILITY FUNCTIONS
    def __unwrap_tuple(self, nested_tuple: tuple) -> str:
        """
        Class utility function to unwrap the nested tuples in input
        :return: the root node
        """
        vertex_set = list(self.base_graph)
        if len(nested_tuple) != 2:
            raise RuntimeError("Provided nested tuple is not a binary tree!")
        # Add a node corresponding to the current nested tuple
        current_node = "B{}".format(self.num_contractions)
        self.binary_tree.add_node(current_node)
        self.num_contractions += 1
        for i in range(2):
            if nested_tuple[i] in vertex_set:
                self.binary_tree.add_node(nested_tuple[i])
                self.binary_tree.add_edge(nested_tuple[i], current_node)
            else:
                # recurse on the subtree
                subtree_root = self.__unwrap_tuple(nested_tuple[i])
                self.binary_tree.add_edge(subtree_root, current_node)
        return current_node

    def __calculate_dict_descendant_leaves(self) -> None:
        """
        Class utility function to populate self.dict_descendant_leaves with a dictionary, where each node Bi of the
        contraction tree will have a list of descendant leaves calculated.
        """
        current_index = self.num_contractions - 1
        while current_index > 0:
            current_node = "B{}".format(current_index)
            neighbors = [i for i in self.binary_tree[current_node]]
            leaves = [i for i in neighbors if not str(i).startswith("B")]
            b_nodes = [i for i in neighbors if i not in leaves]
            # sort b_nodes so that Bi appears before Bj if i<j
            b_indices = sorted([int(x[1:]) for x in b_nodes])  # get the indices as integers, and sort them
            b_nodes = ["B{}".format(idx) for idx in b_indices]
            # Three possible cases for the two children of this node
            if len(leaves) == 2:
                self.dict_descendant_leaves[current_node] = leaves
            elif len(leaves) == 1:
                # find the child node depending on whether we are at the root or not
                if current_index > 1:
                    child = b_nodes[1]
                else:
                    child = b_nodes[0]
                self.dict_descendant_leaves[current_node] = list(self.dict_descendant_leaves[child])
                self.dict_descendant_leaves[current_node] += leaves
            else:
                # find the children nodes: if we're at root b_nodes will have only the 2 children in this case
                # if we're not at the root b_nodes will also have the parent which we need to remove.
                children = b_nodes
                if len(b_nodes) == 3:
                    children.remove(b_nodes[0])
                self.dict_descendant_leaves[current_node] = (list(self.dict_descendant_leaves[children[0]]) +
                                                             list(self.dict_descendant_leaves[children[1]]))
            current_index -= 1
        # print(self.dict_descendant_leaves)

    # TREE COMPLEXITY CALCULATIONS
    def largest_rank_tensor(self) -> tuple:
        """
        Calculate the largest rank tensor encountered when contracting the base graph using this contraction tree
        :return: maximum rank of any tensor encountered during contraction, and where it occurred in the tree
        """
        max_node = None
        max_rank = 0
        for node in list(self.binary_tree):
            if node in list(self.base_graph):
                deg = len(self.base_graph[node])
                if max_rank < deg:
                    max_rank = deg
                    max_node = node
            else:
                nodeset = self.dict_descendant_leaves[node]
                boundary = networkx.cut_size(self.base_graph, nodeset)
                if max_rank < boundary:
                    max_rank = boundary
                    max_node = node
        # self.draw_binary_tree()
        # print("Max rank of {} found at node {}".format(max_rank, max_node))
        return max_rank, max_node

    def total_contraction_steps(self) -> float:

        pass

    # OUTPUT HELPER FUNCTIONS
    def draw_binary_tree(self, output_filename: str = None) -> None:
        """
        Draw, or output as svg the binary tree represented by this object
        :param output_filename: filename to be written to under output/ directory. Default is None
        """
        color = []
        for node in list(self.binary_tree):
            if str(node).startswith("B"):
                color.append("#987D9A")
            else:
                color.append("#EECEB9")
        pos = networkx.nx_agraph.graphviz_layout(self.binary_tree, prog="dot", root="B1")
        networkx.draw(self.binary_tree,
                      pos=pos,
                      with_labels=True,
                      node_color=color)
        if output_filename is None:
            plt.show()
        else:
            plt.savefig('output/'+output_filename)
