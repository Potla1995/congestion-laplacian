# This file defines the functions for running the experiments
import numpy
import qiskit.circuit.random
import scipy
import spectralutils
import circuitutils
import contractiontree
import networkx
from joblib import Parallel, delayed
import multiprocessing
import csv
import cotengra


# UTILITY FUNCTION FOR MULTIGRAPH AND GRAPH CONVERSION
def __graph_to_multigraph(graph: networkx.Graph) -> networkx.MultiGraph:
    """
    Converts a networkx graph into a networkx multigraph (each edge has keys)
    :param graph: input graph
    :return: output multigraph with each edge getting its edge index as key
    """
    multigraph = networkx.MultiGraph()
    i = 0
    for edge in graph.edges:
        multigraph.add_edge(edge[0], edge[1], key=i)
        i += 1
    return multigraph


def __multigraph_to_graph(multigraph: networkx.MultiGraph) -> networkx.Graph:
    """
    Converts a networkx multigraph into a networkx graph (each key is ignored)
    :param multigraph: input multigraph
    :return: output graph with each key ignored
    """
    graph = networkx.Graph()
    for edge in multigraph.edges:
        graph.add_edge(edge[0], edge[1])
    return graph


# BASE EXPERIMENT FOR ANY INPUT GRAPH
def __single_experiment(graph: networkx.MultiGraph) -> list:
    """
    Calculates congestions via different methods for a single graph
    :param graph: the graph whose congestion is to be calculated
    :return: list containing congestions obtained by different techniques
    """
    max_deg = max([len(graph[i]) for i in list(graph)])
    lambda2 = networkx.algebraic_connectivity(graph, normalized=False)
    lambda2_norm = networkx.algebraic_connectivity(graph, normalized=True)
    laplacian = networkx.laplacian_matrix(graph).astype(float)
    largest_eig = scipy.sparse.linalg.eigs(laplacian, k=1, which='LM', return_eigenvectors=False)[0]
    normalized_laplacian = networkx.normalized_laplacian_matrix(graph).astype(float)
    largest_eig_normalized = scipy.sparse.linalg.eigs(normalized_laplacian, k=1, which='LM', return_eigenvectors=False)[0]
    epsilon = spectralutils.spectral_balance(graph, normalized=False)
    epsilon_normalized = spectralutils.spectral_balance(graph, normalized=True)

    # Calculate lower bounds
    lower_bound = 2 * lambda2 / 9 * graph.number_of_nodes()
    lower_bound_normalized = 4 * lambda2_norm / 9 * graph.number_of_edges()

    # Calculate upper bounds (theoretical)
    upper_bound_generic = 2 * largest_eig * graph.number_of_nodes() / 9
    upper_bound_generic_normalized = 4 * largest_eig_normalized * graph.number_of_edges() / 9
    upper_bound_main_thm = max(epsilon * numpy.sqrt((2 * max_deg - lambda2) * lambda2),
                       (1 - epsilon ** 2 + (1.0/graph.number_of_nodes())) * largest_eig / 4.0)
    upper_bound_main_thm *= graph.number_of_nodes()

    upper_bound_main_thm_normalized = max(epsilon_normalized * numpy.sqrt((2 - lambda2_norm) * lambda2_norm),
                       (1 - epsilon_normalized ** 2 + (0.5/graph.number_of_edges())) * largest_eig_normalized / 4.0)
    upper_bound_main_thm_normalized *= (2*graph.number_of_edges())

    # Calculate the actual rank during hierarchical spectral clustering
    hierarchical_btree = spectralutils.hierarchical_spectral_clustering_order(graph, is_root=True, normalized=True)
    contraction_tree = contractiontree.ContractionTree(graph, hierarchical_btree)
    largest_rank, _ = contraction_tree.largest_rank_tensor()

    # Calculate the rank during contraction by competitors
    cotengra_input = []
    cotengra_shapes = []
    for node in graph.nodes:
        node_arr = list(graph[node])
        cotengra_input.append(node_arr)
        cotengra_shapes.append([2 for _ in node_arr])

    greedy_rank = numpy.round(cotengra.array_contract_tree(cotengra_input,
                                                                optimize='greedy',
                                                                shapes=cotengra_shapes).contraction_width())
    cotengra_rank = numpy.round(cotengra.array_contract_tree(cotengra_input,
                                                                  optimize='auto',
                                                                  shapes=cotengra_shapes).contraction_width())
    hyper_opt = cotengra.HyperOptimizer(minimize="size")

    hyper_opt_rank = numpy.round(cotengra.array_contract_tree(cotengra_input,
                                                               optimize=hyper_opt,
                                                               shapes=cotengra_shapes).contraction_width())

    result = [lower_bound, lower_bound_normalized,
              numpy.real(upper_bound_generic), numpy.real(upper_bound_generic_normalized),
              numpy.real(upper_bound_main_thm), numpy.real(upper_bound_main_thm_normalized),
              greedy_rank, cotengra_rank, hyper_opt_rank,
              largest_rank]
    return result


# UTILITY FUNCTION FOR PARALLEL EXECUTION
def __run_parallel(test_graphs: list) -> list:
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores-1} cores")
    full_result = Parallel(n_jobs=num_cores - 1)(
        delayed(__single_experiment)(test_graphs[i]) for i in range(len(test_graphs))
    )
    return full_result


# UTILITY FUNCTION FOR WRITING TO CSV
def __write_to_csv(filename, index_list, index_labels: list, full_result: list) -> None:
    """
    Writes the entries of full_result to csv
    :param index_list: list of indices for full_result, for e.g. [(10, 3, 0), (10, 3, 1), ...]
    :param index_labels: labels of the indices, for e.g. ['Number of nodes', 'Degree', 'Seed']
    :param full_result: array to be written into csv
    :param filename: string, name of file
    """
    results_row_labels = (index_labels +
                          ['Lower Bound', 'Lower Bound (normalized)',
                          'Generic Upper Bound', 'Generic Upper Bound (normalized)',
                          'Main Thm Upper Bound', 'Main Thm Upper Bound (normalized)',
                          'Greedy Upper Bound', 'Cotengra Upper Bound', 'HyperOpt Upper Bound',
                          'Largest HSC Rank'])
    if len(index_labels) != len(index_list[0]):
        raise ValueError(f"{len(index_labels)} many labels provided for {len(index_list[0])} indices per row!")
    elif len(full_result[0]) != len(results_row_labels)-len(index_labels):
        raise ValueError(f"full_result array has {len(full_result[0])} entries per row, expected {len(results_row_labels)-len(index_labels)}!")

    with open(filename, 'w') as f:
        f_writer = csv.writer(f, delimiter=',')
        if f.tell() == 0:
            f_writer.writerow(results_row_labels)
        for i in range(len(full_result)):
            f_writer.writerow(list(index_list[i]) + full_result[i])

    print(f"DEBUG: Mean of results written to {filename}:\n{results_row_labels}\n{numpy.mean(full_result, axis=0)}")


######################################
# EXPERIMENT 1: RANDOM REGULAR GRAPH #
######################################

def run_random_regular_experiment(degree_list: list[int], num_nodes_list: list[int], num_runs: int,  write_to_file: bool=False) -> None:
    """
    Run the RRG experiment (in parallel)
    :param degree_list: list of regularities of the graphs
    :param num_nodes_list: list of number of nodes in the graphs
    :param num_runs: number of runs to output to csv
    :param write_to_file: whether output should be written to csv
    """
    # Generate the test graphs
    test_graphs = []
    index_list = []
    for d in degree_list:
        for n in num_nodes_list:
            for i in range(num_runs):
                if (n * d) % 2 == 0:
                    index_list.append((n, d, i))
                    rrg_multigraph = __graph_to_multigraph(networkx.random_regular_graph(d, n, seed=i))
                    test_graphs.append(rrg_multigraph)

    # Run the tests in parallel
    full_result = __run_parallel(test_graphs)

    # Output the runs to csv
    if write_to_file:
        outfile = f'output/RRG{degree_list}.csv'
        __write_to_csv(outfile, index_list, ["Number of Nodes", "Degree", "Seed"], full_result)



###########################################
# EXPERIMENT 2: ERDOS-RENYI RANDOM GRAPHS #
###########################################

def run_erdos_renyi_experiment(prob_list: list[float], num_nodes_list: list[int], num_runs: int, write_to_file: bool=False) -> None:
    """
    Run the Erdos-Renyi Experiment (in parallel)
    :param prob_list: List of probabilities of the ER graphs
    :param num_nodes_list: List of number of nodes in the graphs
    :param num_runs: number of runs to output to csv
    :param write_to_file: whether output should be written to csv
    """
    # Generate the test graphs
    test_graphs = []
    index_list = []
    for p in prob_list:
        for n in num_nodes_list:
            for i in range(100, 100+num_runs):
                index_list.append((n, p, i))
                if p < numpy.log2(n)/n:
                    test_graphs.append(__graph_to_multigraph(networkx.fast_gnp_random_graph(n=n, p=p, seed=20*i+1)))
                else:
                    test_graphs.append(__graph_to_multigraph(networkx.erdos_renyi_graph(n=n, p=p, seed=20*i+1)))

    # Run the tests in parallel
    full_result = __run_parallel(test_graphs)

    # Output the runs to csv
    if write_to_file:
        outfile = f'output/ERG{str(prob_list)}_{str(num_nodes_list)}.csv'
        __write_to_csv(outfile, index_list, ["Number of Nodes", "Probability", "Seed"], full_result)


#########################################
# EXPERIMENT 3: RANDOM QUANTUM CIRCUIT  #
#########################################

def run_random_qc_experiment(num_qubits_list: list[int], depths_list: list[int], num_operand_distribution: dict, seeds_list: list, write_to_file: bool=False) -> None:
    """
    Run the Random Quantum Circuit Experiment
    :param num_operand_distribution: Distribution of how many k-qubit gates, for e.g.
    num_operand_distribution={1: 0, 2: 0.5, 3: 0.5, 4: 0} means 2 and 3 qubit gates have frequency 1/2
    :param num_qubits_list: List of number of qubits
    :param depths_list: List of depths
    :param num_runs: Number of runs of the experiment
    :param write_to_file: Whether the write the output to csv
    """
    circuits = []
    index_list = []
    for q in num_qubits_list:
        for d in depths_list:
            for seed in seeds_list:
                rc = qiskit.circuit.random.random_circuit(q, d, num_operand_distribution=num_operand_distribution)
                circuits.append(rc)
                index_list.append((q, d, seed))
    test_graphs = [circuitutils.circuit_to_graph(rc) for rc in circuits]

    # Run the tests in parallel
    full_result = __run_parallel(test_graphs)

    # Output the runs to csv
    if write_to_file:
        outfile = f'output/RC{str(num_qubits_list)}-{str(depths_list)}-{num_operand_distribution.values()}.csv'
        __write_to_csv(outfile, index_list, ["Number of Qubits", "Depth", "Seed"], full_result)

