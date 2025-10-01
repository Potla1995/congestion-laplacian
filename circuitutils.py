import qiskit
from qiskit.circuit import Gate, Reset
from qiskit.circuit.barrier import Barrier
import networkx
import numpy


def qiskit_to_gates(qc: qiskit.QuantumCircuit):
    """
    Return all gates in a qiskit circuit in the form of a list of pairs (matrix, qubits)
    :param qc: A `qiskit.QuantumCircuit` object
    :return: A list of (matrix, qubits) pairs, where each entry extracts a gate's matrix representation and the qubits
    it acts on.
    """
    gates = []
    qubits = set()
    for instruction_idx, instruction in enumerate(qc.data):
        instruction_type, instruction_qubits = instruction[0], instruction[1]
        reformatted_qubits = []
        for qubit in instruction_qubits:
            reformatted_qubit = (qubit._register.name, qubit._index)
            reformatted_qubits.append(reformatted_qubit)
            qubits.add(reformatted_qubit)
        if isinstance(instruction_type, Gate):
            matrix = instruction_type.to_matrix()
        elif isinstance(instruction_type, Reset) or isinstance(instruction_type, Barrier):
            continue
        else:
            raise ValueError('Unknown instruction {} in Circuit'.format(instruction_type))
        gates.append((matrix, reformatted_qubits))
    return qubits, gates

def circuit_to_graph(qc: qiskit.QuantumCircuit):
    """
    Convert a given qiskit circuit into a graph representing its tensor network.
    :param qc: qiskit.QuantumCircuit
    :return: networkx.Graph
    """
    qubits, gates = qiskit_to_gates(qc)
    qubits = list(qubits)
    # Build the Tensornetwork using the extracted gates
    network_graph = networkx.MultiGraph()  # Graph G
    qubits_current_gate = {}  # Dictionary keeping track of which gate was last added to each qubit
    for gate_id in range(len(gates)):
        (gate_matrix, gate_qubits) = gates[gate_id]
        # If swap operation is detected
        if len(gate_qubits)==2 and numpy.all(
                gate_matrix == [[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]):
            q0 = gate_qubits[0]
            q1 = gate_qubits[1]
            # If both qubits already have gates, switch them
            if q0 in qubits_current_gate.keys() and q1 in qubits_current_gate.keys():
                temp = qubits_current_gate[q0]
                qubits_current_gate[q0] = qubits_current_gate[q1]
                qubits_current_gate[q1] = temp
                continue
        if len(gate_qubits)>1:
            network_graph.add_node('gate{}'.format(gate_id))

            for q in range(len(gate_qubits)):
                if gate_qubits[q] in qubits_current_gate.keys():
                    previous_gate_id = qubits_current_gate[gate_qubits[q]]
                    network_graph.add_edge('gate{}'.format(previous_gate_id), 'gate{}'.format(gate_id))
                qubits_current_gate[gate_qubits[q]] = gate_id


    graph_contracted = networkx.MultiGraph()
    for u in network_graph.nodes():
        for v in network_graph[u]:
            if network_graph.number_of_edges(u, v) >= 1:  # Assume we are only using 2 qubit gates for now
                graph_contracted = networkx.contracted_nodes(network_graph, u, v, self_loops=False, copy=True)
    if networkx.algebraic_connectivity(graph_contracted) == 0:  # if the graph is disconnected, raise a warning
        print('Input quantum circuit is disconnected')
    return graph_contracted

