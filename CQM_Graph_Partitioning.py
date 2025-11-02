from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler

def solve_graph_partitioning_cqm(graph, num_partitions=3, min_partition_size=None, time_limit=5):

    nodes = list(graph.nodes())
    edges = list(graph.edges())
    partitions = list(range(num_partitions))

    # Set minimum partition size
    s = min_partition_size if min_partition_size else len(nodes) // num_partitions

    # Create binary variables
    x = [[Binary(f'x_{{{i},{l}}}') for l in partitions] for i in nodes]

    # Create CQM
    cqm = ConstrainedQuadraticModel()

    # Build objective function
    min_obj_funs = []
    for i, j in edges:
        i_idx = nodes.index(i)
        j_idx = nodes.index(j)
        for l in partitions:
            min_obj_funs.append(graph[i][j].get('weight', 1.0) *
                                (x[i_idx][l] + x[j_idx][l] - 2 * x[i_idx][l] * x[j_idx][l]))

    cqm.set_objective(quicksum(min_obj_funs))

    # Add one-hot constraints for each node
    for i_idx, i in enumerate(nodes):
        cqm.add_discrete([f'x_{{{i},{l}}}' for l in partitions],
                         label=f'one-hot-node-{i}')

    # Add partition size constraints
    for l in partitions:
        cqm.add_constraint(
            quicksum(x[i_idx][l] for i_idx in range(len(nodes))) >= s,
            label=f'partition-size-{l}'
        )

    # Submit to D-Wave solver
    print("Submitting CQM to D-Wave solver...")
    print(f"  Time limit: {time_limit} seconds")

    try:
        sampler = LeapHybridCQMSampler()
        sampleset = sampler.sample_cqm(cqm,
                                       label='Graph Partitioning - CQM',
                                       time_limit=time_limit)
        best_solution = sampleset.first
    except Exception as e:
        print(f"\nError: Could not connect to D-Wave solver: {e}")
        print("Creating a mock solution for demonstration...")

        # Create a simple greedy partitioning for demonstration
        from collections import namedtuple
        MockSolution = namedtuple('MockSolution', ['sample', 'energy', 'is_feasible'])

        # Simple round-robin assignment
        mock_sample = {}
        for idx, node in enumerate(nodes):
            partition = idx % num_partitions
            for l in partitions:
                mock_sample[f'x_{{{node},{l}}}'] = 1 if l == partition else 0

        # Calculate mock energy (edge cuts)
        mock_energy = 0
        for i, j in edges:
            i_partition = nodes.index(i) % num_partitions
            j_partition = nodes.index(j) % num_partitions
            if i_partition != j_partition:
                mock_energy += graph[i][j].get('weight', 1.0)

        best_solution = MockSolution(sample=mock_sample, energy=mock_energy, is_feasible=True)

    print(f"\nResults:")
    print(f"  Objective value (edge cuts): {int(best_solution.energy)}")
    print(f"  Feasible: {best_solution.is_feasible}")

    # Convert solution to node-partition mapping
    node_assignment = {}
    for i in nodes:
        for l in partitions:
            if best_solution.sample.get(f'x_{{{i},{l}}}', 0) == 1:
                node_assignment[i] = l
                break

    return node_assignment, best_solution


def visualize_partition_text(graph, node_assignment):
    """Print partition assignments (text-based visualization)"""
    print("\nPartition Assignments:")
    partitions = {}
    for node, partition in node_assignment.items():
        if partition not in partitions:
            partitions[partition] = []
        partitions[partition].append(node)

    for partition, nodes in sorted(partitions.items()):
        print(f"  Partition {partition}: {sorted(nodes)} ({len(nodes)} nodes)")

    edge_cuts = 0
    for i, j in graph.edges():
        if node_assignment[i] != node_assignment[j]:
            edge_cuts += 1

    print(f"\n  Total edge cuts: {edge_cuts}")
    print(f"  Total nodes assigned: {len(node_assignment)}/{len(graph.nodes())}")





def load_graph_from_file(filepath):
    """Load a graph from a pickle file"""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results_to_file(graph, node_assignment, num_partitions, filepath):
    """Save partitioning results to a text file"""
    with open(filepath, 'w') as f:
        f.write("Graph Partitioning Results\n")
        f.write("=" * 50 + "\n\n")

        # Partition summary
        partitions = {}
        for node, partition in node_assignment.items():
            if partition not in partitions:
                partitions[partition] = []
            partitions[partition].append(node)

        f.write("Partition Summary:\n")
        for p in range(num_partitions):
            nodes_in_partition = partitions.get(p, [])
            f.write(f"  Partition {p}: {sorted(nodes_in_partition)} ({len(nodes_in_partition)} nodes)\n")

        # Edge cut information
        f.write("\nCut Edges:\n")
        total_cut_weight = 0
        cut_count = 0
        for u, v in graph.edges():
            if node_assignment.get(u, -1) != node_assignment.get(v, -1):
                weight = graph[u][v].get('weight', 1.0)
                f.write(f"  {u} - {v} (weight: {weight})\n")
                total_cut_weight += weight
                cut_count += 1

        f.write(f"\nTotal edges cut: {cut_count}/{len(graph.edges())}\n")
        f.write(f"Total cut weight: {total_cut_weight:.2f}\n")


if __name__ == "__main__":
    print("CQM Solver - Spatial Graph Partitioning")


    print("\nLoading graph from file...")
    G = load_graph_from_file('graph.gpickle')
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges\n")

    # Solve the partitioning problem
    num_partitions = 3
    solution, result = solve_graph_partitioning_cqm(G,
                                                    num_partitions=num_partitions,
                                                    time_limit=15)

    # Text-based visualization
    visualize_partition_text(G, solution)



    # Save results to file
    save_results_to_file(G, solution, num_partitions,
                         'partition_results.txt')
    print("Results saved to: partition_results.txt")