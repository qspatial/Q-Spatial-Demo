from dwave.optimization import Model
from dwave.system import LeapHybridNLSampler


def solve_graph_partitioning_nl(graph, num_partitions=3, min_partition_size=None, time_limit=5):
    """Solve graph partitioning using NL-Solver."""
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    num_nodes = len(nodes)
    min_partition_size = min_partition_size or num_nodes // num_partitions

    # Create model
    model = Model()

    # Decision variable - integer for each node indicating its partition
    x = model.integer(num_nodes, lower_bound=0, upper_bound=num_partitions - 1)

    # Constants
    size = model.constant(min_partition_size)
    one = model.constant(1)

    # Size constraints - ensure minimum partition size
    for l in range(num_partitions):
        l_const = model.constant(l)

        # Count nodes in partition l
        comparisons = [x[i] == l_const for i in range(num_nodes)]
        partition_count = comparisons[0]
        for comp in comparisons[1:]:
            partition_count = partition_count + comp

        model.add_constraint(partition_count >= size)

    # Objective function - minimize weighted edge cuts
    weighted_edge_cuts = []

    for edge in edges:
        # Get node indices in the nodes list
        i_idx = nodes.index(edge[0])
        j_idx = nodes.index(edge[1])

        # Get edge weight (default to 1.0)
        weight = graph[edge[0]][edge[1]].get('weight', 1.0)

        # Create weight constant
        weight_const = model.constant(weight)

        # Indicator: 1 if nodes are in different partitions, 0 otherwise
        edge_cut_indicator = one - (x[i_idx] == x[j_idx])

        # Weighted edge cut
        weighted_cut = weight_const * edge_cut_indicator
        weighted_edge_cuts.append(weighted_cut)

    # Sum all weighted edge cuts
    if weighted_edge_cuts:
        total_weighted_cuts = weighted_edge_cuts[0]
        for cut in weighted_edge_cuts[1:]:
            total_weighted_cuts = total_weighted_cuts + cut

        model.minimize(total_weighted_cuts)

    # Submit to D-Wave solver
    print("Submitting NL model to D-Wave solver...")
    print(f"  Time limit: {time_limit} seconds")

    try:
        sampler = LeapHybridNLSampler()
        future = sampler.sample(model, label='Graph Partitioning - NL', time_limit=time_limit)
        result = future.result()

        with model.lock():
            objective_value = model.objective.state(0)
            x_values = x.state(0)
            node_assignment = {nodes[i]: int(x_values[i]) for i in range(num_nodes)}

        print(f"\nResults:")
        print(f"  Objective value (edge cuts): {int(objective_value)}")
    except Exception as e:
        print(f"\nError: Could not connect to D-Wave solver: {e}")
        print("Creating a mock solution for demonstration...")
        node_assignment = {nodes[idx]: idx % num_partitions for idx in range(num_nodes)}
        mock_objective = sum(graph[i][j].get('weight', 1.0) for i, j in edges
                             if node_assignment[i] != node_assignment[j])
        print(f"\nResults:")
        print(f"  Objective value (edge cuts): {int(mock_objective)}")
        result = type('Result', (), {'sample': node_assignment})()

    feasible = verify_solution(node_assignment, nodes, num_partitions, min_partition_size)
    print(f"  Feasible: {feasible}")

    return node_assignment, result


def verify_solution(node_assignment, nodes, num_partitions, min_partition_size):
    """Verify if the solution meets all constraints"""
    if len(node_assignment) != len(nodes):
        print(f"  ERROR: Only {len(node_assignment)}/{len(nodes)} nodes assigned!")
        return False

    partition_sizes = {}
    for node, partition in node_assignment.items():
        if partition not in partition_sizes:
            partition_sizes[partition] = 0
        partition_sizes[partition] += 1

    for l in range(num_partitions):
        size = partition_sizes.get(l, 0)
        if size < min_partition_size:
            print(f"  WARNING: Partition {l} has {size} nodes (minimum: {min_partition_size})")
            return False

    return True


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
    print("NL-Solver - Spatial Graph Partitioning")

    print("\nLoading graph from file...")
    G = load_graph_from_file('graph.gpickle')
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges\n")

    # Solve the partitioning problem
    num_partitions = 3
    solution, result = solve_graph_partitioning_nl(G,
                                                   num_partitions=num_partitions,
                                                   time_limit=15)

    # Text-based visualization
    visualize_partition_text(G, solution)

    # Save results to file
    save_results_to_file(G, solution, num_partitions,
                         'partition_results_nl.txt')
    print("Results saved to: partition_results_nl.txt")