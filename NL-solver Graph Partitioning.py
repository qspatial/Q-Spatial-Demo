from dwave.optimization import Model
from dwave.system import LeapHybridNLSampler
import networkx as nx

def create_spatial_graph(grid_size=5):

    G = nx.grid_2d_graph(grid_size, grid_size)
    G = nx.convert_node_labels_to_integers(G)
    return G


def solve_graph_partitioning_nl(graph, num_partitions=3, min_partition_size=None, time_limit=5):

    num_nodes = len(graph.nodes())

    if min_partition_size is None:
        min_partition_size = num_nodes // num_partitions

    edge_list = list(graph.edges())
    num_edges = len(edge_list)

    model = Model()

    # variable
    x = model.integer(num_nodes, lower_bound=0, upper_bound=num_partitions - 1)

    # Constants
    size = model.constant(min_partition_size)

    # size constraints, no one-hot constraint

    for l in range(num_partitions):

        l_const = model.constant(l)

        comparisons = [x[i] == l_const for i in range(num_nodes)]
        partition_count = comparisons[0]
        for comp in comparisons[1:]:
            partition_count = partition_count + comp
        model.add_constraint(partition_count >= size)

    # objective function

    #a constant for 1, NL-solver feature
    one = model.constant(1)

    # indicators
    edge_cuts = [one - (x[i] == x[j]) for i, j in edge_list]


    edge_cut_sum = edge_cuts[0]
    for cut in edge_cuts[1:]:
        edge_cut_sum = edge_cut_sum + cut

    model.minimize(edge_cut_sum)

    print("Submitting NL model to D-Wave solver...")
    print(f"  Time limit: {time_limit} seconds")


    sampler = LeapHybridNLSampler()
    future = sampler.sample(model,
                            label='Spatial Graph Partitioning - NL',
                            time_limit=time_limit)


    result = future.result()

    print(f"\nResults:")

    with model.lock():

        objective_value = int(model.objective.state(0))
        print(f"  Objective value (edge cuts): {objective_value}")

        x_values = x.state(0)
        node_assignment = {i: int(x_values[i]) for i in range(num_nodes)}

    feasible = verify_solution(node_assignment, num_nodes, num_partitions, min_partition_size)
    print(f"  Feasible: {feasible}")

    return node_assignment, result


def verify_solution(node_assignment, num_nodes, num_partitions, min_partition_size):

    if len(node_assignment) != num_nodes:
        print(f"  ERROR: Only {len(node_assignment)}/{num_nodes} nodes assigned!")
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


def visualize_partition(graph, node_assignment):

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


if __name__ == "__main__":

    print("NL-Solver - Spatial Graph Partitioning")


    # Create spatial graph
    print("\nCreating spatial graph...")
    G = create_spatial_graph(grid_size=4)  # 4x4 = 16 nodes
    print(f"Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges\n")

    # Solve using NL-Solver
    num_partitions = 4
    solution, result = solve_graph_partitioning_nl(G,
                                                   num_partitions=num_partitions,
                                                   time_limit=15)

    # display results
    visualize_partition(G, solution)
