from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler
import networkx as nx


def create_spatial_graph(grid_size=5):

    G = nx.grid_2d_graph(grid_size, grid_size)

    G = nx.convert_node_labels_to_integers(G)

    return G


def solve_graph_partitioning_cqm(graph, num_partitions=3, min_partition_size=None, time_limit=5):

    num_nodes = len(graph.nodes())

    if min_partition_size is None:
        min_partition_size = num_nodes // num_partitions

    has_remainder = (num_nodes % num_partitions) != 0

    cqm = ConstrainedQuadraticModel()

    x = {(i, l): Binary(f'x_{i}_{l}')
         for i in graph.nodes()
         for l in range(num_partitions)}


    objective = 0

    for i, j in graph.edges():

        W_ij = graph[i][j].get('weight', 1.0)

        edge_in_same_partition = sum(x[(i, l)] * x[(j, l)] for l in range(num_partitions))
        edge_is_cut = 1 - edge_in_same_partition
        objective += W_ij * edge_is_cut

    cqm.set_objective(objective)

    for i in graph.nodes():
        cqm.add_discrete([f'x_{i}_{l}' for l in range(num_partitions)],
                        label=f'one_hot_node_{i}')

    if has_remainder:
        # Use inequality when nodes don't divide evenly
        for l in range(num_partitions):
            partition_total = sum(x[(i, l)] for i in graph.nodes())
            cqm.add_constraint(partition_total >= min_partition_size,
                              label=f'size_partition_{l}')
    else:
        # Use equality when nodes divide evenly (forces exact partition sizes)
        for l in range(num_partitions):
            partition_total = sum(x[(i, l)] for i in graph.nodes())
            cqm.add_constraint(partition_total == min_partition_size,
                              label=f'size_partition_{l}')


    print("Submitting CQM to D-Wave solver...")
    print(f"  Time limit: {time_limit} seconds")

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm,
                                   label='Spatial Graph Partitioning - CQM',
                                   time_limit=time_limit)

    best_solution = sampleset.first

    print(f"\nResults:")
    print(f"  Objective value (edge cuts): {int(best_solution.energy)}")
    print(f"  Feasible: {best_solution.is_feasible}")

    # Convert solution to node-partition mapping
    node_assignment = {}
    for i in graph.nodes():
        assigned = False
        for l in range(num_partitions):
            if best_solution.sample[f'x_{i}_{l}'] == 1:
                node_assignment[i] = l
                assigned = True
                break

        # Debug: check if node wasn't assigned
        if not assigned:
            print(f"  WARNING: Node {i} not assigned to any partition!")

    # Verify all nodes are assigned
    if len(node_assignment) != num_nodes:
        print(f"  ERROR: Only {len(node_assignment)}/{num_nodes} nodes assigned!")

    return node_assignment, best_solution


def visualize_partition(graph, node_assignment):
    """Print partition assignments"""
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
    # Create spatial graph
    print("CQM Solver - Spatial Graph Partitioning")



    print("\nCreating spatial graph...")
    G = create_spatial_graph(grid_size=4)  # 4x4 = 16 nodes
    print(f"Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges\n")


    num_partitions = 4
    solution, result = solve_graph_partitioning_cqm(G,
                                                     num_partitions=num_partitions,
                                                     time_limit=15)  # Give solver more time


    visualize_partition(G, solution)
