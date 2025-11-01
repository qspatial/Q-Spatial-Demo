"""
 Buy exact one scoop

"""

from dwave.system import DWaveSampler, EmbeddingComposite

# QUBO with constraint: x1 + x2 = 1 (buy exactly one)
# Set P=10:
# Q matrix diagonal: (-8, -5)
# Q matrix off-diagonal: 8+12 = 20

Q = {
    ('x1', 'x1'): -8,
    ('x2', 'x2'): -5,
    ('x1', 'x2'): 20,  # Penalty term: 2*P where P=10
}

print("Constrained Ice Cream Problem - D-Wave QPU")
print("=" * 60)
print("Constraint: Buy EXACTLY one scoop")
print("Vanilla (x1): $2")
print("Chocolate (x2): $5")
print("Penalty P = 10")
print()

print("Connecting to D-Wave QPU...")
sampler = EmbeddingComposite(DWaveSampler())

print(f"Connected: {sampler.child.solver.name}")
print("Submitting problem...")
print()


response = sampler.sample_qubo(Q, num_reads=100)

print("Results:")
print("-" * 60)

cost_x1 = 2
cost_x2 = 5

for sample, energy, num in response.data(['sample', 'energy', 'num_occurrences']):
    x1 = sample['x1']
    x2 = sample['x2']

    satisfies = (x1 + x2 == 1)
    status = "✓" if satisfies else "✗"

    actual_cost = cost_x1 * x1 + cost_x2 * x2

    print(f"{status} x1={x1}, x2={x2} | QUBO energy={energy:6.1f} | "
          f"Actual cost=${actual_cost} | Count={num}")

print()
print("=" * 60)
print("Best solution:")
best = response.first
x1_best = best.sample['x1']
x2_best = best.sample['x2']

print(f"x1={x1_best}, x2={x2_best}")
print(f"QUBO energy: {best.energy}")

if x1_best + x2_best == 1:
    actual_cost = cost_x1 * x1_best + cost_x2 * x2_best
    print(f"Actual cost: ${actual_cost}")

    if x1_best == 1:
        print("→ Buy Vanilla scoop")
    else:
        print("→ Buy Chocolate scoop")
else:
    print("Warning: Constraint not satisfied!")