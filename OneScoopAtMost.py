"""
At most one scoop at minimum cost
"""

from dwave.system import DWaveSampler, EmbeddingComposite

# Define QUBO from the slide
# f = 2*x1 + 5*x2
# Q matrix: [[2, 0], [0, 5]]

Q = {
    ('x1', 'x1'): 2,  # Vanilla: $2
    ('x2', 'x2'): 5,  # Chocolate: $5
}

print("Ice Cream Problem - D-Wave QPU")
print("="*50)
print("Vanilla (x1): $2")
print("Chocolate (x2): $5")
print("Objective: Minimize cost")
print()


print("Connecting to QPU:")
sampler = EmbeddingComposite(DWaveSampler())

print(f"Connected: {sampler.child.solver.name}")
print("Submitting problem...")
print()

response = sampler.sample_qubo(Q, num_reads=100)

print("Results:")
print("-"*50)

for sample, energy, num in response.data(['sample', 'energy', 'num_occurrences']):
    x1 = sample['x1']
    x2 = sample['x2']
    print(f"x1={x1}, x2={x2} -> Cost: ${energy} (appeared {num} times)")

print()
print("Best solution:")
best = response.first
print(f"x1={best.sample['x1']}, x2={best.sample['x2']}")
print(f"Minimum cost: ${best.energy}")