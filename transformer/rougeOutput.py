import matplotlib.pyplot as plt

# Data
beam_sizes = [1, 3, 5, 7, 10]
rouge1 = [0.2464, 0.2545, 0.2489, 0.2495, 0.2405]
rouge2 = [0.0459, 0.0497, 0.0524, 0.0542, 0.0486]
rougeL = [0.1680, 0.1730, 0.1692, 0.1704, 0.1627]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(beam_sizes, rouge1, marker='o', label='ROUGE-1')
plt.plot(beam_sizes, rouge2, marker='o', label='ROUGE-2')
plt.plot(beam_sizes, rougeL, marker='o', label='ROUGE-L')

plt.xlabel('Beam Size')
plt.ylabel('ROUGE Score')
plt.title('ROUGE Score vs Beam Size')
plt.legend()
plt.grid(True)

plt.show()
