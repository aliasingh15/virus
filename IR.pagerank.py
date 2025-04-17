#A. Consider a web graph with the following link structure:

#• Page A has links to pages B and C.
#• Page B has a link to page C.
#• Page C has links to pages A.
#Apply the PageRank algorithm and analyze the results.

#output:

import numpy as np

# Pages: A = 0, B = 1, C = 2
N = 3  # number of pages
d = 0.85  # damping factor

# Adjacency matrix (columns are from, rows are to)
link_matrix = np.array([
    [0, 0, 1],   # A receives from C
    [1/2, 0, 0], # B receives from A (A has 2 out-links)
    [1/2, 1, 0]  # C receives from A and B
])

# Initialize rank vector
rank = np.array([1/N, 1/N, 1/N])

# Power iteration
def pagerank(link_matrix, rank, d=0.85, tol=1e-6, max_iter=100):
    N = len(rank)
    for i in range(max_iter):
        new_rank = (1 - d) / N + d * link_matrix @ rank
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank
    return rank

# Run PageRank
final_rank = pagerank(link_matrix, rank)

# Display results
pages = ['A', 'B', 'C']
for i, r in enumerate(final_rank):
    print(f"Page {pages[i]}: {r:.4f}")


