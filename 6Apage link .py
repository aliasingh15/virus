import numpy as np

# Function to compute PageRank
def pagerank(graph, d=0.85, max_iter=100, tol=1.0e-6):
    # Initialize number of pages and the PageRank vector
    N = len(graph)
    PR = np.ones(N) / N  # Initially distribute PageRank equally among all pages

    # Convergence criterion: stop if the change in PageRank values is below a threshold
    for _ in range(max_iter):
        new_PR = np.zeros(N)

        # Apply the PageRank formula
        for page in range(N):
            inbound_pages = [i for i, links in enumerate(graph) if links[page] > 0]
            for inbound_page in inbound_pages:
                out_links = np.sum(graph[inbound_page])  # Out-degree of the inbound page
                new_PR[page] += d * PR[inbound_page] / out_links
            # Apply the teleportation factor (1 - d) and normalization
            new_PR[page] += (1 - d) / N

        # Check for convergence (if the PageRank values do not change significantly)
        if np.linalg.norm(new_PR - PR, 1) < tol:
            break

        PR = new_PR  # Update PageRank values for the next iteration

    return PR

# Define the web graph as an adjacency matrix
# Each element (i, j) in the matrix indicates whether page i links to page j
# If there's a link from i to j, the element will be 1, else 0.
graph = np.array([
    [0, 1, 1, 0],  # Page A links to B, C
    [0, 0, 1, 1],  # Page B links to C, D
    [1, 0, 0, 1],  # Page C links to A, D
    [0, 1, 0, 0]   # Page D links to B
])

# Compute the PageRank
pagerank_values = pagerank(graph)

# Map the PageRank values to the respective pages
pages = ['A', 'B', 'C', 'D']
for i, page in enumerate(pages):
    print(f"Page {page} has a PageRank of {pagerank_values[i]:.4f}")
