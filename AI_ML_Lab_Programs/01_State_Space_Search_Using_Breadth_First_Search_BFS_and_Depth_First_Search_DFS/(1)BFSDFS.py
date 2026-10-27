# Graph represented as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': [],
    'G': []
}

# Breadth-First Search (BFS)
def bfs(graph, start):
    visited = []
    queue = [start]

    while queue:
        node = queue.pop(0)  # Remove first element

        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])

    return visited


# Depth-First Search (DFS)
def dfs(graph, start):
    visited = []
    stack = [start]

    while stack:
        node = stack.pop()  # Remove last element

        if node not in visited:
            visited.append(node)

            # Reverse to maintain left-to-right traversal
            stack.extend(reversed(graph[node]))

    return visited


# Main Program
start_node = 'A'

print("Breadth-First Search (BFS):")
print(bfs(graph, start_node))

print("\nDepth-First Search (DFS):")
print(dfs(graph, start_node))



