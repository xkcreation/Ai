import heapq

# Graph with edge costs
graph = {
    'A': [('B', 2), ('C', 4)],
    'B': [('D', 7), ('E', 3)],
    'C': [('F', 2)],
    'D': [('G', 1)],
    'E': [('G', 5)],
    'F': [('G', 3)],
    'G': []
}

# Heuristic values
heuristic = {
    'A': 9,
    'B': 8,
    'C': 5,
    'D': 1,
    'E': 5,
    'F': 3,
    'G': 0
}

# A* Search Function
def astar(start, goal):

    # Priority Queue
    queue = []

    # (f, g, node, path)
    heapq.heappush(queue, (heuristic[start], 0, start, [start]))
    visited = []

    while queue:
        f, g, current, path = heapq.heappop(queue)

        # Ignore already visited nodes
        if current in visited:
            continue

        print("Current Node :", current)
        print("g(n) =", g)
        print("h(n) =", heuristic[current])
        print("f(n) =", f)
        print()

        visited.append(current)

        # Goal reached
        if current == goal:
            return path, g

        # Visit neighbouring nodes
        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(
                    queue,
                    (
                        new_f,
                        new_g,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, None


# Main Program
start = 'A'
goal = 'G'

path, cost = astar(start, goal)

print("-------------------------")
print("Optimal Path :", " -> ".join(path))
print("Total Cost :", cost)
