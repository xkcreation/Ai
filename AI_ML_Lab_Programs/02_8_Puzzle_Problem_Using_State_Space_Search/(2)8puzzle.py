from collections import deque

# Goal state
goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Convert 2D list into tuple
def to_tuple(state):
    return tuple(tuple(row) for row in state)

# Find blank space position
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# Generate next possible states
def get_neighbors(state):
    neighbors = []
    x, y = find_blank(state)

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in moves:
        new_x = x + dx
        new_y = y + dy

        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_state = [row[:] for row in state]

            new_state[x][y], new_state[new_x][new_y] = \
                new_state[new_x][new_y], new_state[x][y]

            neighbors.append(new_state)

    return neighbors

# BFS search
def solve_puzzle(start_state):
    queue = deque()
    visited = set()

    queue.append((start_state, []))
    visited.add(to_tuple(start_state))

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path + [current_state]

        for neighbor in get_neighbors(current_state):
            if to_tuple(neighbor) not in visited:
                visited.add(to_tuple(neighbor))
                queue.append((neighbor, path + [current_state]))

    return None

# Display puzzle state
def print_state(state):
    for row in state:
        print(row)
    print()

# Main program
start_state = [
    [1, 0, 3],
    [4, 2, 6],
    [7, 5, 8]
]

solution = solve_puzzle(start_state)

if solution:
    print("Solution Found!")
    print("Number of moves:", len(solution) - 1)

    for step, state in enumerate(solution):
        print("\nStep", step)
        print_state(state)
else:
    print("No solution found.")



