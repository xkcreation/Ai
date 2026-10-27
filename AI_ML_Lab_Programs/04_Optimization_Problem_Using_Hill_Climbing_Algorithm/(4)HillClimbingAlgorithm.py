# Objective Function
def objective_function(x):
    return -x * x + 10 * x

# Hill Climbing Function
def hill_climbing(start):
    current = start
    step_size = 1

    while True:
        # Generate neighboring solutions
        left = current - step_size
        right = current + step_size

        # Evaluate current and neighboring states
        current_value = objective_function(current)
        left_value = objective_function(left)
        right_value = objective_function(right)

        # Display current state
        print("Current x =", current, " Objective Value =", current_value)

        # Move to the better neighboring state
        if left_value > current_value:
            current = left

        elif right_value > current_value:
            current = right

        # Stop if no better neighbor exists
        else:
            return current, current_value

# Main Program
start_point = 0

best_x, best_value = hill_climbing(start_point)

print("\nBest Solution Found")
print("Optimal x =", best_x)
print("Maximum Value =", best_value)
