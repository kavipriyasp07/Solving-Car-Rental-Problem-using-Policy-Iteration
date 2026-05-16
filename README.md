# Solving-Car-Rental-Problem-using-Policy-Iteration


## Aim
To implement the Policy Iteration Algorithm for solving the Car Rental Problem in Reinforcement Learning and determine the optimal policy that maximizes expected rewards.
--

## Algorithm

Step-1: Initialize the environment parameters such as maximum cars, moving cost, rental reward, and discount factor.

Step-2: Initialize the state-value function and policy.

Step-3: Perform Policy Evaluation:

Compute the value function for the current policy.
Repeat until the value function converges.
Step-4: Perform Policy Improvement:

Evaluate all possible actions for each state.
Select the action with the highest expected return.
Step-5: Repeat Policy Evaluation and Policy Improvement until the policy becomes stable.

Step-6: Display the optimal policy and value function.


## Program
```
NAME:KAVIPRIYA SP
REG NO:2305002011
```

```python

#Solving car rental problem using Policy Iteration 

import numpy as np

# States: number of cars at two locations
MAX_CARS = 5

# Possible actions
ACTIONS = [-1, 0, 1]

# Discount factor
gamma = 0.9

# Initialize Value Function
V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

# Initialize Policy
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)


# Reward Function
def reward(state, action):
    return (state[0] + state[1]) * 10 - abs(action) * 2


# Next State Function
def next_state(state, action):

    cars1 = min(max(state[0] - action, 0), MAX_CARS)
    cars2 = min(max(state[1] + action, 0), MAX_CARS)

    return (cars1, cars2)


# POLICY ITERATION
stable = False

while not stable:

    # -------------------------------
    # POLICY EVALUATION
    # Bellman Expectation Equation
    # -------------------------------

    while True:

        delta = 0

        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):

                old_value = V[i, j]

                s = (i, j)
                a = policy[i, j]

                ns = next_state(s, a)

                r = reward(s, a)

                # Vπ(s) = Σ p(s',r|s,a)[r + γV(s')]
                V[i, j] = r + gamma * V[ns]

                delta = max(delta, abs(old_value - V[i, j]))

        if delta < 0.001:
            break

    # -------------------------------
    # POLICY IMPROVEMENT
    # Greedy Policy Improvement
    # -------------------------------

    stable = True

    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):

            old_action = policy[i, j]

            best_action = old_action
            best_value = -999999

            for action in ACTIONS:

                s = (i, j)
                ns = next_state(s, action)

                r = reward(s, action)

                # π'(s) = argmax Σ[r + γV(s')]
                val = r + gamma * V[ns]

                if val > best_value:
                    best_value = val
                    best_action = action

            policy[i, j] = best_action

            if old_action != best_action:
                stable = False

print("Optimal Policy:\n")
print(policy)

print("\nOptimal Value Function:\n")
print(V)

```

## Output
<img width="911" height="477" alt="image" src="https://github.com/user-attachments/assets/39ee45cc-277d-495c-832d-c839fafb2f46" />



## Result

The Policy Iteration Algorithm was successfully implemented for solving the Car Rental Problem. The algorithm iteratively evaluated and improved the policy until the optimal policy and value function were obtained, maximizing the expected reward.
