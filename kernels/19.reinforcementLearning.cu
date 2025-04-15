#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

/*
CUDA-accelerated Reinforcement Learning: Q-Learning Implementation

This implementation demonstrates how to accelerate reinforcement learning algorithms using CUDA.
Specifically, it implements Q-learning, a model-free reinforcement learning algorithm that learns
the value of actions in states by exploring the environment and receiving rewards.

Key parallelization strategies:
1. Parallel environment simulation: Multiple agents explore the environment simultaneously
2. Parallel Q-table updates: Q-values for multiple state-action pairs are updated in parallel
3. Parallel action selection: Epsilon-greedy policy decisions are made in parallel
4. Parallel reward calculation: Rewards are calculated in parallel for multiple states

The environment used is a simple grid world with:
- States: Positions in the grid (x,y coordinates)
- Actions: Up, Down, Left, Right
- Rewards: +1 for reaching the goal, -1 for hitting obstacles, -0.01 for each move
- Terminal state: Reaching the goal or maximum steps
*/

// Constants for Q-learning
const int GRID_SIZE = 10;              // Size of the grid world (GRID_SIZE x GRID_SIZE)
const int NUM_ACTIONS = 4;             // Number of possible actions (Up, Down, Left, Right)
const int NUM_STATES = GRID_SIZE * GRID_SIZE;  // Number of states in the environment
const int MAX_EPISODES = 1000;         // Maximum number of episodes to run
const int MAX_STEPS = 100;             // Maximum steps per episode
const int NUM_PARALLEL_AGENTS = 128;   // Number of parallel agents to simulate
const float ALPHA = 0.1f;              // Learning rate
const float GAMMA = 0.99f;             // Discount factor
const float EPSILON_START = 1.0f;      // Initial exploration rate
const float EPSILON_END = 0.01f;       // Final exploration rate
const float EPSILON_DECAY = 0.995f;    // Decay rate for exploration

// Environment constants
const int GOAL_X = GRID_SIZE - 1;
const int GOAL_Y = GRID_SIZE - 1;
const int NUM_OBSTACLES = 10;          // Number of obstacles in the grid

// CUDA random number generation setup
__global__ void setupCurand(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_PARALLEL_AGENTS) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// GPU kernel for epsilon-greedy action selection
__global__ void selectActions(
    float *Q_table,
    int *states,
    int *actions,
    float epsilon,
    curandState *curandState
) {
    int agentIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (agentIdx < NUM_PARALLEL_AGENTS) {
        // Get random value for epsilon-greedy policy
        float random_val = curand_uniform(&curandState[agentIdx]);
        
        // Current state of the agent
        int state = states[agentIdx];
        int action;
        
        // Epsilon-greedy policy
        if (random_val < epsilon) {
            // Explore: choose random action
            action = (int)(curand_uniform(&curandState[agentIdx]) * NUM_ACTIONS);
        } else {
            // Exploit: choose best action
            float max_q_value = -FLT_MAX;
            int best_action = 0;
            
            for (int a = 0; a < NUM_ACTIONS; a++) {
                float q_value = Q_table[state * NUM_ACTIONS + a];
                if (q_value > max_q_value) {
                    max_q_value = q_value;
                    best_action = a;
                }
            }
            action = best_action;
        }
        
        actions[agentIdx] = action;
    }
}

// GPU kernel for environment step simulation
__global__ void simulateStep(
    int *states,
    int *actions,
    int *next_states,
    float *rewards,
    bool *dones,
    int *obstacles,
    int numObstacles,
    curandState *curandState
) {
    int agentIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (agentIdx < NUM_PARALLEL_AGENTS && !dones[agentIdx]) {
        // Current state
        int state = states[agentIdx];
        int x = state % GRID_SIZE;
        int y = state / GRID_SIZE;
        
        // Execute action
        int action = actions[agentIdx];
        int new_x = x;
        int new_y = y;
        
        if (action == 0) new_y = max(0, y - 1);          // Up
        else if (action == 1) new_y = min(GRID_SIZE - 1, y + 1);  // Down
        else if (action == 2) new_x = max(0, x - 1);          // Left
        else if (action == 3) new_x = min(GRID_SIZE - 1, x + 1);  // Right
        
        // Check for obstacle collision
        bool hit_obstacle = false;
        for (int i = 0; i < numObstacles; i++) {
            int obs_x = obstacles[i] % GRID_SIZE;
            int obs_y = obstacles[i] / GRID_SIZE;
            if (new_x == obs_x && new_y == obs_y) {
                hit_obstacle = true;
                break;
            }
        }
        
        // Calculate reward and next state
        float reward;
        if (hit_obstacle) {
            reward = -1.0f;
            new_x = x;  // Stay in place if hit obstacle
            new_y = y;
        } else if (new_x == GOAL_X && new_y == GOAL_Y) {
            reward = 1.0f;
            dones[agentIdx] = true;
        } else {
            reward = -0.01f;  // Small penalty for each step
        }
        
        next_states[agentIdx] = new_y * GRID_SIZE + new_x;
        rewards[agentIdx] = reward;
    }
}

// GPU kernel for Q-table update
__global__ void updateQValues(
    float *Q_table,
    int *states,
    int *actions,
    int *next_states,
    float *rewards,
    bool *dones
) {
    int agentIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (agentIdx < NUM_PARALLEL_AGENTS && !dones[agentIdx]) {
        int state = states[agentIdx];
        int action = actions[agentIdx];
        int next_state = next_states[agentIdx];
        float reward = rewards[agentIdx];
        
        // Find max Q-value for next state
        float max_next_q = -FLT_MAX;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            float q = Q_table[next_state * NUM_ACTIONS + a];
            max_next_q = max(max_next_q, q);
        }
        
        // Q-learning update rule
        float current_q = Q_table[state * NUM_ACTIONS + action];
        float td_target = reward + GAMMA * max_next_q;
        float new_q = current_q + ALPHA * (td_target - current_q);
        
        Q_table[state * NUM_ACTIONS + action] = new_q;
    }
}

// Host function to generate random obstacles
void generateObstacles(int *obstacles, int numObstacles) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, NUM_STATES - 1);
    
    for (int i = 0; i < numObstacles; i++) {
        int obstacle = dis(gen);
        
        // Make sure obstacles are not at start or goal
        while (obstacle == 0 || obstacle == (GRID_SIZE * GRID_SIZE - 1)) {
            obstacle = dis(gen);
        }
        
        obstacles[i] = obstacle;
    }
}

// Visualization function for Q-table
void visualizeQTable(float *Q_table, int *obstacles, int numObstacles) {
    std::cout << "\nQ-Table Visualization (Max Q-value per state):" << std::endl;
    std::cout << "Goal is at position (" << GOAL_X << ", " << GOAL_Y << ")" << std::endl;
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            int state = y * GRID_SIZE + x;
            
            // Check if this position is an obstacle
            bool is_obstacle = false;
            for (int i = 0; i < numObstacles; i++) {
                if (obstacles[i] == state) {
                    is_obstacle = true;
                    break;
                }
            }
            
            if (is_obstacle) {
                std::cout << std::setw(6) << "X ";
            } else if (x == GOAL_X && y == GOAL_Y) {
                std::cout << std::setw(6) << "G ";
            } else {
                // Find max Q-value for this state
                float max_q = -FLT_MAX;
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    max_q = std::max(max_q, Q_table[state * NUM_ACTIONS + a]);
                }
                std::cout << std::setw(6) << std::fixed << std::setprecision(2) << max_q << " ";
            }
        }
        std::cout << std::endl;
    }
}

int main() {
    // Allocate host memory
    float *h_Q_table = new float[NUM_STATES * NUM_ACTIONS]();
    int *h_obstacles = new int[NUM_OBSTACLES];
    
    // Generate random obstacles
    generateObstacles(h_obstacles, NUM_OBSTACLES);
    
    // Allocate device memory
    float *d_Q_table;
    int *d_states, *d_next_states, *d_actions, *d_obstacles;
    float *d_rewards;
    bool *d_dones;
    curandState *d_curandState;
    
    cudaMalloc((void**)&d_Q_table, NUM_STATES * NUM_ACTIONS * sizeof(float));
    cudaMalloc((void**)&d_states, NUM_PARALLEL_AGENTS * sizeof(int));
    cudaMalloc((void**)&d_next_states, NUM_PARALLEL_AGENTS * sizeof(int));
    cudaMalloc((void**)&d_actions, NUM_PARALLEL_AGENTS * sizeof(int));
    cudaMalloc((void**)&d_rewards, NUM_PARALLEL_AGENTS * sizeof(float));
    cudaMalloc((void**)&d_dones, NUM_PARALLEL_AGENTS * sizeof(bool));
    cudaMalloc((void**)&d_obstacles, NUM_OBSTACLES * sizeof(int));
    cudaMalloc((void**)&d_curandState, NUM_PARALLEL_AGENTS * sizeof(curandState));
    
    // Initialize Q-table and obstacles on device
    cudaMemcpy(d_Q_table, h_Q_table, NUM_STATES * NUM_ACTIONS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacles, h_obstacles, NUM_OBSTACLES * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize random states
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_PARALLEL_AGENTS + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    setupCurand<<<blocksPerGrid, threadsPerBlock>>>(d_curandState, seed);
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Main training loop
    float epsilon = EPSILON_START;
    int total_steps = 0;
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        // Reset agents to starting positions
        std::vector<int> initial_states(NUM_PARALLEL_AGENTS, 0); // All agents start at (0,0)
        std::vector<bool> initial_dones(NUM_PARALLEL_AGENTS, false);
        
        cudaMemcpy(d_states, initial_states.data(), NUM_PARALLEL_AGENTS * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dones, initial_dones.data(), NUM_PARALLEL_AGENTS * sizeof(bool), cudaMemcpyHostToDevice);
        
        int steps = 0;
        bool all_done = false;
        
        while (!all_done && steps < MAX_STEPS) {
            // Select actions using epsilon-greedy policy
            selectActions<<<blocksPerGrid, threadsPerBlock>>>(
                d_Q_table, d_states, d_actions, epsilon, d_curandState
            );
            
            // Simulate one step for all agents
            simulateStep<<<blocksPerGrid, threadsPerBlock>>>(
                d_states, d_actions, d_next_states, d_rewards, d_dones, 
                d_obstacles, NUM_OBSTACLES, d_curandState
            );
            
            // Update Q-values
            updateQValues<<<blocksPerGrid, threadsPerBlock>>>(
                d_Q_table, d_states, d_actions, d_next_states, d_rewards, d_dones
            );
            
            // Copy next states to current states
            cudaMemcpy(d_states, d_next_states, NUM_PARALLEL_AGENTS * sizeof(int), cudaMemcpyDeviceToDevice);
            
            // Check if all agents are done
            std::vector<bool> h_dones(NUM_PARALLEL_AGENTS);
            cudaMemcpy(h_dones.data(), d_dones, NUM_PARALLEL_AGENTS * sizeof(bool), cudaMemcpyDeviceToHost);
            
            all_done = true;
            for (int i = 0; i < NUM_PARALLEL_AGENTS; i++) {
                if (!h_dones[i]) {
                    all_done = false;
                    break;
                }
            }
            
            steps++;
            total_steps++;
        }
        
        // Decay epsilon
        epsilon = std::max(EPSILON_END, epsilon * EPSILON_DECAY);
        
        // Print progress every 100 episodes
        if ((episode + 1) % 100 == 0) {
            std::cout << "Episode " << episode + 1 << " completed. Epsilon: " << epsilon << std::endl;
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Copy Q-table back to host
    cudaMemcpy(h_Q_table, d_Q_table, NUM_STATES * NUM_ACTIONS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nTraining complete!" << std::endl;
    std::cout << "Total steps: " << total_steps << std::endl;
    std::cout << "Training time: " << duration << " ms" << std::endl;
    
    // Visualize Q-table
    visualizeQTable(h_Q_table, h_obstacles, NUM_OBSTACLES);
    
    // Free device memory
    cudaFree(d_Q_table);
    cudaFree(d_states);
    cudaFree(d_next_states);
    cudaFree(d_actions);
    cudaFree(d_rewards);
    cudaFree(d_dones);
    cudaFree(d_obstacles);
    cudaFree(d_curandState);
    
    // Free host memory
    delete[] h_Q_table;
    delete[] h_obstacles;
    
    return 0;
}