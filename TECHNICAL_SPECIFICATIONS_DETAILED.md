# TECHNICAL SPECIFICATIONS & IMPLEMENTATION DETAILS
## Supplementary Reference for Traffic Signal Optimization RL Systems

---

## PART A: DETAILED IMPLEMENTATION SPECIFICATIONS

### A.1 State Space Encoding Details

#### Project 1: Binary Cell Occupancy (80-dimensional)

**Cell Boundaries (Distance from TL in meters)**:
```
Cell 0:  [0.0,   7.0]  m  → 7 m wide
Cell 1:  [7.0,  14.0]  m  → 7 m wide
Cell 2:  [14.0, 21.0]  m  → 7 m wide
Cell 3:  [21.0, 28.0]  m  → 7 m wide
Cell 4:  [28.0, 40.0]  m  → 12 m wide
Cell 5:  [40.0, 60.0]  m  → 20 m wide
Cell 6:  [60.0, 100.0] m  → 40 m wide
Cell 7:  [100.0, 160.0] m → 60 m wide
Cell 8:  [160.0, 400.0] m → 240 m wide
Cell 9:  [400.0, 750.0] m → 350 m wide
```

**Lane Grouping**:
```
North approach:
  ├─ Group A (lanes N2TL_0, N2TL_1, N2TL_2): Straight + Right
  └─ Group B (lane N2TL_3): Left Turn Only

South approach:
  ├─ Group A (lanes S2TL_0, S2TL_1, S2TL_2): Straight + Right
  └─ Group B (lane S2TL_3): Left Turn Only

East approach:
  ├─ Group A (lanes E2TL_0, E2TL_1, E2TL_2): Straight + Right
  └─ Group B (lane E2TL_3): Left Turn Only

West approach:
  ├─ Group A (lanes W2TL_0, W2TL_1, W2TL_2): Straight + Right
  └─ Group B (lane W2TL_3): Left Turn Only
```

**Index Mapping** (state vector position):
```
Indices 0-9:    North Group A, cells 0-9
Indices 10-19:  North Group B, cells 0-9
Indices 20-29:  South Group A, cells 0-9
Indices 30-39:  South Group B, cells 0-9
Indices 40-49:  East Group A, cells 0-9
Indices 50-59:  East Group B, cells 0-9
Indices 60-69:  West Group A, cells 0-9
Indices 70-79:  West Group B, cells 0-9
```

**Pseudocode**:
```python
state_index = direction_offset + lane_group_offset + cell_number
```

#### Project 2: 4-Channel 24×24 Matrix

**Channel Meanings**:
```
Channel 0: Number of vehicles in each grid cell
           Range: [0, max_vehicles_per_cell]
           
Channel 1: Average speed in each grid cell
           Range: [0, v_max] (typically [0, 15] m/s)
           Normalized: [0, 1] by dividing v_max
           
Channel 2: Cumulative waiting time in each grid cell
           Range: [0, cumulative seconds]
           Normalized: [0, 1] (or raw)
           
Channel 3: Number of queued vehicles (v < 0.1 m/s)
           Range: [0, max_vehicles_per_cell]
           Binary or count
```

**Grid Resolution**: 24×24 represents ~100m × 100m area discretized into 4.2m × 4.2m cells

#### Project 3 (sumo_rl): Composite Observation

Standard obs in sumo_rl:
```python
obs = [
    # One-hot phase (n_green_phases elements)
    phase_one_hot,
    
    # Min green constraint flag
    min_green_elapsed,
    
    # Lane densities (normalized)
    density_lane_1, density_lane_2, ..., density_lane_m,
    
    # Queue lengths (normalized)
    queue_lane_1, queue_lane_2, ..., queue_lane_m
]
```

Total size = n_phases + 1 + 2*n_controlled_lanes

---

### A.2 Traffic Scenario Definitions

#### Scenario: "balanced" (Default Training)

```python
{
    'name': 'balanced',
    'description': 'Uniform traffic from all 4 directions',
    'phases': [
        {
            'start': 0.0,
            'end': 1.0,
            'direction_weights': [1, 1, 1, 1],  # N, S, E, W equal
            'left_turn_probability': 0.10
        }
    ]
}
```

**Vehicle Generation Algorithm**:
```
For each time quantum t ∈ [0, T_max]:
    For each direction d ∈ {N, S, E, W}:
        p_arrival = (weight[d] / sum_weights) * (vehicles_per_second)
        if rand() < p_arrival:
            spawn_vehicle(direction=d)
            
        if rand() < left_turn_prob:
            assign_lane = d + "_left"
        else:
            assign_lane = d + "_straight"
```

#### Scenario: "training" (Time-Varying)

5-phase scenario within single episode:

```
Phase 1 (0-1080s):  Rush hour NS [3:3:1:1] weights, 0.10 left prob
Phase 2 (1080-2160s): Balanced [1:1:1:1], 0.35 left prob (high turns)
Phase 3 (2160-3240s): Rush hour EW [1:1:3:3], 0.10 left prob
Phase 4 (3240-4320s): Random bursts [4:1:1:4], 0.20 left prob
Phase 5 (4320-5400s): Cool-down balanced [1:1:1:1], 0.10 left prob
```

**Purpose**: Agent experiences multiple patterns within single episode → learns adaptive policy

#### Scenario: "high_left_turn"

```
All directions equal weight, but 40% left-turn demand
direction_weights: [1, 1, 1, 1]
left_turn_probability: 0.40
```

---

### A.3 SUMO Configuration Details

#### Network File Properties (environment.net.xml)

```xml
<net>
  <!-- 4-way intersection -->
  <intersection id="TL" x="500" y="500">
    <!-- Incoming edges (750m long) -->
    <edge id="N2TL" from="N" to="TL" length="750">
      <lane id="N2TL_0" ... />  <!-- Lane 0: Straight -->
      <lane id="N2TL_1" ... />  <!-- Lane 1: Straight -->
      <lane id="N2TL_2" ... />  <!-- Lane 2: Straight -->
      <lane id="N2TL_3" ... />  <!-- Lane 3: Left turn only -->
    </edge>
    
    <!-- Similar for S, E, W edges -->
    
    <!-- Outgoing edges (exit lanes) -->
    <edge id="TL2N" ... />
    <edge id="TL2S" ... />
    <edge id="TL2E" ... />
    <edge id="TL2W" ... />
  </intersection>
  
  <!-- Traffic light phases -->
  <tlLogic id="TL">
    <phase duration="indefinite" state="GGGG..." />  <!-- Phase 0: NS Green -->
    <phase duration="indefinite" state="yyyy..." />  <!-- Phase 1: NS Yellow -->
    <phase duration="indefinite" state="rrrr..." />  <!-- Phase 2: EW Green -->
    <!-- ... more phases ... -->
  </tlLogic>
</net>
```

#### Key SUMO Parameters

```python
# Simulation
sumo_binary = "sumo"  # or "sumo-gui"
num_seconds = 5400    # Total simulation seconds
step_length = 1.0     # Physics update frequency (1 second)

# Vehicle parameters (defaults)
max_speed = 15 m/s
acceleration = 2.6 m/s²
deceleration = 4.5 m/s²
min_gap = 2.5 m       # Space between stopped vehicles
reaction_time = 1.0 s # Driver reaction time

# Intersection coordinates
intersection_center = (500, 500)
arm_length = 750 m
```

---

### A.4 Memory/Experience Replay Details

#### Project 1: Simple FIFO Buffer

```python
class Memory:
    def __init__(self, size_max=50000, size_min=600):
        self._samples = deque(maxlen=size_max)
        self._size_min = size_min
    
    def add_sample(self, sample):
        # sample = (state, action, reward, next_state)
        self._samples.append(sample)
        # Auto-remove oldest when full (FIFO)
    
    def get_samples(self, n):
        if len(self._samples) < self._size_min:
            return []  # Not enough data; don't train yet
        return random.sample(self._samples, min(n, len(self._samples)))
```

**Buffer Properties**:
- Max size: 50,000 transitions
- Min before sampling: 600 transitions
- Sampling: Uniform random

#### Project 2: Deque-based with Named Tuples

```python
class ReplayBuffer:
    def __init__(self, max_size=1000000, min_size=1000, batch_size=32):
        self.memory = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                                     ["state", "action", "reward", "next_state"])
    
    def add_experience(self, s, a, r, s_prime):
        exp = self.experience(state=s, action=a, reward=r, next_state=s_prime)
        self.memory.append(exp)
    
    def get_sample_from_memory(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.tensor([e.state for e in experiences]).float()
        actions = torch.tensor([e.action for e in experiences]).long()
        rewards = torch.tensor([e.reward for e in experiences]).float()
        next_states = torch.tensor([e.next_state for e in experiences]).float()
        return (states, actions, rewards, next_states)
```

**Key Properties**:
- Max: 1,000,000
- Min to start training: 1,000
- GPU conversion happens here

---

## PART B: TRAINING DYNAMICS IN DETAIL

### B.1 Episode Timeline and Time Accounting

```
Episode Start (t_sim = 0)
│
├─ [t_sim = 0 to 10s]   Action a_0 executed (green)
│
├─ [t_sim = 10 to 14s]  Yellow phase (if action changed)
│
├─ [t_sim = 14 to 24s]  Action a_1 executed
│
├─ [t_sim = 24 to 28s]  Yellow (if action changed)
│
└─ Repeat until t_sim ≥ 5400s
```

**Key Point**: 11 decision points per episode (5400 / 14 ≈ 386 full cycles)

### B.2 Batch Training Workflow

```
For each episode:
    Generate traffic
    Start SUMO
    
    For t = 0 to 5400:
        1. get_state_observation()
           └─ Costs: Iterate cars in SUMO (~1000), map to cells → O(n_cars)
        
        2. collect_waiting_times()
           └─ Sum over cars; Quadratic normalization → O(n_cars)
        
        3. choose_action(state, epsilon)
           └─ Network inference: O(feed-forward, ~500k params)
        
        4. Execute action + simulate
           └─ SUMO physics: O(complex; dominates time)
        
        5. Store (s, a, r, s') in memory
           └─ O(1) append to buffer
    
    After episode ends:
        For training_epochs = 1:  (in Project 1)
            batch = sample(memory, batch_size=100)
            
            For each (s, a, r, s') in batch:
                y_target = r + 0.75 * max_a Q(s', a'; old_weights)
                y_pred = Q(s, a; current_weights)
                loss += (y_target - y_pred)²
            
            Backprop and update weights
```

**Computational Cost per Episode**:
- Simulation: ~5-30 seconds (depends on GUI, SUMO complexity)
- Training (batch): ~1-5 seconds

---

### B.3 Typical Convergence Pattern

```
Episode Reward (Cumulative Negative)
│
│  0 ├─────────────────────────────────
│    │
│ -50 ├─ *
│    │ ** ***
│-100 ├─ *  *  *
│    │ *    *  **  *
│-150 ├─*    *  ** **  *
│    │       *  * *     *  *
│-200 ├─     *  * *  *  * *  *
│    │       **  * * **  * * *  *
│-250 ├─       **  * * *  * * * ** *
│    │           * * ** * * * * * * **
│-300 ├─          **  *****  **  * ***  ←─ Converged (avg ≈ -260)
│    │              ***  * * * * * *
└─────┴───────────────────────────────
     0  10  20  30  40  50  60  70  80  90  100
        Episode Number
```

**Characteristics**:
- **Episodes 0-20**: Rapidly declining (random → exploratory learning)
- **Episodes 20-60**: Stabilizing; oscillations as network refines
- **Episodes 60-100**: Converged; rewards hover around stable value with noise

---

## PART C: PERFORMANCE BENCHMARKS

### C.1 Fixed-Time Baseline

**Configuration**: Standard traffic controller

```
North-South Green: 30 seconds
Yellow:           4 seconds
East-West Green: 30 seconds
Yellow:           4 seconds
Cycle: 68 seconds
```

**Performance on Balanced Traffic**:
- Avg Queue Length: ~15 vehicles
- Avg Delay: ~60 seconds/vehicle
- Throughput: ~1000 vehicles/5400s ≈ 0.185 veh/s

### C.2 Trained DQN Performance

**After 100 episodes (Project 1)**:
- Avg Queue Length: ~8-10 vehicles (↓ 35%)
- Avg Delay: ~35-40 seconds/vehicle (↓ 40%)
- Throughput: ~1000 vehicles (same)

**By Traffic Scenario**:
```
Scenario        | Fixed Queue | DQN Queue | Improvement
────────────────┼─────────────┼───────────┼────────────
balanced        | 15.2        | 8.5       | -44%
rush_hour_ns    | 18.5        | 10.2      | -45%
rush_hour_ew    | 19.1        | 11.3      | -41%
high_left_turn  | 16.8        | 12.5      | -26%
random_bursts   | 22.4        | 17.8      | -20%
```

---

## PART D: HYPERPARAMETER SENSITIVITY ANALYSIS

### D.1 Learning Rate Impact

```
α = 0.0001  →  Slow convergence; stable
α = 0.001   →  Standard; balanced
α = 0.01    →  Fast oscillations; potential divergence
α = 0.1     →  Diverges; loss explodes
```

### D.2 Gamma (Discount Factor) Impact

```
γ = 0.5   →  Myopic; ignores future; slow learning
γ = 0.75  →  Moderate lookahead; practical (Project 1)
γ = 0.95  →  Strong future focus; standard RL (Project 2)
γ = 0.99  →  Very long-term; stable convergence (Project 3)
```

### D.3 Batch Size Impact

```
Batch Size = 16   →  Noisier gradients; faster updates
Batch Size = 64   →  Sweet spot; stable
Batch Size = 256  →  Smoother but slower progress
```

### D.4 Memory Size Impact

```
Max Memory = 1000    →  Rapid forgetting; only recent bias
Max Memory = 50000   →  Well-balanced buffer
Max Memory = 1000000 →  Requires more RAM; marginal improvement
```

---

## PART E: DEBUGGING & MONITORING

### E.1 Common Issues and Diagnostics

#### Issue: Loss remains flat/high

**Diagnosis**:
```python
# Check 1: Is reward being calculated?
reward_history = [r for (s,a,r,s') in memory]
assert sum(reward_history) != 0, "No rewards!"

# Check 2: Are Q-values increasing?
q_values_t0 = model.predict(states)  # Early training
q_values_tN = model.predict(states)  # Late training
assert mean(q_values_tN) > mean(q_values_t0), "No learning!"

# Check 3: Check for NaN/Inf
assert not np.isnan(loss), "NaN detected!"
```

#### Issue: Divergence (loss explodes)

**Causes & Fixes**:
```
→ Learning rate too high
  Fix: Reduce α by 10x

→ Unbounded rewards
  Fix: Clip rewards to [-1, 1]

→ Target network updates too frequent
  Fix: Increase update interval

→ Gradient explosion
  Fix: Enable gradient clipping, normalize inputs
```

#### Issue: Slow convergence

**Causes**:
```
→ State representation too sparse
  Fix: Verify state encodes sufficient info

→ Exploration too high
  Fix: Increase epsilon decay speed

→ Vehicle arrivals too sparse
  Fix: Increase n_cars_generated
```

### E.2 Real-Time Monitoring Code

```python
import numpy as np

class RewardTracker:
    def __init__(self, window=20):
        self.rewards = []
        self.window = window
    
    def add(self, r):
        self.rewards.append(r)
    
    def moving_avg(self):
        if len(self.rewards) < self.window:
            return np.mean(self.rewards)
        return np.mean(self.rewards[-self.window:])
    
    def trend(self):
        if len(self.rewards) < 2*self.window:
            return "INITIALIZING"
        recent = np.mean(self.rewards[-self.window:])
        older = np.mean(self.rewards[-2*self.window:-self.window])
        if recent < older:
            return f"IMPROVING (Δ={recent-older:.1f})"
        else:
            return f"PLATEAUING (Δ={recent-older:.1f})"
```

---

## PART F: SCALING TO MULTI-AGENT (BTP Framework)

### F.1 Multi-Intersection Setup

Project 3 (sumo_rl) supports:

```python
env = SumoEnvironment(
    net_file="4x4.net.xml",  # 4×4 grid = 16 intersections
    route_file="4x4.rou.xml",
    single_agent=False,  # Multi-agent mode
)

# Environment returns:
# obs = {ts_id_1: state_1, ts_id_2: state_2, ...}  # 16 agents
# rewards = {ts_id_1: r_1, ts_id_2: r_2, ...}
# dones = {ts_id_1: done, ts_id_2: done, "__all__": global_done}
```

### F.2 Agent Coordination Options

**Option 1: Decentralized** (independent trains)
- Each agent learns independently
- No communication between agents
- Faster, simpler, but suboptimal

**Option 2: Centralized Training, Decentralized Execution** (CTDE)
- Central critic sees all agents' states
- Individual actors execute locally
- Better coordination, more complex

**Option 3: Communication-based**
- Agents share observations/values with neighbors
- Cooperative learning
- Research area; not implemented here

---

## PART G: MATHEMATICAL CONSTANTS & UNIT CONVERSIONS

### G.1 SUMO Default Constants

```
MIN_GAP (vehicle spacing) = 2.5 m
SPEED_THRESHOLD (queued) = 0.1 m/s
DEFAULT_MAX_SPEED = 15 m/s (per lane; vehicles ~50 km/h)
DEFAULT_ACCELERATION = 2.6 m/s²
DEFAULT_DECELERATION = 4.5 m/s²
REACTION_TIME = 1.0 s
SIMULATION_TIMESTEP = 1.0 s
```

### G.2 Unit Conversions

```
Speed:
  1 m/s = 3.6 km/h = 2.237 mph

Distance:
  750 m (intersection arm length) ≈ 0.47 miles ≈ typical urban block

Time:
  5400 s = 90 minutes ≈ 1.5 hours (peak commute period)
  10 s green = 0.172 minutes (short intersection phase)
  14 s cycle = 0.233 minutes (decision frequency)
```

---

## PART H: RESEARCH EXTENSIONS

### H.1 Suggested Modifications

1. **Variable Cycle Times**
   - Action = (phase, duration)
   - Expand action space from 4 to 40 (4 phases × 10 durations)
   - More flexible control but larger action space

2. **Adaptive Arrival Patterns**
   - Use real traffic data (loop detectors)
   - Time-of-day variations
   - Incident detection

3. **Pedestrian Integration**
   - Add crossing phases to traffic light
   - Penalty for pedestrian wait time
   - Multi-objective reward

4. **Coordination Protocol**
   - Agents share queue information
   - Cooperative signal timing
   - Centralized vs distributed

5. **Advanced Reward Shaping**
   - Safety metrics (collision avoidance)
   - Emission reduction
   - Fairness (% of vehicles waiting >60s)

---

## PART I: REPRODUCIBILITY CHECKLIST

To reproduce exactly:

- [ ] SUMO version: 1.x (specify minor)
- [ ] Python version: 3.7+
- [ ] Framework: TensorFlow 2.x / PyTorch 1.x
- [ ] Numpy/Scipy versions fixed
- [ ] Random seed set globally
- [ ] GPU/CPU choice documented
- [ ] SUMO network files (.net.xml) identical
- [ ] Vehicle routing files (.rou.xml) regenerated with same seed
- [ ] Hyperparameters match config file exactly
- [ ] Batch operations deterministic (same order each run)

---

**END OF TECHNICAL SPECIFICATIONS**
