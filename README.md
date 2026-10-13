# BC-RNN with Robosuite and Robomimic: Contact-Rich Manipulation Benchmark

Benchmarking contact-rich robotic manipulation (Nut Assembly / Peg Insertion) using **robosuite** and **robomimic**, investigating the role of **6-axis end-effector force/torque (F/T) sensing** and **temporal recurrence (BC-RNN)** on imitation learning policies running efficiently on CPU.

---

## 🎯 Detailed Project Objective & Scientific Motivation

### The Challenge of Contact-Rich Robotic Assembly
In high-precision manufacturing and industrial robotics, assembly operations such as **peg-in-hole insertion** and **nut-threading** represent some of the most difficult manipulation problems:
* **Sub-Millimeter Clearances**: The tolerance between the square nut hole and the square peg is fractions of a millimeter. Even a minor angular misalignment ($\approx 1^\circ$) or translational drift ($\approx 1\text{ mm}$) leads to **jamming and wedging**, where opposing contact normal forces create frictional lock-up.
* **The Failure of Pure Vision & Proprioception**: Vision-only or position-only policies suffer from visual occlusion (the robot gripper and nut occlude the insertion hole as they near the mating surface). Proprioceptive encoders only report kinematic joint angles and Cartesian end-effector coordinates—they provide zero information about whether the tool is in free space or exerting destructive forces into a rigid fixture.
* **Compounding Errors in Behavioral Cloning**: Offline imitation learning fits a policy $\pi(a \mid o)$ on human demonstrations. During closed-loop rollouts, small tracking deviations compound, steering the robot into out-of-distribution states. A position-based policy will stubbornly push along its memorized open-loop path, driving the motors into severe high-force collisions (often $>1,000\text{ N}$).

### Core Research Questions
1. **Can 6-axis wrist Force/Torque (F/T) sensing serve as an implicit contact governor?** Can an imitation learning policy learn compliant behaviors directly from tactile feedback to avoid jamming and lower contact stresses?
2. **Does temporal recurrence (BC-RNN) overcome the high-frequency noise of raw force readings?** Raw force sensors produce noisy, instantaneous spikes upon initial impact. Can an LSTM policy over a temporal horizon ($H=10$) learn force derivatives ($dF/dt$) to differentiate gentle sliding from hard collisions?
3. **How do memoryless policies (BC-MLP) compare to sequence models (BC-RNN) in closed-loop physical interaction?**

---

## 🧪 Detailed Explanation of the Three Strategies

To systematically isolate the influence of force sensing and temporal memory, we formulate a 3-way ablation benchmark using the canonical 7-DoF Franka Emika Panda robot on the `NutAssemblySquare` task:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ABLATION MATRIX                                  │
├───────────────────┬──────────────┬──────────────┬─────────┬─────────────────┤
│ Strategy          │ Architecture │ Inputs       │ Horizon │ Role in Study   │
├───────────────────┼──────────────┼──────────────┼─────────┼─────────────────┤
│ Model A (Baseline)│ BC (MLP)     │ 23 dims      │ H = 1   │ Negative Control│
│ Model B (Instant) │ BC (MLP)     │ 29 dims + FT │ H = 1   │ Tactile Aware   │
│ Model C (Recurrent│ BC-RNN (LSTM)│ 29 dims + FT │ H = 10  │ Temporal Tactile│
└───────────────────┴──────────────┴──────────────┴─────────┴─────────────────┘
```

---

### Strategy 1: Model A — Standard Behavioral Cloning Baseline (BC-MLP, No F/T)

* **Architecture**: Deep Feedforward Multi-Layer Perceptron (MLP) with hidden layers `[300, 400]`, ReLU activations, and a continuous Gaussian action head.
* **Observation Space (23 Dimensions)**:
  * `robot0_eef_pos` (3D): End-effector Cartesian position $(x, y, z)$.
  * `robot0_eef_quat` (4D): End-effector orientation quaternion $(w, x, y, z)$.
  * `robot0_gripper_qpos` (2D): Finger joint positions (gripper aperture width).
  * `object` (14D): Pose of the square nut relative to the peg and robot base.
  * **Force/Torque**: *None (0D)*.
* **Temporal Structure**: Memoryless, single-step Markovian assumption ($H=1$):
  $$a_t = \pi_\theta(o_t)$$
* **Scientific Role & Theoretical Mechanics**:
  Model A serves as the **unaugmented negative control**. Because it has no access to contact forces, it operates entirely in kinematic space. When the square nut strikes the chamfer or rim of the peg, Model A receives no signal that an obstacle has been encountered. It continues executing its learned trajectory, treating mechanical resistance as non-existent and generating large destructive downward forces.

---

### Strategy 2: Model B — Tactile-Augmented Behavioral Cloning (BC-MLP with Instant F/T)

* **Architecture**: Feedforward Multi-Layer Perceptron (MLP) with hidden layers `[300, 400]`, ReLU activations, and a continuous Gaussian action head.
* **Observation Space (29 Dimensions)**:
  * Proprioception & Object state (23 dimensions from Model A).
  * `robot0_eef_force` (3D): Measured 3-axis reaction forces $(F_x, F_y, F_z)$ at the wrist sensor in Newtons.
  * `robot0_eef_torque` (3D): Measured 3-axis reaction torques $(\tau_x, \tau_y, \tau_z)$ at the wrist sensor in Newton-meters.
* **Temporal Structure**: Memoryless with instantaneous tactile observation ($H=1$):
  $$a_t = \pi_\theta(o_t, F_t, \tau_t)$$
* **Scientific Role & Theoretical Mechanics**:
  Model B investigates whether **instantaneous tactile awareness** alone is sufficient to induce compliant assembly behavior. By having immediate access to $F$ and $\tau$, the neural network can approximate an implicit **impedance control law**:
  * If a high vertical reaction force ($F_z \gg 0$) is sensed, the policy can throttle its downward displacement command $\Delta z$.
  * If lateral contact forces ($F_x, F_y$) are sensed, the policy can adjust horizontal alignment toward the direction of least resistance.
  * *Limitation*: Because $H=1$, Model B has no memory of previous forces; it cannot compute numerical derivatives ($\Delta F / \Delta t$) or distinguish between transient impact shockwaves and steady-state resting contact.

---

### Strategy 3: Model C — Recurrent Behavioral Cloning with Force/Torque History (BC-RNN with LSTM)

* **Architecture**: 2-Layer Recurrent Long Short-Term Memory (LSTM) network with hidden state dimension **400**, followed by an MLP action decoder `[300, 400]` and Gaussian action head.
* **Observation Space (29 Dimensions)**:
  * Full observation vector identical to Model B (Proprioception + Object + 6-axis F/T).
* **Temporal Structure**: Recurrent sequence modeling over a sliding history window of **$H = 10$ timesteps** ($0.5$ seconds of continuous interaction at 20 Hz):
  $$h_t, c_t = \text{LSTM}(o_t, h_{t-1}, c_{t-1})$$
  $$a_t = \pi_\theta(h_t)$$
* **Scientific Role & Theoretical Mechanics**:
  Model C evaluates whether **temporal force dynamics** enable superior contact reasoning over memoryless snapshots:
  * **Force Derivative Estimation ($dF/dt$)**: The recurrent hidden state $h_t$ can capture the rate of force buildup, allowing the policy to anticipate hard impacts before peak force is reached.
  * **Sensor Noise Filtering**: Physical contact sensors exhibit high-frequency vibrations upon impact. The gating mechanisms of the LSTM allow the model to integrate forces across time, effectively acting as an adaptive low-pass filter.
  * **Phase Identification**: The sequence memory allows the network to distinguish distinct operational phases of assembly: (1) Free-space approach $\to$ (2) First table touch $\to$ (3) Hole search/alignment $\to$ (4) Downward sliding $\to$ (5) Final seating.

---

## 📈 Training Convergence & Loss Profile

All three models were trained on CPU for 50 epochs on 200 proficient human demonstrations (30,154 state-action pairs) using a batch size of 64:

| Model | Architecture | CPU Train Time | Initial Train Loss | Final Train Loss | Best Val Loss | Best Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | **BC (No F/T)** | 47.1 s | 0.1922 | 0.0434 | 0.0440 | Epoch 43 |
| **Model B** | **BC (Instant F/T)** | 45.9 s | 0.1935 | 0.0516 | 0.0560 | Epoch 48 |
| **Model C** | **BC-RNN (F/T History)** | 10.8 min | 0.1826 | **0.0352** 🏆 | **0.0424** 🏆 | **Epoch 50** |

![Training Curves](results/training_curves.png)

* **Key Takeaway**: Model C achieved the lowest training error and validation loss, confirming that recurrence over force histories provides superior predictive power for offline state-action mapping.

---

## 📊 Closed-Loop Simulation Evaluation (50 Episodes / Model)

To evaluate real-world policy behavior, each model was evaluated across **50 paired deterministic seeds** (`seeds 42..91`) inside the `NutAssemblySquare` simulation workcell with maximum horizon $T=400$ steps (20 Hz control):

| Metric | Model A: BC Baseline (No F/T) | Model B: BC (Instant F/T) | Model C: BC-RNN (F/T History) | Physical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Mean Peak Contact Force ($\|F\|_2$)** | **392.27 N** | **87.23 N** 🛡️ | **129.79 N** | **77.8% lower peak impact force with F/T** |
| **Worst-Case Force Spike** | **1,561.11 N** 💥 | **431.44 N** 🛡️ | **1,364.34 N** | Model A causes severe destructive impacts |
| **Mean RMS Contact Force** | **52.58 N** | **19.05 N** 🛡️ | **74.85 N** | **63.8% reduction in sustained contact stress** |
| **Jamming / Wedging Rate (%)** | **74.0%** | **28.0%** 🛡️ | **100.0%** | **Drastic drop in jammed episodes with Instant F/T** |
| **Evaluation Runtime (CPU)** | 5.5 min | 5.3 min | 5.7 min | Total pipeline: **16.5 min** (~60 steps/sec) |

![Evaluation Metrics](results/evaluation_metrics.png)

### Contact Force Modulation Over Time

![Force Profile Comparison](results/force_profile_comparison.png)

---

## 🔬 Key Scientific Findings & Analysis

### 1. Tactile Feedback Acts as a Contact Governor (Model B vs. Model A)
* **Destructive Impact Prevention**: Blind to contact forces, **Model A** repeatedly hammers downward into misaligned surfaces, generating mean peak contact forces of **392.3 N** and severe spikes exceeding **1,561 N**.
* **Compliant Accommodation**: **Model B** observes reaction forces immediately upon touching the peg or table, throttling aggressive downward velocity. This yields a **77.8% reduction in peak impact forces** and reduces the jamming rate from **74% down to 28%**.

### 2. The Recurrent Locking Dilemma in Open-Loop Imitation (Model C)
* Although **BC-RNN (Model C)** achieved the lowest offline validation loss ($0.0424$), in closed-loop simulation it exhibited a **100% jamming rate** characterized by persistent limit-cycle oscillations (~70–80 N).
* **The Mechanism**: Because the 2-layer LSTM integrates force history over $H=10$ steps, prolonged contact creates a strong inertia of resistance within the recurrent hidden state. Without explicit recovery/disengagement demonstrations (such as in DAgger or interactive RL), the hidden state locks into a contact-bound state, continuously executing micro-compliant adjustments rather than breaking contact to re-align.

---

## 📂 Project Structure

```text
project-root/
├── configs/                       # Robomimic experiment JSON configurations
│   ├── config_bc_no_ft.json       # Model A config (MLP, no F/T)
│   ├── config_bc_with_ft.json     # Model B config (MLP, instant F/T)
│   └── config_bc_rnn_with_ft.json # Model C config (LSTM, F/T history H=10)
├── datasets/                      # Demonstration datasets (.hdf5, ignored by git)
│   └── square/ph/
│       ├── low_dim_v141.hdf5      # Raw human demos (50.5 MB)
│       └── low_dim_with_ft.hdf5   # Re-simulated with 6-axis F/T (53.5 MB)
├── models/                        # Trained policy weights (ignored by git)
│   ├── bc_no_ft_best.pth          # Best checkpoint Model A
│   ├── bc_with_ft_best.pth        # Best checkpoint Model B
│   └── bc_rnn_with_ft_best.pth    # Best checkpoint Model C
├── results/                       # Plots, telemetry, and videos
│   ├── demo_0_playback.mp4        # Demonstration replay video
│   ├── demo_0_telemetry.png       # Demonstration sensor telemetry
│   ├── eval_bc_no_ft_ep0.mp4      # Model A rollout recording
│   ├── eval_bc_with_ft_ep0.mp4    # Model B rollout recording
│   ├── eval_bc_rnn_with_ft_ep0.mp4# Model C rollout recording
│   ├── evaluation_metrics.png     # 4-panel evaluation comparison chart
│   ├── evaluation_results.json    # Complete 150-episode rollout telemetry
│   ├── force_profile_comparison.png # Force over time curves
│   └── training_curves.png        # Train vs Validation loss curves
├── scripts/                       # Executable python pipeline
│   ├── sanity_check.py            # Environment & sensor verification
│   ├── render_snapshot.py         # Robot camera snapshot generator
│   ├── visualize_interactive.py   # 3D Free-Cam GUI visualizer
│   ├── inspect_dataset.py         # HDF5 dataset validator
│   ├── plot_demo_telemetry.py     # Trajectory & force curve plotter
│   ├── playback_demo.py           # 3D demo replayer & MP4 exporter
│   ├── generate_configs.py        # Config builder
│   ├── train.py                   # CPU training pipeline
│   ├── evaluate.py                # Closed-loop evaluation pipeline
│   └── plot_force_profiles.py     # Force profile curves plotter
├── plan_contact_rich_bc_ft.md     # Initial technical plan
├── PROJECT_DOCUMENTATION.md       # Comprehensive runbook & technical manual
└── README.md                      # Public project overview
```

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 2. Run simulation sanity check
python scripts/sanity_check.py

# 3. Train all 3 models on CPU (50 epochs each)
python scripts/train.py --model all --epochs 50

# 4. Run closed-loop 50-episode evaluation
python scripts/evaluate.py --models all --n_episodes 50 --video_episodes 1

# 5. Plot comparative contact force curves
python scripts/plot_force_profiles.py
```

For complete technical documentation, Windows WGL/MuJoCo compatibility patches, and complete command history, refer to [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## 📄 License
MIT License.
