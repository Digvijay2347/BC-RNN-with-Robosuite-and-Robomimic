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

```text
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

## 🛠️ Complete End-to-End Setup & Reproduction Guide

This guide enables anyone to reproduce the complete project from scratch on any standard machine (Windows, Linux, or macOS).

### 1. Hardware & System Prerequisites
* **CPU**: Modern multi-core x86_64 or ARM processor (4+ cores recommended).
* **RAM**: 8 GB RAM or higher (peak usage is lightweight: ~500 MB).
* **Disk Space**: ~2.5 GB available space (dataset ~55 MB, virtualenv ~1.5 GB, logs & models ~500 MB).
* **Python**: Python 3.10 or 3.11 installed.
* **Git**: Installed and available in terminal.

---

### 2. Clone the Repository & Virtual Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/BC-RNN-with-Robosuite-and-Robomimic.git
cd BC-RNN-with-Robosuite-and-Robomimic

# Create isolated Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

---

### 3. Install Core Dependencies

Install CPU-optimized PyTorch first (saving over 2.5 GB of unused CUDA binaries), followed by pinned simulation libraries:

```bash
# 1. Install CPU-only PyTorch
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install pinned MuJoCo and Robosuite
# NOTE: mujoco==3.2.7 is required to prevent the MjData.qM sparse matrix deprecation in 3.10+
# NOTE: robosuite==1.4.1 matches the bit-exact state frames used by canonical robomimic datasets
python -m pip install "mujoco==3.2.7" "robosuite==1.4.1" --no-deps

# 3. Install robomimic without pulling unpinned dependencies
python -m pip install robomimic --no-deps

# 4. Install scientific computing, visualization, and data libraries
python -m pip install h5py matplotlib seaborn pandas tqdm tensorboard tensorboardX imageio "imageio[ffmpeg]" pynput termcolor numba scipy opencv-python
```

---

### 4. Applied Compatibility Patches (Automated or Manual)

Because robotics simulators were traditionally designed for Linux/EGL, several targeted fixes ensure seamless execution on Windows and modern MuJoCo bindings:

#### A. Modern MuJoCo Binding in Robomimic
In `venv/Lib/site-packages/robomimic/envs/env_robosuite.py`, replace legacy `mujoco_py` references with Google DeepMind's official `mujoco` bindings:
```python
import mujoco
```

#### B. Windows OpenGL Backend Patch
In `venv/Lib/site-packages/robosuite/utils/binding_utils.py`, set the default OpenGL context to `"wgl"` on Windows instead of Linux `"egl"`:
```python
if sys.platform == "win32":
    os.environ["MUJOCO_GL"] = "wgl"
```

#### C. Force/Torque Channel Injection
In `venv/Lib/site-packages/robomimic/envs/env_robosuite.py`, ensure `get_observation()` queries the robot's load cell:
```python
di["robot0_eef_force"] = np.array(self.env.robots[0].ee_force)
di["robot0_eef_torque"] = np.array(self.env.robots[0].ee_torque)
```

#### D. Offscreen Renderer Protection
In `venv/Lib/site-packages/robomimic/envs/env_robosuite.py`, safely wrap the optional `egl_probe` check in a `try...except` block so offscreen rendering succeeds on non-EGL platforms.

---

### 5. Step-by-Step Execution Pipeline

Run each command in sequence from the project root:

#### Step 5.1: Verify Simulation Sanity
Tests environment instantiation, robot kinematics, and verifies that 6-axis F/T telemetry is actively streaming:
```bash
python scripts/sanity_check.py
```

#### Step 5.2: Launch Robot 3D Visualizer & Render Cameras
Launch the real-time interactive 3D viewer (with mouse orbit/pan/zoom) or render high-resolution 512×512 camera views:
```bash
# Real-time free-cam interactive viewer
python scripts/visualize_interactive.py

# Render camera snapshots (frontview, agentview, sideview) to results/
python scripts/render_snapshot.py
```

#### Step 5.3: Download Dataset & Extract Force/Torque Channels
Download the official 200 human demonstrations (50.5 MB) and re-simulate to extract synchronized wrist forces:
```bash
# Download official proficient-human Square dataset
python -m robomimic.scripts.download_datasets --download_dir datasets --tasks square --dataset_types ph --hdf5_types low_dim

# Re-simulate all 200 trajectories to populate 6-axis F/T channels
python -m robomimic.scripts.dataset_states_to_obs --dataset datasets/square/ph/low_dim_v141.hdf5 --output_name low_dim_with_ft.hdf5 --copy_rewards --copy_dones

# Verify dataset integrity and sensor keys
python scripts/inspect_dataset.py datasets/square/ph/low_dim_with_ft.hdf5

# Generate 4-panel sensor telemetry plot for demonstration 0
python scripts/plot_demo_telemetry.py demo_0

# Replay demonstration in 3D simulator and export MP4 video
python scripts/playback_demo.py --demo demo_0 --video results/demo_0_playback.mp4
```

#### Step 5.4: Generate Model Configurations
Generates validated robomimic experiment JSON files for Model A, Model B, and Model C in `configs/`:
```bash
python scripts/generate_configs.py
```

#### Step 5.5: Train All Three Policies on CPU
Runs 50 epochs of batch training for each model, logs validation metrics, saves best checkpoints in `models/`, and produces loss curves:
```bash
# Train all 3 models sequentially (~12.4 minutes total on CPU)
python scripts/train.py --model all --epochs 50

# Or train individually:
# python scripts/train.py --model bc_no_ft --epochs 50
# python scripts/train.py --model bc_with_ft --epochs 50
# python scripts/train.py --model bc_rnn_with_ft --epochs 50
```

#### Step 5.6: Run Closed-Loop Simulation Evaluation
Executes 50 deterministic evaluation episodes per model (150 total rollouts), logs per-step force profiles and jamming events, records representative MP4 rollout videos, and outputs metrics:
```bash
python scripts/evaluate.py --models all --n_episodes 50 --video_episodes 1
```

#### Step 5.7: Generate Comparative Force Profile Curves
Plots population mean contact force curves over time across all 50 evaluation seeds:
```bash
python scripts/plot_force_profiles.py
```

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

## 🔍 Troubleshooting & Known Gotchas

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `AttributeError: 'MjData' object has no attribute 'qM'` | MuJoCo $\ge 3.10$ removed the sparse mass matrix `qM` in favor of CSR `M`. `robosuite` relies on `qM` for OSC controllers. | Pin `mujoco==3.2.7` via `pip install "mujoco==3.2.7"`. |
| `KeyError: 'gripper0_ft_frame'` during state reload | `robosuite 1.5.x` renamed sensor frames compared to dataset version 1.4.1. | Pin `robosuite==1.4.1` via `pip install "robosuite==1.4.1"`. |
| `ImportError: DLL load failed` or black window on Windows | Default Linux `MUJOCO_GL="egl"` assignment fails on Windows OpenGL. | Set `os.environ["MUJOCO_GL"] = "wgl"` in `robosuite.utils.binding_utils`. |
| `ModuleNotFoundError: No module named 'egl_probe'` | Robosuite checks for Linux headless EGL devices. | Wrap import in `try...except (ImportError, Exception): pass`. |

---

## 📄 License
MIT License.
