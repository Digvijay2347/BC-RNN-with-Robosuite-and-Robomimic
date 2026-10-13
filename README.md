# BC-RNN with Robosuite and Robomimic: Contact-Rich Manipulation Benchmark

Benchmarking contact-rich manipulation (Nut Assembly / Peg Insertion) using **robosuite** and **robomimic**, investigating the impact of **end-effector force/torque (F/T) history** on Behavioral Cloning (BC) and Recurrent Behavioral Cloning (BC-RNN) policies on CPU.

---

## 🎯 Overview & Motivation

In tight-tolerance contact-rich assembly tasks (such as peg-in-hole and nut assembly), policies relying strictly on Cartesian end-effector poses or visual features often suffer from:
1. **Contact Jamming & Wedging**: Millimeter alignment drift leads to high reaction forces and severe wedging against the peg.
2. **Ambiguous Contact States**: Kinematic occlusions prevent memoryless policies from recognizing exact insertion depth.

This project investigates:
* **Can wrist force/torque sensor streams govern contact compliance in imitation learning?**
* **Does recurrence (BC-RNN) capture force derivatives ($dF/dt$) to avoid jamming?**

---

## 🧪 Experimental Conditions (Ablation Matrix)

| Condition | Architecture | Input Observations | Horizon | Core Hypothesis |
| :--- | :--- | :--- | :---: | :--- |
| **Model A (Baseline)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object (23 dims) | $H=1$ (1 step) | Position-only baseline; vulnerable to contact jamming. |
| **Model B (Instant F/T)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object + **F/T** (29 dims) | $H=1$ (1 step) | Can react to contact thresholds, but sensitive to raw impact noise. |
| **Model C (F/T History)** | **BC-RNN (LSTM)** | EE Pos + Quat + Gripper + Object + **F/T** (29 dims) | **$H=10$ (Sequence)** | Learns force rate-of-change ($dF/dt$) and achieves lowest loss. |

---

## 📈 Training Convergence

All three models were trained on CPU for 50 epochs on 200 proficient human demonstrations (30,154 transitions) in 12.4 minutes total:

| Model | Architecture | Training Time (CPU) | Initial Train Loss | Final Train Loss | Best Val Loss | Best Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** | **BC (No F/T)** | 47.1 s | 0.1922 | 0.0434 | 0.0440 | Epoch 43 |
| **Model B** | **BC (Instant F/T)** | 45.9 s | 0.1935 | 0.0516 | 0.0560 | Epoch 48 |
| **Model C** | **BC-RNN (F/T History)** | 10.8 min | 0.1826 | **0.0352** 🏆 | **0.0424** 🏆 | **Epoch 50** |

![Training Curves](results/training_curves.png)

---

## 📊 Closed-Loop Simulation Evaluation (50 Episodes / Model)

Evaluated across 50 paired deterministic seeds ($42 \dots 91$) in `NutAssemblySquare` with maximum horizon $T=400$ steps (20 Hz control):

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

## 🔬 Key Scientific Takeaways

1. **Tactile Feedback Acts as a Contact Governor (Model B vs. Model A)**:
   * Blind to contact forces, Model A applies massive downward forces (mean peak $392\text{ N}$, max $1,561\text{ N}$) when misaligned with the table or peg.
   * Observing instantaneous wrist force and torque allows Model B to immediately throttle downward thrust, **reducing peak impact force by 77.8%** and dropping jamming rates from **74% down to 28%**.

2. **The Recurrent Locking Dilemma in Open-Loop Imitation (Model C)**:
   * While BC-RNN achieved the lowest offline validation loss ($0.0424$), in closed-loop evaluation its recurrent hidden state accumulated contact force history over $H=10$ steps, entering persistent limit-cycle oscillations (~70–80 N) rather than disengaging to retry alignment.

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Activate environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 2. Run simulation sanity check
python scripts/sanity_check.py

# 3. Train all 3 models on CPU (50 epochs)
python scripts/train.py --model all --epochs 50

# 4. Run closed-loop 50-episode evaluation
python scripts/evaluate.py --models all --n_episodes 50 --video_episodes 1

# 5. Plot comparative force profiles
python scripts/plot_force_profiles.py
```

For complete implementation details, Windows compatibility fixes, and architecture guides, refer to [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## 📄 License
MIT License.
