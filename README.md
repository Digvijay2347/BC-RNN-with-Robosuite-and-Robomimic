# BC-RNN with Robosuite and Robomimic

Benchmarking contact-rich manipulation (Nut Assembly / Peg Insertion) using **robosuite** and **robomimic**, investigating the impact of **end-effector force/torque (F/T) history** on Behavioral Cloning (BC) and Recurrent Behavioral Cloning (BC-RNN) policies on CPU.

---

## 🎯 Overview

In tight-tolerance contact-rich assembly tasks (such as peg-in-hole and nut assembly), policies relying strictly on Cartesian end-effector poses or visual features often suffer from:
1. **Contact Jamming & Wedging**: Millimeter alignment drift leads to high reaction forces and wedging against the peg.
2. **Ambiguous Contact States**: Visual occlusions prevent standard policies from recognizing exact insertion depth.

This project evaluates whether **end-effector force/torque sensing** and **temporal history modeling (BC-RNN)** allow the policy to learn compliant insertion strategies from low-dimensional human demonstrations.

---

## 🧪 Experimental Conditions (Ablation Matrix)

| Model | Architecture | Input Observations | Temporal Horizon | Core Hypothesis |
| :--- | :--- | :--- | :---: | :--- |
| **Model A (Baseline)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object | $H=1$ (Single step) | Position-only baseline; vulnerable to contact jamming. |
| **Model B (Instant F/T)** | Standard BC (MLP) | EE Pos + Quat + Gripper + Object + **F/T** | $H=1$ (Single step) | Can react to contact thresholds, but lacks transient context ($dF/dt$). |
| **Model C (F/T History)** | **BC-RNN (LSTM/GRU)** | EE Pos + Quat + Gripper + Object + **F/T** | $H=10$ (Sequence) | Captures force derivatives and contact transitions for compliant insertion. |

---

## 📊 Evaluation Metrics

1. **Success Rate (%)**: Percentage of episodes resulting in fully seated nut insertion.
2. **Insertion Horizon**: Number of environment steps required to complete the task.
3. **Peak & RMS Contact Forces ($\|F\|_2$)**: Contact safety and smoothness (assessing reduction in excessive contact forces).

---

## 📂 Project Structure

```text
BC-RNN-with-Robosuite-and-Robomimic/
├── configs/                  # Robomimic experiment JSON configurations
│   ├── config_bc_no_ft.json
│   ├── config_bc_with_ft.json
│   └── config_bc_rnn_with_ft.json
├── datasets/                 # Demonstration datasets (.hdf5, ignored by git)
├── scripts/                  # Automation scripts
│   ├── inspect_dataset.py    # Dataset validation & sensor key inspector
│   ├── train.py              # CPU training launcher
│   └── evaluate.py           # Closed-loop simulation evaluation & metrics
├── results/                  # Training curves, evaluation metrics, and plots
├── plan_contact_rich_bc_ft.md# Detailed technical execution plan
├── .gitignore                # Robotics and ML ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Digvijay2347/BC-RNN-with-Robosuite-and-Robomimic.git
cd BC-RNN-with-Robosuite-and-Robomimic

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install robosuite robomimic mujoco h5py matplotlib seaborn pandas tqdm
```

### 2. Download Pre-collected Demonstrations
```bash
python -m robomimic.scripts.download_datasets --tasks square --dataset_types ph --hdf5_types low_dim
```

### 3. Inspect Dataset Keys
```bash
python scripts/inspect_dataset.py --dataset datasets/square/ph/low_dim.hdf5
```

---

## 📄 License
MIT License.
