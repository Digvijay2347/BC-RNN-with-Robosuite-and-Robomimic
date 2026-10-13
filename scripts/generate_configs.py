"""
Configuration Generator Script for Robomimic BC and BC-RNN Experiments.
Generates 3 controlled experiment configurations:
1. configs/config_bc_no_ft.json        : Baseline BC-MLP without Force/Torque (Position only)
2. configs/config_bc_with_ft.json      : Ablation 1 BC-MLP with Instantaneous Force/Torque
3. configs/config_bc_rnn_with_ft.json  : Ablation 2 BC-RNN (LSTM) with Force/Torque History (H=10)
"""

import os
import sys
import json
import robomimic.config as Config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_base_config(exp_name, output_dir, dataset_path="datasets/square/ph/low_dim_with_ft.hdf5"):
    config = Config.config_factory(algo_name="bc")

    # Experiment settings
    config.experiment.name = exp_name
    config.experiment.validate = True
    config.experiment.logging.terminal_output_to_txt = False
    config.experiment.logging.log_tb = True
    config.experiment.save.enabled = True
    config.experiment.save.every_n_epochs = 10
    config.experiment.save.on_best_validation = True
    config.experiment.epoch_every_n_steps = 100
    config.experiment.validation_epoch_every_n_steps = 10
    config.experiment.rollout.enabled = False  # Keep training fast on CPU; rollouts handled in Phase 5

    # Training settings
    config.train.data = dataset_path
    config.train.output_dir = output_dir
    config.train.num_data_workers = 0  # Safe for Windows environment
    config.train.hdf5_cache_mode = "all"  # Cache low-dim dataset in RAM
    config.train.hdf5_use_swmr = False
    config.train.hdf5_normalize_obs = False
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid"
    config.train.cuda = False  # CPU training
    config.train.batch_size = 64
    config.train.num_epochs = 50
    config.train.seed = 42

    # Policy / Network settings
    config.algo.optim_params.policy.optimizer_type = "adam"
    config.algo.optim_params.policy.learning_rate.initial = 1e-4
    config.algo.optim_params.policy.learning_rate.decay_factor = 0.1
    config.algo.optim_params.policy.learning_rate.epoch_schedule = [40]
    config.algo.loss.l2_weight = 1.0
    config.algo.loss.l1_weight = 0.0
    config.algo.loss.cos_weight = 0.0
    config.algo.actor_layer_dims = [300, 400]
    config.algo.gaussian.enabled = False
    config.algo.gmm.enabled = False
    config.algo.vae.enabled = False

    return config


def generate_all_configs():
    os.makedirs("configs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 72)
    print("  GENERATING EXPERIMENT CONFIGURATIONS (PHASE 3)")
    print("=" * 72)

    # Core observation modalities without force/torque
    baseline_obs = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ]

    # Observation modalities with 6-axis force & torque
    ft_obs = baseline_obs + [
        "robot0_eef_force",
        "robot0_eef_torque",
    ]

    # -------------------------------------------------------------
    # 1. Config A: Baseline BC (No Force/Torque, MLP)
    # -------------------------------------------------------------
    cfg_a = build_base_config("bc_no_ft", "models/bc_no_ft")
    cfg_a.algo.rnn.enabled = False
    cfg_a.train.seq_length = 1
    cfg_a.train.pad_seq_length = True
    cfg_a.observation.modalities.obs.low_dim = baseline_obs
    cfg_a_path = os.path.join("configs", "config_bc_no_ft.json")
    cfg_a.dump(filename=cfg_a_path)
    print(f"\n[1/3] Generated Config A (Baseline - No F/T):")
    print(f"      Path       : {cfg_a_path}")
    print(f"      Type       : Standard BC-MLP (Dense Layers [300, 400])")
    print(f"      Horizon    : H = 1 (Instantaneous)")
    print(f"      Inputs ({len(baseline_obs)} keys): {baseline_obs}")

    # -------------------------------------------------------------
    # 2. Config B: Ablation 1 (Instantaneous Force/Torque, MLP)
    # -------------------------------------------------------------
    cfg_b = build_base_config("bc_with_ft", "models/bc_with_ft")
    cfg_b.algo.rnn.enabled = False
    cfg_b.train.seq_length = 1
    cfg_b.train.pad_seq_length = True
    cfg_b.observation.modalities.obs.low_dim = ft_obs
    cfg_b_path = os.path.join("configs", "config_bc_with_ft.json")
    cfg_b.dump(filename=cfg_b_path)
    print(f"\n[2/3] Generated Config B (Ablation 1 - Instant F/T):")
    print(f"      Path       : {cfg_b_path}")
    print(f"      Type       : BC-MLP with Force/Torque (Dense Layers [300, 400])")
    print(f"      Horizon    : H = 1 (Instantaneous)")
    print(f"      Inputs ({len(ft_obs)} keys): {ft_obs}")

    # -------------------------------------------------------------
    # 3. Config C: Ablation 2 (BC-RNN with Force/Torque History)
    # -------------------------------------------------------------
    cfg_c = build_base_config("bc_rnn_with_ft", "models/bc_rnn_with_ft")
    cfg_c.algo.rnn.enabled = True
    cfg_c.algo.rnn.horizon = 10
    cfg_c.algo.rnn.hidden_dim = 400
    cfg_c.algo.rnn.rnn_type = "LSTM"
    cfg_c.algo.rnn.num_layers = 2
    cfg_c.algo.rnn.open_loop = False
    cfg_c.algo.rnn.kwargs.bidirectional = False
    cfg_c.train.seq_length = 10
    cfg_c.train.pad_seq_length = True
    cfg_c.observation.modalities.obs.low_dim = ft_obs
    cfg_c_path = os.path.join("configs", "config_bc_rnn_with_ft.json")
    cfg_c.dump(filename=cfg_c_path)
    print(f"\n[3/3] Generated Config C (Ablation 2 - BC-RNN with F/T History):")
    print(f"      Path       : {cfg_c_path}")
    print(f"      Type       : Recurrent BC-RNN (2-Layer LSTM, hidden_dim=400)")
    print(f"      Horizon    : H = 10 (Sequence History)")
    print(f"      Inputs ({len(ft_obs)} keys): {ft_obs}")

    print("\n" + "=" * 72)
    print("  ALL 3 CONFIGURATIONS READY FOR TRAINING")
    print("=" * 72)


if __name__ == "__main__":
    generate_all_configs()
