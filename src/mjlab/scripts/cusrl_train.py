"""Script to train RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import tyro

import cusrl
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainConfig:
  env: Any
  agent: cusrl.Trainer.Factory
  registry_name: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False


def run_train(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  registry_name: str | None = None

  if isinstance(cfg.env, TrackingEnvCfg):
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    cfg.env.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "cusrl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  env = gym.make(
    task, cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  if cfg.video:
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=os.path.join(log_dir, "videos", "train"),
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")


  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  trainer = cfg.agent(
    environment=cusrl.environment.MjlabEnvAdapter(env),
    logger_factory=cusrl.make_logger_factory("wandb", log_dir),
    agent_overrides={"device": cfg.device},
  )

  # run training
  trainer.run_training_loop()

  env.close()


def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "cusrl_cfg_entry_point")

  config = TrainConfig(env=env_cfg, agent=agent_cfg)
  args = tyro.cli(
    dict,
    args=remaining_args,
    default=cusrl.utils.to_dict(config),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  config = cusrl.utils.from_dict(config, args)
  del env_cfg, agent_cfg, remaining_args

  run_train(chosen_task, config)


if __name__ == "__main__":
  main()
