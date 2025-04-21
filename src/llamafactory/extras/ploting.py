# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any, Dict, List

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt


logger = logging.get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: List[Dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""
    Plots loss curves in LlamaBoard.
    """
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def plot_loss(save_dictionary: str, keys: List[str] = None) -> None:
    r"""
    Plots loss curves and saves the image.
    If `keys` is not provided, it automatically detects all keys containing "loss".
    """
    plt.switch_backend("agg")
    trainer_state_path = os.path.join(save_dictionary, TRAINER_STATE_NAME)
    if not os.path.exists(trainer_state_path):
        logger.warning_rank0(f"No trainer state file found at {trainer_state_path}")
        return

    with open(trainer_state_path, encoding="utf-8") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    if not log_history:
        logger.warning_rank0("No log history found in trainer state.")
        return

    # Auto-detect all keys that contain "loss" if none are provided
    if keys is None:
        all_keys = set()
        for log in log_history:
            all_keys.update(log.keys())
        keys = sorted(k for k in all_keys if "loss" in k)

    for key in keys:
        steps, metrics = [], []
        for log in log_history:
            if key in log:
                steps.append(log.get("step", log.get("current_steps", len(steps))))
                metrics.append(log[key])

        if not metrics:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, f"training_{key.replace('/', '_')}.png")
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)

