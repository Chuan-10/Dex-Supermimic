"""Original 20/50 Hz task recipes, independent of the reference repository."""

from .env_cfgs import wuji_hand2_reorient_env_cfg


def make_env_cfg(num_envs=1024, control_hz=50, play=False):
    if control_hz not in (20, 50):
        raise ValueError("control_hz must be 20 or 50")
    cfg = wuji_hand2_reorient_env_cfg(num_envs=num_envs, play=play)
    cfg.scene.num_envs = num_envs
    # Dex SuperMimic reconstructs current critic observations during imagination;
    # only the actor uses stacked observations.
    for term in cfg.observations["critic"].terms.values():
        if term.history_length > 1:
            term.history_length = 1
    # These four sensors cover all active rewards/metrics. The original three
    # extra force sensors have no consumers in this task.
    required = {
        "tip_object_contact",
        "palm_object_found",
        "distal_finger_object_found",
        "finger_collision",
    }
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors if sensor.name in required
    )
    if control_hz == 50:
        cfg.decimation = 2
        cfg.actions["joint_pos"].ema_alpha = 0.3
        cfg.rewards["action_rate"].weight = 0.0
        cfg.rewards["action_hf"].weight = -30.0
        cfg.rewards["hold_escalation"].weight = 4.694
        cfg.commands["reorient_command"].success_hold_steps = 13
        cfg.commands["reorient_command"].goal_switch_delay = 50
        if "adaptive_episode" in cfg.curriculum:
            cfg.curriculum["adaptive_episode"].params["target_steps"] = 2000
        cfg.rewards["cage_escape"].params.update(max_outside_steps=25, max_count=37.5)
        cfg.terminations["cage_drop"].params["max_outside_steps"] = 25
        cfg.observations["critic"].terms["cage_counter"].params["max_outside_steps"] = (
            25
        )
    return cfg
