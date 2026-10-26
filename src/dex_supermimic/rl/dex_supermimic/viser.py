"""Dex overlays for the existing Viser viewer and its built-in scene tree.

The real scene uses mjlab's original materials. Predicted hand/object meshes
are separate, translucent light-blue nodes. No extra visibility GUI is added.
"""

from types import MethodType

import numpy as np
import torch
from mjlab.viewer import ViserPlayViewer

from .networks import WorldModel
from .observations import TaskObservations
from .sim import SimReader
from .state import STATE_DIM, unpack
from .rotations import mat_to_quat, mm, mv

PREDICTION_COLOR = (150, 210, 255)


def install_box_groups(scene):
    """Keep box labels that mjlab 1.6 otherwise discards when batching.

    Adapt only this viewer instance; the installed mjlab package is unchanged.
    Reuse its renderer/clear lifecycle, including selected-world tracking and
    debug visibility. Each group also retains its own opacity.
    """
    from mjlab.viewer.viser.scene import _BatchedPrimitive

    if getattr(scene, "_dex_grouped_boxes", False):
        return
    scene._dex_grouped_boxes = True
    original_add, original_sync, original_clear = scene.add_box, scene._sync_boxes, scene.clear
    labels = []
    primitives = {"boxes": scene._boxes}
    for name in ("cage", "target"):
        primitives[name] = _BatchedPrimitive(name, scene._boxes.mesh_factory)
        scene._all_primitives.append(primitives[name])

    def add_box(self, center, size, mat, color, label=None):
        count = len(self._queued_boxes)
        original_add(center, size, mat, color, label=label)
        if len(self._queued_boxes) > count:
            group = "cage" if (label or "").startswith("reorient_cage_") else (
                "target" if (label or "").startswith("goal_") else "boxes"
            )
            labels.append(group)

    def clear(self):
        original_clear()
        labels.clear()

    def sync_boxes(self):
        queued, boxes = self._queued_boxes, self._boxes
        # Also handle boxes queued before the adapter was installed.
        names = ["boxes"] * max(0, len(queued) - len(labels)) + labels
        try:
            for name, primitive in primitives.items():
                self._queued_boxes = [item for item, group in zip(queued, names) if group == name]
                self._boxes = primitive
                original_sync()
        finally:
            self._queued_boxes, self._boxes = queued, boxes

    scene.add_box = MethodType(add_box, scene)
    scene.clear = MethodType(clear, scene)
    scene._sync_boxes = MethodType(sync_boxes, scene)


class WorldModelTracker:
    """Action-conditioned open-loop predictions, with per-environment resets."""

    def __init__(self, env, checkpoint, horizon=8):
        self.env, self.horizon = env, horizon
        self.reader = SimReader(env)
        if list(checkpoint["joint_names"]) != list(self.reader.robot.joint_names):
            raise ValueError("WM checkpoint/environment joint order mismatch")
        if list(map(tuple, checkpoint["context_names"])) != self.reader.context_names:
            raise ValueError("WM checkpoint/environment simulator context mismatch")
        state = checkpoint["modules"]["wm"]
        width = state["norm.mean"].numel()
        if width != STATE_DIM + self.reader.xi.shape[-1] + 9 + 20:
            raise ValueError("Unsupported checkpoint WM input layout")
        self.wm = WorldModel(width - 20, checkpoint["contact_dim"],
                             tuple(checkpoint["config"]["wm_hidden"]), amp=False).to(env.device)
        self.wm.load_state_dict(state, strict=True)
        self.wm.eval().requires_grad_(False)
        self.observations = TaskObservations(self.reader)
        self.prediction = self.reader.state()
        self.age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.reader.attach_capture()

    @torch.no_grad()
    def before_step(self):
        self.reader.refresh_frame()
        self.prediction = torch.where((self.age == 0)[:, None], self.reader.state(), self.prediction)
        aux = self.observations.auxiliary()
        self.dynamics = self.observations.dynamics(self.prediction, self.reader.xi, aux[:, 12:21])

    @torch.no_grad()
    def after_step(self):
        target = self.env.action_manager.get_term("joint_pos").processed_action
        control = target - self.reader.robot.data.encoder_bias
        self.prediction, _ = self.wm(self.prediction, control, self.dynamics)
        if not torch.isfinite(self.prediction).all():
            raise RuntimeError("Nonfinite WM prediction; playback paused")
        self.age = (self.age + 1) % self.horizon

    @torch.no_grad()
    def reset(self, env_ids=None):
        self.reader.refresh_frame()
        ids = slice(None) if env_ids is None else env_ids
        self.prediction[ids] = self.reader.state()[ids]
        self.age[ids] = 0

    def close(self):
        self.reader.detach_capture()


class WorldModelMeshes:
    """Render selected environment under /world_model/{hand,object}."""

    def __init__(self, server, tracker, opacity=0.4):
        from mjviser.conversions import group_geoms_by_visual_compat, merge_geoms

        self.tracker, self.frames = tracker, {}
        reader, model = tracker.reader, tracker.env.sim.mj_model
        self.root = server.scene.add_frame("/world_model", show_axes=False)
        server.scene.add_frame("/world_model/hand", show_axes=False)
        server.scene.add_frame("/world_model/object", show_axes=False)
        bodies = []
        for local, bid in enumerate(reader.kin.body_ids):
            short = model.body(bid).name.rsplit("/", 1)[-1]
            if short == "right_palm_link" or short.startswith("right_finger"):
                bodies.append((bid, "hand", short, local))
        for bid in reader.obj.indexing.body_ids.cpu().tolist():
            bodies.append((bid, "object", model.body(bid).name.rsplit("/", 1)[-1], None))
        mesh_counts = {"hand": 0, "object": 0}
        for bid, group, name, local in bodies:
            path = f"/world_model/{group}/{name}"
            self.frames[(group, local)] = server.scene.add_frame(path, show_axes=False)
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                     and model.geom_group[g] < 3 and model.geom_rgba[g, 3] > 0]
            for i, ids in enumerate(group_geoms_by_visual_compat(model, geoms)):
                mesh = merge_geoms(model, ids)
                server.scene.add_mesh_simple(f"{path}/mesh_{i}", mesh.vertices, mesh.faces,
                                             color=PREDICTION_COLOR, opacity=opacity,
                                             side="double", cast_shadow=False)
                mesh_counts[group] += 1
        if not all(mesh_counts.values()):
            raise ValueError(f"Missing WM hand/object render geometry: {mesh_counts}")
        print(f"[WM] scene tree: world_model/hand ({mesh_counts['hand']} meshes), "
              f"world_model/object ({mesh_counts['object']} meshes); "
              f"light blue, opacity={opacity}, horizon={tracker.horizon}", flush=True)

    @torch.no_grad()
    def update(self, scene):
        tracker, reader = self.tracker, self.tracker.reader
        idx = scene.env_idx
        q, qd, object_p, object_r, _, _ = unpack(tracker.prediction)
        body_p, body_r, _, _ = reader.kin.forward(q, qd)
        body_p = reader.palm_pos[:, None] + mv(reader.palm_rot[:, None], body_p)
        body_r = mm(reader.palm_rot[:, None], body_r)
        object_p = reader.palm_pos + mv(reader.palm_rot, object_p)
        object_r = mm(reader.palm_rot, object_r)
        positions = body_p[idx].cpu().numpy()
        quaternions = mat_to_quat(body_r[idx]).cpu().numpy()
        offset = np.zeros(3)
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            offset = -tracker.env.sim.data.xpos[idx, scene._tracked_body_id].cpu().numpy()
        self.root.position = offset
        for (group, local), handle in self.frames.items():
            if group == "hand":
                handle.position, handle.wxyz = positions[local], quaternions[local]
            else:
                handle.position = object_p[idx].cpu().numpy()
                handle.wxyz = mat_to_quat(object_r[idx]).cpu().numpy()


class DexViserPlayViewer(ViserPlayViewer):
    def __init__(self, env, policy, checkpoint, wm_horizon=8, wm_opacity=0.4, **kwargs):
        super().__init__(env, policy, **kwargs)
        self.tracker = None
        self.wm_meshes = None
        self.wm_opacity = wm_opacity
        if checkpoint.get("format_version") == 3 and "wm" in checkpoint.get("modules", {}):
            self.tracker = WorldModelTracker(env, checkpoint, wm_horizon)
        else:
            print("[WM] This checkpoint has no compatible format-3 WM; playing actor only.", flush=True)

    def setup(self):
        super().setup()
        install_box_groups(self._scene)
        print("[Viser] separate scene nodes: debug/env_<n>/cage and debug/env_<n>/target", flush=True)
        if self.tracker is not None:
            self.wm_meshes = WorldModelMeshes(self._server, self.tracker, self.wm_opacity)

    def _execute_step(self):
        if self.tracker is None:
            return super()._execute_step()
        # Keep the base viewer's policy/exception handling. WM input is captured
        # immediately before its real step; actuator targets are read afterward.
        with torch.no_grad():
            self.tracker.before_step()
        ok = super()._execute_step()
        if ok:
            try:
                with torch.no_grad():
                    self.tracker.after_step()
                    done = self.env.unwrapped.reset_buf.nonzero(as_tuple=False).flatten()
                    if done.numel():
                        self.env.reset(env_ids=done)
                        self.tracker.reset(done)
            except Exception:
                import traceback
                self._last_error = traceback.format_exc()
                print(self._last_error, flush=True)
                self.pause()
                return False
        return ok

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        if self.wm_meshes is not None:
            with self._sim_lock, self._server.atomic():
                self.wm_meshes.update(self._scene)

    def reset_environment(self):
        super().reset_environment()
        if self.tracker is not None:
            self.tracker.reset()

    def _handle_gui_reset(self, all_envs):
        super()._handle_gui_reset(all_envs)
        if self.tracker is not None:
            ids = None if all_envs else [self._scene.env_idx]
            self.tracker.reset(ids)

    def close(self):
        try:
            super().close()
        finally:
            if self.tracker is not None:
                self.tracker.close()
