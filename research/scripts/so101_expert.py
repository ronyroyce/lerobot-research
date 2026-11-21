"""Scripted expert for the SO-101 MuJoCo tabletop: kinematics (this part) and pick-and-place planning.

Part 1 - inverse kinematics (IK) for a top-down grasp.

The arm has 5 joints that position the gripper (pan, lift, elbow, wrist_flex, wrist_roll) and one gripper
joint. A top-down grasp needs 6 numbers: tool position (x, y, z), approach axis pointing straight down
(2 numbers), and a yaw about that axis to line the jaws up with the cube's faces. 6 constraints on 5 joints
looks over-determined, but it is not: the arm is planar after the pan joint, so pointing the approach
axis straight down only costs one constraint (the in-plane pitch). Pan sets the azimuth, lift/elbow/flex
set reach, height and pitch, and roll sets the yaw. That is 5 for 5, so a solution exists inside the
workspace.

We solve it numerically with damped least squares (Levenberg-Marquardt) on MuJoCo's own Jacobian:

    e  = [p* - p ; orientation error]            (6,)   task-space error
    J  = d(p, orientation)/dq                    (6, 5) from mj_jac at the tool point
    dq = J^T (J J^T + lambda^2 I)^-1 e           damping keeps steps bounded near singularities

IK runs on a private MjData copy, so it never disturbs the live simulation.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
SITE = "gripperframe"
# Tool point in the gripperframe site's axes: x = approach (toward the tips), z = jaw closing axis.
# 0.8 cm back from the tips (part 2 measured: at 1.5 cm deep the fingertips hit the table and the tool stops
# 0.7 cm short, so 0.8 cm is the deepest physically reachable grasp). Along z the fixed jaw's tips are at -2.0 cm;
# a 3 cm cube centred at z=0 clears it by 0.5 cm on the way down, and closing slides it onto the fixed jaw.
# (Part 1 used z=-0.5 cm, which leaves 0 clearance on the fixed side: changed in part 2.)
TOOL_OFFSET = np.array([-0.008, 0.0, 0.0])
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class IKResult:
    q: np.ndarray  # arm joint angles [rad], ARM_JOINTS order
    pos_err: float  # [m]
    ang_err: float  # [rad]
    iters: int
    converged: bool
    heading: float = 0.0  # jaw closing-axis heading used [rad]


class SO101Kinematics:
    """Forward and inverse kinematics of the arm, on a private copy of the env's model state."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)  # private: IK must not touch the live sim
        self.site = model.site(SITE).id
        self.body = model.site_bodyid[self.site]
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo, self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
        self.gripper_qadr = model.joint("gripper").qposadr[0]
        g_lo, g_hi = model.jnt_range[model.joint("gripper").id]
        self.gripper_rad = lambda value: g_lo + value / 100 * (g_hi - g_lo)  # env units 0-100 -> rad
        # Collision geoms of the hand: everything that can touch a neighbouring cube during a grasp.
        hand = {model.body(b).id for b in ("wrist", "gripper", "moving_jaw_so101_v1", "camera_mount")}
        self.hand_geoms = [
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] in hand and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool point position (3,) and site rotation matrix (3, 3) in world, for arm angles q [rad]."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        # mj_jac reads the com-based frames (cdof, subtree_com) that mj_comPos computes; with
        # mj_kinematics alone the Jacobian is silently all zeros (found the hard way, see research_log.md).
        mujoco.mj_comPos(self.model, self.data)
        R = self.data.site_xmat[self.site].reshape(3, 3)
        return self.data.site_xpos[self.site] + R @ TOOL_OFFSET, R.copy()

    def ik_topdown(
        self,
        target: np.ndarray,
        yaw: float,
        q0: np.ndarray,
        iters: int = 200,
        damping: float = 1e-2,
        tol_pos: float = 5e-4,
        tol_ang: float = 1e-2,
    ) -> IKResult:
        """Joint angles putting the tool point at `target` [m], approach axis down, closing axis at `yaw` [rad].

        yaw is the world-frame heading of the jaw closing axis in the table plane.
        """
        want_x = DOWN  # approach axis
        want_z = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # closing axis
        q = np.clip(np.asarray(q0, dtype=float).copy(), self.lo, self.hi)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for k in range(1, iters + 1):
            p, R = self.fk(q)
            e_pos = target - p
            # Small-angle orientation error that rotates the current axes onto the wanted ones.
            e_ang = 0.5 * (np.cross(R[:, 0], want_x) + np.cross(R[:, 2], want_z))
            if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_ang) < tol_ang:
                return IKResult(q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_ang)), k, True, yaw)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, p, self.body)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])  # (6, 5)
            e = np.concatenate([e_pos, e_ang])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        e_ang = 0.5 * (np.cross(R[:, 0], want_x) + np.cross(R[:, 2], want_z))
        return IKResult(q, float(np.linalg.norm(target - p)), float(np.linalg.norm(e_ang)), iters, False, yaw)

    def ik_grasp(self, target: np.ndarray, cube_yaw: float, q0: np.ndarray) -> IKResult:
        """Top-down IK for a cube: a cube looks the same every 90 deg, so try the 4 jaw headings and keep
        the converged one that needs the least wrist roll."""
        return min(self.grasp_candidates(target, cube_yaw, q0), key=lambda r: (not r.converged, abs(r.q[4] - q0[4])))

    def grasp_candidates(self, target: np.ndarray, cube_yaw: float, q0: np.ndarray) -> list[IKResult]:
        """IK for each of the 4 equivalent jaw headings of a cube."""
        return [self.ik_topdown(target, cube_yaw + k * np.pi / 2, q0) for k in range(4)]

    # ---- collision clearance, measured on MuJoCo's real collision geometry ----
    def sync(self, live: mujoco.MjData) -> None:
        """Copy the live scene (cube poses etc.) into the private data used for planning checks."""
        self.data.qpos[:] = live.qpos

    def clearance(self, q_arm: np.ndarray, gripper_rad: float, robot_geoms, obstacle_geoms, cap: float = 0.05) -> float:
        """Smallest distance [m] between the robot geoms and the obstacle geoms at arm pose q_arm.
        mj_geomDistance works on the actual collision shapes (boxes, spheres, convex mesh hulls)."""
        self.data.qpos[self.qadr] = q_arm
        self.data.qpos[self.gripper_qadr] = gripper_rad
        mujoco.mj_kinematics(self.model, self.data)
        fromto = np.zeros(6)
        m = self.model
        # Only pairs that MuJoCo would actually let collide: (contype1 & conaffinity2) or the reverse.
        # mj_geomDistance ignores these bitmasks; without the filter, hand geoms that can never touch a cube
        # reported 0 mm "clearance" in episodes where nothing moved (4.2b).
        pairs = [
            (g1, g2) for g1 in robot_geoms for g2 in obstacle_geoms
            if (m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])
        ]
        return min((mujoco.mj_geomDistance(m, self.data, g1, g2, cap, fromto) for g1, g2 in pairs), default=cap)


def body_yaw(data: mujoco.MjData, body: int) -> float:
    """Heading of a body's x axis in the table plane [rad]."""
    R = data.xmat[body].reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def max_topdown_height(
    kin: SO101Kinematics, point: np.ndarray, yaw: float, q_grasp: np.ndarray, cap: float = 0.06, step: float = 0.005
) -> tuple[float, IKResult]:
    """Highest height above `point` (up to `cap`) still reachable top-down, and its IK solution.

    Measured in 4.2b: the limit is the wrist_flex joint (+-95 deg), not arm length. Staying vertical higher up
    needs more wrist bend, and the far cubes run out first (4 cm at r = 0.30 m, 9 cm at r = 0.18 m).
    Walks up from the grasp solution so each solve starts next to the previous one.
    """
    best_h, best = 0.0, None
    q = q_grasp
    for h in np.arange(step, cap + 1e-9, step):
        res = kin.ik_grasp(point + [0.0, 0.0, h], yaw, q)
        if not res.converged:
            break
        best_h, best, q = float(h), res, res.q
    return best_h, best


# ---------------------------------------------------------------------------------------------------------
# Part 2 - pick-and-place planning.
#
# A pick is a fixed sequence of segments. Each segment produces one action per env step (30 fps), in the
# env's units (degrees, gripper 0-100). Two kinds of motion:
#   - joint segments: minimum-jerk interpolation per joint, for free-space moves (fast, no IK per step)
#   - Cartesian segments: straight vertical line of the tool point, IK at every step (warm-started), for
#     descending onto / lifting off a cube, where a joint-space curve could swing the jaws into it.
# Minimum jerk s(t) = 10t^3 - 15t^4 + 6t^5 starts and ends with zero velocity and acceleration, which
# keeps the servos out of torque saturation (M3 step 3.3: steps saturate and ring, smooth targets don't).
# ---------------------------------------------------------------------------------------------------------

GRIPPER_OPEN = 35.0  # 5.4 cm jaw gap around a 3 cm cube (measured: 3 cm gap = gripper 18.5)
GRIPPER_CLOSE = 0.0  # the cube stops the jaw near 18.5, so the servo squeezes at its torque limit
BOX_SLOT = 0.022  # drop points at (+-2.2, +-2.2) cm in the box frame, so cubes land spread out
# Low-drop placement (part 2, three attempts measured):
#  1. lowering a held cube onto the box floor: the closed jaws stick out beside the cube and land on the
#     3.5 cm walls or on cubes already placed (only 0.8-1.4 cm of room) -> cubes end up stacked.
#  2. releasing above env.py's 6 cm "in box" height (cube centre 6.8 cm): far-side slots are out of top-down
#     reach (wrist_flex limit), and a 4 cm fall makes cubes tumble and lean on walls/each other.
#  3. (this) release with the cube's bottom 1 mm above the walls: the jaws (~0.7 cm above the cube's bottom)
#     clear walls and flat-placed cubes (top 3.4 cm), and a 1.7 cm fall lands the cube flat in its quadrant.
#     Known env limitation: the 4th cube is below 6 cm while still held, so env.py reports success (and
#     terminates) just before the last release. Candidate M5 change: success only for released cubes.
DROP_CUBE_Z = 0.051
MIN_CLEARANCE = 0.005  # hand-to-neighbour gap wanted during the descent [m]
DROP_CLEAR = 0.015  # approach from 1.5 cm above (tool at ~8.3 cm; top-down reach over the box ends near 10 cm)
# Where the cube's centre sits relative to the tool point, in gripperframe axes, once grasped (measured, stable
# from closing to release). -0.78 cm along the closing axis: the moving jaw slides the cube onto the fixed
# jaw. Along the approach axis it's ~0 now that TOOL_OFFSET is 0.8 cm deep. Used to put the cube's centre,
# not the tool point, over the slot.
CUBE_IN_GRIPPER = np.array([0.0007, -0.0001, -0.0078])  # re-measured: 24 grasps, std <= 0.04 mm


def min_jerk(n: int) -> np.ndarray:
    """n samples of the minimum-jerk profile on (0, 1], ending exactly at 1."""
    t = np.arange(1, n + 1) / n
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class PickPlaceExpert:
    """Puts the cubes in the box one by one. Each cube is planned when it starts, from the live sim state,
    then executed open loop. A cube's plan therefore sees where the previous cubes actually ended up."""

    def __init__(self, world, cube_bodies, box_body, fps: int = 30, speed: float = 1.0):
        self.world = world
        self.kin = SO101Kinematics(world.model)
        self.cubes = [world.model.body(c).id for c in cube_bodies]
        self.names = list(cube_bodies)
        self.box = world.model.body(box_body).id
        self.fps = fps
        self.speed = speed  # >1 is faster
        self.log: list[dict] = []  # one entry per cube: plan details, for analysis

    # ---- helpers ----
    def _n(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps / self.speed))

    def _act(self, q: np.ndarray, g: float) -> np.ndarray:
        return np.concatenate([np.rad2deg(q), [g]]).astype(np.float32)

    def _joint(self, q0, q1, g0, g1, seconds):
        for s in min_jerk(self._n(seconds)):
            yield self._act(q0 + (q1 - q0) * s, g0 + (g1 - g0) * s)

    def _cartesian(self, p0, p1, heading, q0, g, seconds):
        q = q0
        for s in min_jerk(self._n(seconds)):
            res = self.kin.ik_topdown(p0 + (p1 - p0) * s, heading, q)
            q = res.q  # keep the best effort even if a step didn't fully converge
            yield self._act(q, g)

    def _hold(self, q, g, seconds):
        for _ in range(self._n(seconds)):
            yield self._act(q, g)

    def slot(self, k: int) -> np.ndarray:
        """World position where cube number k's centre is released."""
        R = self.world.data.xmat[self.box].reshape(3, 3)
        sx, sy = (-1, 1, -1, 1)[k % 4] * BOX_SLOT, (-1, -1, 1, 1)[k % 4] * BOX_SLOT
        p = self.world.data.xpos[self.box] + R @ np.array([sx, sy, 0.0])
        p[2] = DROP_CUBE_Z
        return p

    @staticmethod
    def topdown_rotation(heading: float) -> np.ndarray:
        """gripperframe axes for a top-down pose: x = down, z = closing axis at `heading`, y = z cross x."""
        x, z = DOWN, np.array([np.cos(heading), np.sin(heading), 0.0])
        return np.column_stack([x, np.cross(z, x), z])

    # ---- one cube ----
    def box_centre(self) -> np.ndarray:
        """Drop point at the box centre: 3 cm clearance to every wall for a 3 cm cube."""
        p = self.world.data.xpos[self.box].copy()
        p[2] = DROP_CUBE_Z
        return p

    def pick_and_place(self, i: int, k: int | None = None):
        """Actions that move cube i to slot k, or to the box centre when k is None. Planned now, from the
        current sim state."""
        data = self.world.data
        q_now = data.qpos[self.kin.qadr].copy()
        g_now = float(self.world.joint_state()[5])
        cube = data.xpos[self.cubes[i]].copy()

        # Choose the jaw heading: of the 4 equivalent ones, those whose descent keeps the hand at least
        # MIN_CLEARANCE from every other cube (measured with mj_geomDistance), least wrist roll first; then
        # the rest by clearance. Fall through to the next heading if the rest of the plan is infeasible.
        # (4.2b: choosing by wrist roll alone bumped a neighbour in 3/40 episodes, always while descending.)
        self.kin.sync(data)
        obstacles = [g for j, c in enumerate(self.cubes) if j != i
                     for g in np.flatnonzero(self.world.model.geom_bodyid == c)]
        g_open = self.kin.gripper_rad(GRIPPER_OPEN)
        cands = []
        for res in self.kin.grasp_candidates(cube, body_yaw(data, self.cubes[i]), q_now):
            if not res.converged:
                continue
            h, pre = max_topdown_height(self.kin, cube, res.heading, res.q)
            if pre is None:
                continue
            # Clearance along the descent: pre-grasp, two intermediate poses, grasp.
            path = [pre.q] + [self.kin.ik_topdown(cube + [0, 0, f * h], res.heading, res.q).q for f in (0.66, 0.33)] + [res.q]
            self.kin.sync(data)
            clear = min(self.kin.clearance(q, g_open, self.kin.hand_geoms, obstacles) for q in path)
            cands.append(((clear < MIN_CLEARANCE, -clear if clear < MIN_CLEARANCE else abs(res.q[4] - q_now[4])), res, h, pre, clear))
        cands.sort(key=lambda c: c[0])

        drop_cube = self.box_centre() if k is None else self.slot(k)
        entry = {"cube": self.names[i], "slot": k, "cube_xyz": cube.round(4).tolist(),
                 "drop_xyz": drop_cube.round(4).tolist(), "headings_feasible": len(cands)}
        self.log.append(entry)
        plan = None
        for _, grasp, h_pre, pre, clear in cands:
            # Pick the jaw heading at the drop (least roll change), then move the tool so the cube's centre,
            # not the tool point, ends up over the drop point.
            first = self.kin.ik_grasp(drop_cube, grasp.heading, grasp.q)
            heading = first.heading
            drop_p = drop_cube - self.topdown_rotation(heading) @ CUBE_IN_GRIPPER
            drop = self.kin.ik_topdown(drop_p, heading, first.q)  # warm start next to the answer
            above_p = drop_p + [0, 0, DROP_CLEAR]
            above = self.kin.ik_topdown(above_p, heading, drop.q)
            if first.converged and drop.converged and above.converged:
                plan = (grasp, h_pre, pre, heading, drop_p, drop, above_p, above)
                entry.update(grasp_heading_deg=round(float(np.rad2deg(grasp.heading)), 1),
                             descent_clearance_mm=round(1e3 * clear, 1), pre_h=h_pre)
                break
        if plan is None:
            entry["skipped"] = True
            return
        grasp, h_pre, pre, heading, drop_p, drop, above_p, above = plan

        lift_p = cube + [0, 0, h_pre]
        yield from self._joint(q_now, pre.q, g_now, GRIPPER_OPEN, 1.5)  # over the cube, jaws open
        yield from self._cartesian(lift_p, cube, grasp.heading, pre.q, GRIPPER_OPEN, 1.0)  # straight down
        yield from self._joint(grasp.q, grasp.q, GRIPPER_OPEN, GRIPPER_CLOSE, 0.5)  # close
        yield from self._hold(grasp.q, GRIPPER_CLOSE, 0.3)  # let the grip force build
        yield from self._cartesian(cube, lift_p, grasp.heading, grasp.q, GRIPPER_CLOSE, 1.0)  # straight up
        yield from self._joint(pre.q, above.q, GRIPPER_CLOSE, GRIPPER_CLOSE, 1.5)  # over the slot
        yield from self._cartesian(above_p, drop_p, heading, above.q, GRIPPER_CLOSE, 0.5)  # to drop height
        yield from self._joint(drop.q, drop.q, GRIPPER_CLOSE, GRIPPER_OPEN, 0.4)  # release: cube drops
        yield from self._hold(drop.q, GRIPPER_OPEN, 0.3)  # let it land
        yield from self._cartesian(drop_p, above_p, heading, drop.q, GRIPPER_OPEN, 0.5)  # back up

    def actions(self, order=None):
        """All cubes, nearest to the box first (fewest carries over other cubes)."""
        if order is None:
            box_xy = self.world.data.xpos[self.box][:2]
            order = sorted(range(len(self.cubes)),
                           key=lambda i: np.linalg.norm(self.world.data.xpos[self.cubes[i]][:2] - box_xy))
        for k, i in enumerate(order):
            yield from self.pick_and_place(i, k)


# Same geometry as env.py's cubes_in_box (inner half-width 4.5 cm, below 6 cm), but per cube, plus "released":
# the env counts a cube that is still held, so for our episodes success also requires the jaws open and the
# cube at rest.
BOX_INNER_HALF = 0.045
IN_BOX_MAX_HEIGHT = 0.06


def cube_in_box(world, cube: int, box: int) -> bool:
    R = world.data.xmat[box].reshape(3, 3)
    local = (world.data.xpos[cube] - world.data.xpos[box]) @ R
    return bool(np.all(np.abs(local[:2]) < BOX_INNER_HALF) and local[2] < IN_BOX_MAX_HEIGHT)


def cube_placed(world, cube: int, box: int, gripper_open: float = 25.0, max_speed: float = 0.01) -> bool:
    """In the box, jaws open (not holding it) and resting (linear speed below 1 cm/s)."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(world.model, world.data, mujoco.mjtObj.mjOBJ_BODY, cube, vel, 0)
    return (
        cube_in_box(world, cube, box)
        and world.joint_state()[5] > gripper_open
        and float(np.linalg.norm(vel[3:])) < max_speed
    )
