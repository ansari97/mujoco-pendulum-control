import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco.viewer

import os

from qpsolvers import solve_qp


def quat_inverse(quat):
    quat_inverse = quat.copy()
    quat_inverse *= -1
    quat_inverse[0] = quat[0]
    return quat_inverse


def getEulerFromQuat(quat):
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    eul = r.as_euler("xyz", degrees=False)
    return eul


# Load the UR5e model
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = file_path + "/assets/ur5e/ur5e_pend_bob.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Parameters
y_size = 6  # output dimension (end-effector position in 3D)
ee_body_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
pend_body_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
# Trajectory parameters
radius = 0.2  # radius of the circle
omega = 0.1  # angular velocity
constant = 0.4  # constant x position
t = 0.0  # initial time
dt = 0.001  # time step

# --- Controller Parameters ---
# Gains (Tuned for stability)
lam = -5  # eigenvalues for Kp and Kd calculations; no imaginary part
kp_val = lam * lam
kd_val = -(lam + lam)
o_g = 50  # desired orientation gain
lambda_ = 10e-4  # damping for pseudo-inverse regularization
# assembles as Kp and Kd matrices
Kp = np.diag([kp_val, kp_val, kp_val, o_g * kp_val, o_g * kp_val, o_g * kp_val])
Kd = np.diag([kd_val, kd_val, kd_val, o_g * kd_val, o_g * kd_val, o_g * kd_val])
# For mapping 6 torques to the 7 joints
B = np.zeros((model.nv, model.nu))
B[: model.nu, :] = np.eye(model.nu)
print(B)
# input()

# Pre-allocation
M = np.zeros((model.nv, model.nv))  # mass-matrix
J_task = np.zeros((y_size, model.nv))  # for the first step
J_task_prev = np.zeros((y_size, model.nv))  # for the first step
R = np.zeros(3)  # rotation matrix
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
y = np.zeros(y_size)
ddy = np.zeros_like(y)

# Quaternion utilities
theta = np.deg2rad(0)
R_des = np.array(
    [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
)
quat_des = Rotation.from_matrix(R_des)
quat_des = quat_des.as_quat(scalar_first=True)

quat_curr = np.zeros_like(quat_des)
quat_err = np.zeros_like(quat_des)

model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # disable collisions

# QP weights
w_balance = 10  # Priority 1: Balance Pendulum
w_pos = 10.0  # Priority 2: Keep Wrist near Home
w_reg = 0.0001  # Regularize Torques (Minimize Effort)
w_posture = 0.1  # Regularize Accelerations (Stop twitching)

# joint limits (+-/both sides)
qpos_lim = 2 * np.pi * np.ones(model.nv)  # including the pendulum
qpos_lim[2] = np.pi  # smaller limit for the elbow
qpos_lim[6] = np.inf  # inf limit for the pendulum

qvel_lim = np.pi * np.ones(model.nv)  # pi/s
qvel_lim[6] = np.inf  # limit for the pendulum

beta = 1000  # decelerating torque max rad/s2^s

# soft wall limit gains
# kp_lim = 100
# kv_lim = 2 * np.sqrt(kp_lim)

TIMESTEP = model.opt.timestep

step_count = 0


with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set initial joint positions (home position)
    data.qpos = [-np.pi, -np.pi / 2, 0, -np.pi, -np.pi / 2, 0, 0.20]
    data.qvel = np.zeros(model.nv)
    # data.mocap_pos[0] = [1.0, 0.0, 1.0]  # Initial position of the mocap sphere

    # print(data.xpos[pend_body_ID])
    # print(data.xpos[ee_body_ID])

    print("model gravity")
    print(model.opt.gravity)
    print("model.nv")
    print(model.nv)
    print("model.nq")
    print(model.nq)

    print("model.nu")
    print(model.nu)

    step_count = 0

    # --- SET CAMERA ANGLE ONCE ---
    viewer.cam.azimuth = 90  # Azimuth angle (degrees)
    viewer.cam.elevation = -45  # Elevation angle (degrees)
    viewer.cam.distance = 5.0  # Distance from the lookat point (meters)
    viewer.cam.lookat[:] = [0, 0, 0.5]  # Point the camera is looking at [x, y, z]

    # Apply the changes
    mujoco.mj_forward(model, data)
    viewer.sync()

    # Simulation loop
    while viewer.is_running():
        # Step the simulation
        mujoco.mj_forward(model, data)

        q = data.qpos[: model.nv].copy()
        dq = data.qvel[: model.nv].copy()
        ddq = data.qacc.copy()

        # quat_des needss be updated every step if changing orientation
        theta = data.xquat[pend_body_ID]
        theta = getEulerFromQuat(theta)[2]
        R_des = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        # quat_des = Rotation.from_matrix(R_des)
        # quat_des = quat_des.as_quat(scalar_first=True)

        # quat_curr = np.zeros_like(quat_des)
        # quat_err = np.zeros_like(quat_des)

        t = data.time

        y_traj_des = np.array(
            [
                0.103,
                -0.1,
                0.5,
            ]
        )
        dy_des = np.zeros_like(y_traj_des)
        ddy_des = np.zeros_like(y_traj_des)

        # y_traj_des = np.array([0.49, 0.09, 0.6])

        # dy_des = np.array([0.0, 0.0, 0.0])
        # ddy_des = np.array([0.0, 0.0, 0.0])

        R_flat = data.xmat[pend_body_ID]
        # change to quaternion
        mujoco.mju_mat2Quat(quat_curr, R_flat)
        if np.dot(quat_des, quat_curr) < 0:
            quat_curr = -quat_curr

        pos_curr = data.xpos[ee_body_ID]
        y_pos = pos_curr - y_traj_des
        # y_ori = quat_curr - quat_des
        mujoco.mju_mulQuat(quat_err, quat_des, quat_inverse(quat_curr))
        y_ori = -2 * quat_err[1:]
        # y_ori = y_ori[1:]

        y = np.hstack((y_pos, y_ori))
        dy_des = np.hstack((dy_des, np.zeros(3)))
        ddy_des = np.hstack((ddy_des, np.zeros(3)))
        # print(y)

        # get site jacobian and calculate J_task
        mujoco.mj_jacBody(model, data, jacp, jacr, pend_body_ID)
        jacr_t = jacr.copy()
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_ID)
        J_task = np.vstack((jacp, jacr_t))

        dy = J_task @ dq - dy_des  # y_dot is the angular velocity error (w_des is zero)
        ddy = -Kp @ y - Kd @ dy
        # ddy[:3] = Kp[:3, :3] @ y[:3] - Kd[:3, :3] @ dy[:3]
        # calculate dJ_task

        dJ_task = (J_task - J_task_prev) / dt  # finite difference
        if t == 0:
            dJ_task = np.zeros_like(J_task)
        J_task_prev = J_task.copy()  # for next step iter

        # --- Dynamics ---
        # Bring in the dynamics
        h_total = data.qfrc_bias.copy()  # includes joint damping
        mujoco.mj_fullM(model, M, data.qM)  # get mass matrix

        # M_inv = np.linalg.inv(M)  # mass matrix inverse
        # h_total = h_total
        # b = ddy + J_task @ M_inv @ h_total - dJ_task @ dq + ddy_des
        # A = J_task @ M_inv @ B
        # # regularize A@A^T if near singularity by checking SVD
        # A_T = np.transpose(A)
        # G = A @ A_T
        # G_damped = G + lambda_**2 * np.eye(y_size)
        # G_inv = np.linalg.inv(G_damped)
        # A_dagger = A_T @ G_inv  # right pseudo-inverse of A
        # tau_task = A_dagger @ b
        # # print(tau_task)
        # # Total Torque
        # tau_total = tau_task  # + (N @ tau_pos)
        # tau_total = J_task.T @ ddy
        # # --- Apply ---

        ## QP Formulation

        k = ddy - dJ_task @ dq + ddy_des
        dim_q = model.nv
        dim_tau = model.nu
        dim_x = dim_q + dim_tau  # 13 variables
        P = np.zeros((dim_x, dim_x))
        q_vec = np.zeros(dim_x)

        P[:dim_q, :dim_q] = J_task.T @ J_task
        P[dim_q:, dim_q:] = w_reg * np.eye(dim_tau)
        q_vec[:dim_q] = -J_task.T @ k

        A_eq = np.hstack([M, -B])
        b_eq = -h_total  # RHS

        # ddq_lim = 10
        # tau_lim = np.concatenate((150 * np.ones(3), 28 * np.ones(3)))
        tau_lim = np.inf * np.ones(model.nu)

        # qacc_limit_upper = 2 / TIMESTEP**2 * (qpos_lim - q - dq * TIMESTEP)
        # qacc_limit_lower = 2 / TIMESTEP**2 * (-qpos_lim - q - dq * TIMESTEP)
        # print(qacc_limit_upper)
        # print(qacc_limit_lower)

        ## Get safe acceleration bounds based on the invariance method
        # predict next joint positions
        # q_next = q + dq * TIMESTEP

        # # predict available distance from the limits
        # d_upper = np.maximum(np.zeros(model.nv), qpos_lim - q_next)
        # d_lower = np.maximum(np.zeros(model.nv), q_next - -(qpos_lim))

        # # get safe upper and upper lower bounds for velocity based on available distance in the next predicted step
        # qvel_safe_upper = np.sqrt(2 * beta * d_upper)
        # qvel_safe_lower = -np.sqrt(2 * beta * d_lower)

        # # get bounds from limits and safe
        # qvel_upper_bound = np.minimum(qvel_lim, qvel_safe_upper)
        # qvel_lower_bound = np.maximum(-qvel_lim, qvel_safe_lower)

        # # calculate safe bounds for acc using limits for vel and current vel
        # qacc_upper_bound = (qvel_upper_bound - dq) / TIMESTEP
        # qacc_lower_bound = (qvel_lower_bound - dq) / TIMESTEP

        # print(d_upper)
        # input()

        qacc_upper_bound = np.inf * np.ones(model.nv)
        qacc_lower_bound = -np.inf * np.ones(model.nv)

        # qacc_lim = np.ones(model.nv) * np.inf

        lower_bound = np.concatenate((qacc_lower_bound, -tau_lim))
        upper_bound = np.concatenate((qacc_upper_bound, tau_lim))

        x_sol = solve_qp(
            P,
            q_vec.flatten(),
            None,
            None,
            A_eq,
            b_eq.flatten(),
            lb=lower_bound.flatten(),
            ub=upper_bound.flatten(),
            solver="osqp",
        )

        # # --- Apply ---
        if x_sol is not None:
            # Extract Torques (Last 6 elements)
            tau_cmd = x_sol[dim_q:]
            data.ctrl[:] = tau_cmd
        else:
            print("QP Failed! Sending Zero Torque.")
            data.ctrl[:] = 0

        # data.ctrl[:] = tau_total

        if step_count % 100 == 0:
            print("***---***")
            print("q: ", data.qpos)
            print("dq: ", data.qvel)
            print("ddq: ", data.qacc)

            print("quat_curr: ", quat_curr)
            print("quat_des: ", quat_des)

            print("pos_curr: ", pos_curr)
            print("pos_des: ", y_traj_des)

            print("lower_bound: ", lower_bound)
            print("upper_bound: ", upper_bound)

            print("y_norm: ", np.linalg.norm(y))

        # data.mocap_pos[0] = y_traj_des
        mujoco.mj_step(model, data)
        # input()
        t += dt
        step_count += 1
        # Sync the viewer
        viewer.sync()
