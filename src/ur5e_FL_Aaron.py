import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco.viewer

import os


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
y_size = 6  # output dimension (pendulum orientation and end-effector position in 3D)
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
print("B")
print(B)

# Pre-allocation
M = np.zeros((model.nv, model.nv))  # mass-matrix

J_task = np.zeros((y_size, model.nv))  # task jacobian
J_task_prev = np.zeros((y_size, model.nv))  # for the first step

R = np.zeros(3)  # rotation matrix

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

jacr_rot = np.zeros((3, model.nv))
jacp_pos = np.zeros((3, model.nv))

# output error and derivatives
y = np.zeros(y_size)
dy = np.zeros_like(y)
ddy = np.zeros_like(y)

# Quaternion utilities
# Just a static value for now
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

# disable collisions
model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

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

    # 1. Check Total Mass
    print(f"Total Mass: {model.body_mass[pend_body_ID]}")
    # Expected: 0.3

    # 2. Check CoM Position (offset from the Body Frame origin)
    print(f"CoM Offset: {model.body_ipos[pend_body_ID]}")
    # Expected: [0. 0. 0.375]

    # 3. Check Inertia
    print(f"Inertia: {model.body_inertia[pend_body_ID]}")

    # Simulation loop
    while viewer.is_running():

        mujoco.mj_forward(model, data)
        mujoco.mj_rnePostConstraint(model, data)

        q = data.qpos[: model.nv].copy()
        dq = data.qvel[: model.nv].copy()

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

        # quat_des is not being updated in the loop, it's a static value
        # print("quat_des: ", quat_des)

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
        dy_traj_des = np.zeros_like(y_traj_des)
        ddy_traj_des = np.zeros_like(y_traj_des)

        R_flat = data.xmat[pend_body_ID]
        # change to quaternion
        mujoco.mju_mat2Quat(quat_curr, R_flat)

        if np.dot(quat_des, quat_curr) < 0:
            quat_curr = -quat_curr

        # print("quat_curr: ", quat_curr)

        pos_curr = data.xpos[ee_body_ID]
        y_pos = pos_curr - y_traj_des

        # y_ori = quat_curr - quat_des
        mujoco.mju_mulQuat(quat_err, quat_des, quat_inverse(quat_curr))
        y_ori = -2 * quat_err[1:]
        # y_ori = y_ori[1:]

        y = np.hstack((y_pos, y_ori))
        dy_des = np.hstack((dy_traj_des, np.zeros(3)))
        ddy_des = np.hstack((ddy_traj_des, np.zeros(3)))
        # print(y)

        # get site jacobian and calculate J_task
        mujoco.mj_jacBody(model, data, jacp, jacr_rot, pend_body_ID)
        # jacr_rot = jacr.copy()
        mujoco.mj_jacBody(model, data, jacp_pos, jacr, ee_body_ID)
        J_task = np.vstack((jacp_pos, jacr_rot))

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

        M_inv = np.linalg.inv(M)  # mass matrix inverse

        b = ddy + J_task @ M_inv @ h_total - dJ_task @ dq + ddy_des

        rank_J_task = np.linalg.matrix_rank(J_task)
        # print("rank_J_task")
        # print(rank_J_task)
        A = J_task @ M_inv @ B
        rank_A = np.linalg.matrix_rank(A)

        if rank_A < model.nu:
            print("A")
            print(A)
            print(f"A rank {rank_A} is < {model.nu}!")

        # regularize A@A^T
        A_T = np.transpose(A)
        G = A @ A_T

        # rank_G = np.linalg.matrix_rank(G)
        # print("rank_G")
        # print(rank_G)

        G_damped = G + lambda_**2 * np.eye(y_size)

        # rank_G_damped = np.linalg.matrix_rank(G_damped)
        # print("rank_G_damped")
        # print(rank_G_damped)

        G_inv = np.linalg.inv(G_damped)
        A_dagger = A_T @ G_inv  # right pseudo-inverse of A
        tau_task = A_dagger @ b
        # print(tau_task)
        # Total Torque
        tau_total = tau_task  # + (N @ tau_pos)
        # tau_total = J_task.T @ ddy
        # # --- Apply ---
        data.ctrl[:] = tau_total

        # get ee acc
        ee_cacc = np.zeros(6)
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_BODY, ee_body_ID, ee_cacc, 0
        )
        ee_cacc = ee_cacc[3:].copy()
        ee_cacc = ee_cacc + model.opt.gravity[:]

        if step_count % 100 == 0:
            print("***---***")
            print("q: ", data.qpos)
            print("dq: ", data.qvel)
            print("ddq: ", data.qacc)

            print("quat_curr: ", quat_curr)
            print("quat_des: ", quat_des)

            print("pos_curr: ", pos_curr)
            print("pos_des: ", y_traj_des)

            # print("lower_bound: ", lower_bound)
            # print("upper_bound: ", upper_bound)

            print("y_norm: ", np.linalg.norm(y))

            print("ee_acc: ", ee_cacc)

        # data.mocap_pos[0] = y_traj_des

        # Step the simulation
        # input()
        mujoco.mj_step(model, data)
        # input()
        t += dt
        step_count += 1
        # Sync the viewer
        viewer.sync()
