import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco.viewer

def getEulerFromQuat(quat):
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    eul = r.as_euler('xyz', degrees=False)
    return eul

# Load the UR5e model
model = mujoco.MjModel.from_xml_path("C:\\Python\\mujoco\\assets\\ur5e_sphere.xml")
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
lam = -1  # eigenvalues for Kp and Kd calculations; no imaginary part
kp_val = lam * lam
kd_val = -(lam + lam)
o_g = 50 # desired orientation gain
lambda_ = 0.001  # damping for pseudo-inverse regularization
# assembles as Kp and Kd matrices
Kp = np.diag([kp_val, kp_val, kp_val, o_g*kp_val, o_g*kp_val, o_g*kp_val])
Kd = np.diag([kd_val, kd_val, kd_val, o_g*kd_val, o_g*kd_val, o_g*kd_val])
# For mapping 6 torques to the 7 joints
B = np.zeros((model.nv, model.nu))
B[: model.nu, :] = np.eye(model.nu)

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
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
quat_des = Rotation.from_matrix(R_des)
quat_des = quat_des.as_quat(scalar_first=True)
quat_curr = np.zeros_like(quat_des)
quat_err = np.zeros_like(quat_des)

model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # disable collisions

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set initial joint positions (home position)
    data.qpos[:6] = [-np.pi, -np.pi/2, 0, np.pi/2, -np.pi/2, 0]
    #data.mocap_pos[0] = [1.0, 0.0, 1.0]  # Initial position of the mocap sphere

    print(data.xpos[pend_body_ID])
    print(data.xpos[ee_body_ID])

    # Simulation loop
    while viewer.is_running():
        # Step the simulation
        mujoco.mj_step(model, data)

        q = data.qpos[: model.nv].copy()
        dq = data.qvel[: model.nv].copy()

        #quat_des needss be updated every step if changing orientation
        theta = data.xquat[pend_body_ID]
        theta = getEulerFromQuat(theta)[2]
        R_des = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
        )
        quat_des = Rotation.from_matrix(R_des)
        quat_des = quat_des.as_quat(scalar_first=True)
        quat_curr = np.zeros_like(quat_des)
        quat_err = np.zeros_like(quat_des)
        
        t = data.time

        y_traj_des = radius * np.array(
            [
                constant/radius,
                np.sin(omega * t),
                np.cos(omega * t) + 3,
            ]
        )
        dy_des = (  radius * omega * np.array(
                [
                    0,
                    np.cos(omega * t),
                    -np.sin(omega * t),
                ]
            )
        )
        ddy_des = (radius * omega**2 * np.array(
                [
                    0,
                    -np.sin(omega * t),
                    -np.cos(omega * t),
                ]
            )
        )

        # y_traj_des = np.array([0.49, 0.09, 0.6])

        # dy_des = np.array([0.0, 0.0, 0.0])
        # ddy_des = np.array([0.0, 0.0, 0.0])

        R_flat = data.xmat[pend_body_ID]
        # change to quaternion
        mujoco.mju_mat2Quat(quat_curr, R_flat)
        if np.dot(quat_des, quat_curr) < 0:
            quat_curr = -quat_curr
        print(quat_curr)
        pos_curr = data.xpos[ee_body_ID]
        y_pos = pos_curr - y_traj_des
        y_ori = quat_curr - quat_des
        y_ori = y_ori[1:]
        y = np.hstack((y_pos, y_ori))
        dy_des = np.hstack((dy_des, np.zeros(3)))
        ddy_des = np.hstack((ddy_des, np.zeros(3)))
        # print(y)
        # get site jacobian and calculate J_task
        mujoco.mj_jacBody(model, data, jacp, jacr, pend_body_ID)
        jacr_t = jacr.copy()
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_ID)
        J_task = np.vstack((jacp, jacr_t))
        dy = (
            J_task @ dq - dy_des
        )  # y_dot is the angular velocity error (w_des is zero)
        ddy = -Kp @ y - Kd @ dy
        #ddy[:3] = Kp[:3, :3] @ y[:3] - Kd[:3, :3] @ dy[:3]
        # calculate dJ_task
        dJ_task = (J_task - J_task_prev) / dt  # finite difference
        if dt == 0:
            dJ_task = np.zeros_like(J_task)
        J_task_prev = J_task.copy()  # for next step iter
        # --- Dynamics ---
        # Bring in the dynamics
        h_total = data.qfrc_bias.copy()  # includes joint damping
        mujoco.mj_fullM(model, M, data.qM)  # get mass matrix
        M_inv = np.linalg.inv(M)  # mass matrix inverse
        h_total = h_total
        b = ddy + J_task @ M_inv @ h_total - dJ_task @ dq + ddy_des
        A = J_task @ M_inv @ B
        # regularize A@A^T if near singularity by checking SVD
        A_T = np.transpose(A)
        G = A @ A_T
        G_damped = G + lambda_**2 * np.eye(y_size)
        G_inv = np.linalg.inv(G_damped)
        A_dagger = A_T @ G_inv  # right pseudo-inverse of A
        tau_task = A_dagger @ b
        # print(tau_task)
        # Total Torque
        tau_total = tau_task  # + (N @ tau_pos)
        # tau_total = J_task.T @ ddy
        # # --- Apply ---
        data.ctrl[:] = tau_total
        #print(np.linalg.norm(y))

        #data.mocap_pos[0] = y_traj_des
        t += dt
        # Sync the viewer
        viewer.sync()

