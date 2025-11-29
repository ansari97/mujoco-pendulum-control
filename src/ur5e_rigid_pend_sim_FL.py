import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --- Configuration ---
current_path = os.getcwd()
XML_PATH = current_path + "/assets/ur5e_rigid_pend.xml"


def skew2Vec(R):
    return np.array([R[2, 1], -R[2, 0], R[1, 0]])


def E(R):
    return 1 / 2 * (np.trace(R) * np.eye(3) - R)


def main():

    # Load Model
    if not os.path.exists(XML_PATH):
        print(f"Error: XML file not found at {os.path.abspath(XML_PATH)}")
        return

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except ValueError as e:
        print(f"Error: Could not load {XML_PATH}. {e}")
        return

    data = mujoco.MjData(model)

    # Reset and Initialize
    mujoco.mj_resetData(model, data)

    SIM_DURATION = 20.0
    TIMESTEP = model.opt.timestep

    # --- INITIALIZATION ---
    q_init = np.array([0, -1.63, 1.51, 1.6, -0.314, -1])
    n_joints = min(len(q_init), model.nq)
    data.qpos[:n_joints] = q_init[:n_joints]

    # Disable for now
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # disable collisions
    model.dof_damping[:] = 0.1  # disable damping
    model.dof_armature[:] = 0  # disable motor inertia
    model.geom_friction[:] = 0  # disable friction

    # Settle physics
    mujoco.mj_forward(model, data)

    # define IDs
    ee_body_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    ee_site_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "UR_TCP")

    # check IDs
    # for i in range(model.nu):
    #     dof_id, _ = model.actuator_trnid[i]
    #     joint_id = model.dof_jntid[dof_id]
    #     print(
    #         f"Actuator {i} → DOF {dof_id} → Joint {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)}"
    #     )

    # Define tasks space output here
    # R_des = np.eye(3)
    R_des = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    w_des = np.zeros(3)  # no angular velocity desired
    y_size = 3

    # --- Controller Parameters ---
    # Gains (Tuned for stability)
    lam = -200.0  # eigenvalues for Kp and Kd calculations
    kp_val = lam * lam
    kd_val = -(lam + lam)

    # assembles as Kp and Kd matrices
    Kp = kp_val * np.eye(y_size)
    Kd = kd_val * np.eye(y_size)

    # For null-space control
    # Posture Gains (Increased Damping to stop "dangling" wrist)
    kp_posture = 100.0
    kd_posture = 100.0  #

    q_des = np.array([0, -1.5708, 1.5708, 0, 1.5708, 0])
    # q_init[:6]  # Home position for null-space joint control

    # For mapping 6 torques to the 7 joints
    B = np.eye(model.nv, model.nu)
    # print(B)

    # Pre-allocation
    M = np.zeros((model.nv, model.nv))  # mass-matrix
    J_task = np.zeros((y_size, model.nv))  # for the first step
    J_task_prev = np.zeros((y_size, model.nv))  # for the first step
    R = np.zeros(y_size)  # rotation matrix
    jacp = np.zeros((y_size, model.nv))
    jacr = np.zeros((y_size, model.nv))

    # tolerance for SVD
    sig_tol = 1e-6
    warning_flag = False

    print("Starting simulation...")

    # save_grav = model.opt.gravity
    model.opt.gravity[:] = 0
    print(model.opt.gravity)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step_count = 0

        while viewer.is_running() and data.time < SIM_DURATION:
            step_start = time.time()
            mujoco.mj_forward(model, data)

            # --- Read State ---
            q = data.qpos[: model.nv].copy()
            dq = data.qvel[: model.nv].copy()

            # print(q)
            # print(dq)

            # --- Kinematics ---
            # get current rotation matrix
            R = data.xmat[ee_body_ID].reshape(3, 3)

            # get output (output error)
            R_err = R_des.T @ R
            y = 0.5 * skew2Vec(R_err - R_err.T)  # this is the error output

            # get site jacobian and calculate J_task
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_ID)
            J_task = E(R_err) @ R.T @ jacr

            dy = J_task @ dq  # y_dot is the angular velocity error (w_des is zero)

            ddy = -Kp @ y - Kd @ dy

            # calculate dJ_task
            dJ_task = (J_task - J_task_prev) / TIMESTEP  # finite difference

            if step_count == 0:
                dJ_task = np.zeros_like(J_task)

            J_task_prev = J_task.copy()  # for next step iter

            # --- Dynamics ---
            # Bring in the dynamics
            h_total = data.qfrc_bias.copy()  # includes joint damping
            mujoco.mj_fullM(model, M, data.qM)  # get mass matrix
            M_inv = np.linalg.inv(M)  # mass matrix inverse

            h_total = h_total

            b = ddy + J_task @ M_inv @ h_total - dJ_task @ dq
            A = J_task @ M_inv @ B

            # regularize A@A^T if near singularity by checking SVD
            A_T = np.transpose(A)
            G = A @ A_T

            U, S, Vh = np.linalg.svd(G)
            S_inv = np.zeros_like(S)

            for i, sigma in enumerate(S):
                if sigma > sig_tol:
                    S_inv[i] = 1.0 / sigma
                else:
                    S_inv[i] = 0.0
                    warning_flag = True

            S_inv = np.diag(S_inv)

            G_inv = np.transpose(Vh) @ S_inv @ np.transpose(U)

            A_dagger = A_T @ G_inv  # right pseudo-inverse of A
            tau_task = A_dagger @ b
            # print(tau_task)

            # --- Null Space Posture ---
            N = np.eye(6) - (A_dagger @ A)

            tau_pos = -kp_posture * (q[:6] - q_des) - kd_posture * dq[:6]

            # Total Torque
            tau_total = tau_task + (N @ tau_pos)
            # tau_total = J_task.T @ ddy

            # # --- Apply ---
            data.ctrl[:] = tau_total

            if warning_flag:
                print(step_count, warning_flag)
            warning_flag = False  # reset warning flag

            if step_count % 100 == 0:
                print("t: ", data.time)
                print("S: ", S)
                print("Pos: ", data.qpos)
                print("Vel: ", data.qvel)
                print("CTRL: ", data.ctrl)
                print("R:", R)
                print("R_err:", R_err)
                print("y:", y)
                print("dy:", dy)
                print("y_norm:", np.linalg.norm(y))
                print("h: ", h_total)
                print("M: ", M)
                # print("u_global: ", u_global)
                # print("dJ_task: ", dJ_task)

            mujoco.mj_step(model, data)
            viewer.sync()

            step_count += 1
            # # if step_count % 100 == 0:
            # #     # DEBUG: Check if pendulum joint (index 6) is moving
            # #     # If this stays exactly 0.000, the joint is physically locked
            # #     print(f"Time: {data.time:.2f} | Pendulum Angle: {q[6]:.4f}")

            # time_until_next_step = model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
