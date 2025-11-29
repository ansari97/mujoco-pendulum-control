import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# --- Configuration ---
current_path = os.getcwd()
XML_PATH = current_path + "/assets/ur5e.xml"


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

    SIM_DURATION = 15.0
    TIMESTEP = model.opt.timestep

    pend_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint"
    )
    pend_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum")
    pend_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pend_com_site")

    print(f"Pendulum Joint ID: {pend_joint_id}")
    print(f"Pendulum Body ID: {pend_body_id}")
    print(model.body_mass[pend_body_id])
    print(model.body_inertia[pend_body_id])

    # --- INITIALIZATION ---
    # "Elbow Up" pose to avoid singularities and hold the pendulum UP
    # q_init = [Pan, Lift, Elbow, Wrist1, Wrist2, Wrist3, Pendulum]
    # q_init = np.array([0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0])
    q_init = np.array([0, -1.5708, 1.5708, -1.5708, +1.5708, 0, 0])
    n_joints = min(len(q_init), model.nq)
    data.qpos[:n_joints] = q_init[:n_joints]

    # Disable for now
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # disable collisions
    model.dof_damping[:] = 0.0  # disable damping
    model.dof_armature[:] = 0  # disable motor inertia
    model.geom_friction[:] = 0  # disable friction

    # Settle physics
    mujoco.mj_forward(model, data)

    # Task Definitions (pendulum z-axis along the cylinder)
    u_local = np.array([0, 0, 1])
    Q = np.array([[1, 0, 0], [0, 1, 0]])  # Control X and Y
    # Q = np.eye(3)  # Control X and Y and Z

    y_orien_des = np.array([0, 0])
    # z_pend_com_des = (
    #     0.5  # data.site_xpos[pend_site_id][2]  # get z_pos of the pendulum com at start
    # )

    # print(z_pend_com_des)

    # y_des = np.concatenate(([z_pend_com_des], y_orien_des))
    y_des = y_orien_des

    # --- Controller Parameters ---
    # Gains (Tuned for stability)
    lam = -5.0  # eigenvalues for Kp and Kd calculations
    kp_val = lam * lam
    kd_val = -(lam + lam)

    # assembles as Kp and Kd matrices
    Kp = kp_val * np.eye(len(y_des))
    Kd = kd_val * np.eye(len(y_des))

    # For null-space control
    # Posture Gains (Increased Damping to stop "dangling" wrist)
    kp_posture = 10.0
    kd_posture = 10.0  #

    q_des = q_init[:6]  # Home position for null-space joint control

    # For mapping 6 torques to the 7 joints
    B = np.zeros((model.nv, model.nu))
    for i in range(model.nu):
        # each actuator i typically acts on a single dof index:
        dof_id = model.actuator_trnid[i][0]  # 0th element of (dof, something)
        B[dof_id, i] = 1.0
    # B[:-1, :] = np.eye(model.nu)
    print(B)

    # Pre-allocation for mass
    M = np.zeros((model.nv, model.nv))

    # tolerance for SVD
    sig_tol = 1e-5

    print("Starting simulation...")
    # print("g: ", model.opt.gravity)  # only for debugging

    J_task_prev = np.zeros((len(y_des), model.nv))  # for the first step

    warning_flag = False

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
            R_pend = data.xmat[pend_body_id].reshape(3, 3)
            u_global = R_pend @ u_local

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, pend_body_id)
            J_w = jacr  # looking at the angular velocity jacobian

            mujoco.mj_jacSite(model, data, jacp, jacr, pend_site_id)
            J_z = jacp[2:3, :]

            # Task Jacobian
            u_skew = np.array(
                [
                    [0, -u_global[2], u_global[1]],
                    [u_global[2], 0, -u_global[0]],
                    [-u_global[1], u_global[0], 0],
                ]
            )
            J_task_orien = -Q @ u_skew @ J_w

            # J_task = np.vstack((J_z, J_task_orien))
            J_task = J_task_orien

            J_task_dot = (J_task - J_task_prev) / TIMESTEP  # finite difference

            if step_count == 0:
                J_task_dot = np.zeros_like(J_task)

            J_task_prev = J_task.copy()  # for next step iter

            y_orien = Q @ u_global

            z_pend_com = data.site_xpos[pend_site_id][2]
            y_z_com = z_pend_com

            # y = np.concatenate(([y_z_com], y_orien)) - y_des
            y = y_orien - y_des
            dy = J_task @ dq
            ddy = -Kp @ y - Kd @ dy

            h_total = data.qfrc_bias.copy()
            mujoco.mj_fullM(model, M, data.qM)
            # print(M)
            M_inv = np.linalg.inv(M)

            b = ddy + J_task @ M_inv @ h_total - J_task_dot @ dq
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

            # print("SVD")
            # print(U)
            # print(S)
            # print(Vh)

            # --- Dynamics ---
            A_dagger = A_T @ G_inv
            tau_task = A_dagger @ b
            # print(tau_task)

            # --- Null Space Posture ---
            N = np.eye(6) - (A_dagger @ A)

            tau_pos = -kp_posture * (q[:6] - q_des) - kd_posture * dq[:6]

            # Total Torque
            tau_total = tau_task + (N @ tau_pos)

            # # --- Apply ---
            data.ctrl[:] = tau_total

            if warning_flag:
                print(step_count, warning_flag)
            warning_flag = False

            if step_count % 100 == 0:
                print("t: ", data.time)
                print("S: ", S)
                print("Pos: ", data.qpos)
                print("Vel: ", data.qvel)
                print("CTRL: ", data.ctrl)
                print("y:", y)  # should go toward [0, 0]
                print("dy:", dy)
                print("||y||:", np.linalg.norm(y))
                print("u_global: ", u_global)
                print("J_task_dot: ", J_task_dot)

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
