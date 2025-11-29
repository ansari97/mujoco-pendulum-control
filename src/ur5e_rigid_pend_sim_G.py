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
    q_init = np.array(
        [
            -2.07725076e-03,
            -4.96070088e-01,
            -5.48030467e-01,
            2.61150000e00,
            -1.68027287e00,
            1.57081598e00,
        ]
    )
    data.qpos[:] = q_init
    data.qvel[:] = 0

    # Disable for now
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # disable collisions
    model.dof_damping[:] = 0.0  # disable damping
    model.dof_armature[:] = 0  # disable motor inertia
    model.geom_friction[:] = 0  # disable friction

    # Settle physics
    mujoco.mj_forward(model, data)

    # define IDs
    ee_body_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    ee_site_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "UR_TCP")

    # Pre-allocation
    M = np.zeros((model.nv, model.nv))  # mass-matrix

    print("Starting simulation...")

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

            save_damping = model.dof_damping[:].copy()

            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            bias_torque = data.qfrc_bias.copy()

            ## restore joint velocity
            data.qvel[:] = dq.copy()

            data.ctrl[:] = bias_torque

            if step_count % 100 == 0:
                print("t: ", data.time)
                # print("S: ", S)
                print("Pos: ", q)
                print("Vel: ", dq)
                print("C+G: ", bias_torque)
                print("CTRL: ", data.ctrl)
                # print("R:", R)
                # print("R_err:", R_err)
                # print("y:", y)
                # print("dy:", dy)
                # print("y_norm:", np.linalg.norm(y))
                # print("h: ", h_total)
                # print("M: ", M)
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
