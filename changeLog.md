# Changes  (12/9/2025) - ur5e_FL_Aaron
    Commented out sphere in ur5e_sphere.xml
    moved step to the bottom of the simulation
    changed J_task if block condition from dt to t
    removed h_total = h_total line because redundant
    changed initial conditions to pendulum up
    mujoco.mj_rnePostConstraint(model, data) to compute ee_cacc
    added camer setting
    changed xml to ur5e_pend_bob
    added data.qvel = np.zeros(model.nv)

# new (12/9/2025) - ur5e_pend_bob
    created file and added spherical geom

# new (12/9/2025) - ur5e_pend_bob_collision
    added contype, conaffinity and margin (0.02 or 2cm)
    commented out the second collision geom in wrist_2_link
    excluded contact pairs for adjacent link (wrist_3_link and pendulum are still active)

# new (12/9/2025) - ur5e_QP_collision
    enabled collision
    applied collision avoidance algorithm

