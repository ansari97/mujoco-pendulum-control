// Feedback Linearization code

#include <iostream>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Eigen/Dense> // for matrix algebra

// Define a shorthand for Row-Major matrices (Matches MuJoCo layout)
template <int R, int C>
using MatrixRowMaj = Eigen::Matrix<double, R, C, Eigen::RowMajor>;

// --- Global Variables ---
mjModel *m = nullptr;
mjData *d = nullptr;
mjvCamera cam;
mjvScene scn;
mjrContext con;
mjvOption vopt;
mjvPerturb pert;
GLFWwindow *window = nullptr;

// --- Mouse and Keyboard State ---
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// --- Callbacks Declarations ---
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow *window, int button, int act, int mods);
void mouse_move(GLFWwindow *window, double xpos, double ypos);
void scroll(GLFWwindow *window, double xoffset, double yoffset);

// --- Global helper needed for camera control
char xmlpath[200] = {0}; // Stores the path to the XML file

// Main function
int main(int argc, const char **argv)
{
    // 1. Check for the XML file path; needs to be passed as cmd line input
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path/to/ur5e.xml>" << std::endl;
        return 1;
    }

    // 2. Load the model from the MJCF file
    char error[1000] = "Could not load XML model";
    m = mj_loadXML(argv[1], 0, error, 1000);
    if (!m)
    {
        std::cerr << "Error loading model: " << error << std::endl;
        return 1;
    }

    // 3. Create the data structure
    d = mj_makeData(m);
    mj_resetData(m, d);

    // Getting key_qpos from xml
    if (m->nkey > 0)
    {
        mju_copy(d->qpos, m->key_qpos, m->nq);
        mj_forward(m, d);
    }

    // Initialize GLFW library
    if (!glfwInit())
    {
        std::cerr << "Could not initialize GLFW" << std::endl;
        return 1;
    }

    // Create the GLFW window
    window = glfwCreateWindow(1200, 900, "UR5e MuJoCo Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "Could not create GLFW window" << std::endl;
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-sync

    // Initialize MuJoCo visualization structures
    mjv_defaultCamera(&cam);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Initialize visualization options (vopt) and perturbation (pert)
    mjv_defaultOption(&vopt);
    vopt.frame = mjFRAME_BODY;
    // vopt.frame = mjFRAME_;
    mjv_defaultPerturb(&pert);

    // Create the scene and context based on the model
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Set initial camera position for a good view (e.g., side view)
    double lookat_point[3] = {0, 0, 0};
    cam.lookat[0] = lookat_point[0];
    cam.lookat[1] = lookat_point[1];
    cam.lookat[2] = lookat_point[2];

    cam.azimuth = 225;   // Horizontal rotation
    cam.elevation = -30; // Vertical rotation
    cam.distance = 4;    // Distance from the target

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-sync

    // --- NEW: Set up callbacks ---
    glfwSetKeyCallback(window, keyboard);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);

    // Store XML path globally for potential reloading in keyboard callback
    std::strcpy(xmlpath, argv[1]);

    // disable collisions
    // m->opt.disableflags |= mjDSBL_CONTACT;

    // Simulation Loop parameters
    double sim_duration = 15.0;         // seconds
    double time_step = m->opt.timestep; // Time step defined in the XML (e.g., 0.002s)
    int step_count = 0;

    std::cout << "Starting UR5e zero-input simulation for " << sim_duration << "s..." << std::endl;
    std::cout << "Time step: " << time_step << "s" << std::endl;

    // --- Initial calculations ---
    // mjtNum is "double"
    // Eigenvalues; all real and negative, no imaginary parts
    mjtNum des_eig_1 = -10;
    mjtNum des_eig_2 = -10;

    mjtNum des_eig_3 = -10;
    mjtNum des_eig_4 = -10;

    // for Kp and Kd matrices
    mjtNum k_p_1 = des_eig_1 * des_eig_2;
    mjtNum k_p_2 = des_eig_3 * des_eig_4;

    mjtNum k_d_1 = -(des_eig_1 + des_eig_2);
    mjtNum k_d_2 = -(des_eig_3 + des_eig_4);

    // in row-major form
    mjtNum Kp[4] = {k_p_1, 0, 0, k_p_2};
    mjtNum Kd[4] = {k_d_1, 0, 0, k_d_2};

    // --- Variables ---
    // states
    mjtNum q_current[7];
    mjtNum dq_current[7];

    // pendulum variables
    int pend_body_id = mj_name2id(m, mjOBJ_BODY, "pendulum"); // stores pendulum_body_ID
    mjtNum *R_pend;                                           // for storing the rotation matrix wrt global frame
    mjtNum J_p_pend[3 * m->nv];                               // positon/linear velocity jacobian
    mjtNum J_w_pend[3 * m->nv];                               // angular velocity jacobian

    mjtNum J_task_pend[2 * m->nv]; // task jacobian

    mjtNum Q[2 * 3] = {1, 0, 0, 0, 1, 0}; // take only x and y oordinates
    mjtNum u_local[3] = {0, 1, 0};        // along the pendulum frame y-axis

    mjtNum gravity[3];
    mju_copy(gravity, m->opt.gravity, 3);

    std::cout << "Gravity" << gravity[0] << ", " << gravity[1] << ", " << gravity[2] << std::endl;

    mjtNum A[2 * (m->nq - 1)];
    mjtNum y[2 * 1];
    mjtNum dy[2 * 1];
    mjtNum nu[2 * 1];
    mjtNum b[2 * 1];

    while (d->time < sim_duration)
    {
        // update all values
        mj_forward(m, d);

        // get state vector (q and dq)
        mju_copy(q_current, d->qpos, m->nq);
        mju_copy(dq_current, d->qvel, m->nq);

        // print out states
        // std::cout << "\nStates" << std::endl;
        // std::cout << "Pos: ";

        // for (int i = 0; i < m->nq; ++i)
        // {
        //     std::cout << q_current[i] << ", ";
        // }

        // std::cout << "\nVel: ";

        // for (int i = 0; i < m->nq; ++i)
        // {
        //     std::cout << dq_current[i] << ", ";
        // }

        // Get rotation matrix for pendulum frame
        R_pend = &d->xmat[pend_body_id * 9];

        // std::cout << "Rotation Matrix of pendulum\n";
        // std::cout << R_pend[0] << " " << R_pend[1] << " " << R_pend[2] << "\n";
        // std::cout << R_pend[3] << " " << R_pend[4] << " " << R_pend[5] << "\n";
        // std::cout << R_pend[6] << " " << R_pend[7] << " " << R_pend[8] << "\n";

        // Get angular velocity Jacobian for pendulum body frame
        mj_jacBody(m, d, J_p_pend, J_w_pend, pend_body_id);

        // Evaluate J_task
        mjtNum u_global[3];
        mju_mulMatVec3(u_global, R_pend, u_local);
        mjtNum skew_Ru[9] = {0, -u_global[2], u_global[1], u_global[2], 0, -u_global[0], -u_global[1], u_global[0], 0}; // skew-symmetric matrix

        mjtNum temp_mat[6];
        mju_mulMatMat(temp_mat, Q, skew_Ru, 2, 3, 3);                // calculate Q*(R*u)_tilde
        mju_mulMatMat(J_task_pend, temp_mat, J_w_pend, 2, 3, m->nv); // calculate Q*(R*u)_tilde*J_w_task

        mju_scl(J_task_pend, J_task_pend, -1, 2 * m->nv); // calculate -Q*(R*u)_tilde*J_w_task; multiply by -1

        // calculate dJ_task by proxy
        // set gravity to 0
        mju_zero(m->opt.gravity, 3);

        // change q_acc to zero
        mju_zero(d->qacc, m->nv);

        // this step caculates the torque required to produce the desired q_acc (0 in our case)
        mj_rne(m, d, 0, d->qfrc_inverse);

        // get com accelerations (both angular and linear)
        mjtNum alpha_bias[3];
        mju_copy(alpha_bias, &d->cacc[pend_body_id * 6], 3);

        // get omega; also possible to do this using J_w_pend*qvel
        mjtNum omega[3];
        mju_copy(omega, &d->cvel[pend_body_id * 6], 3);

        // start matrix multiplication and cross-products
        mjtNum temp_vec[3];
        mjtNum temp_vec2[3];
        mjtNum temp_vec3[3];
        mjtNum temp_vec_res[3];
        mjtNum eta[2];                         // J_task_dot*q_dot
        mju_cross(temp_vec, omega, u_global);  // w cross u_global
        mju_cross(temp_vec2, omega, temp_vec); // w cross (w cross u_global)

        mju_cross(temp_vec3, alpha_bias, u_global); // alpha_bias cross u_global
        mju_add(temp_vec_res, temp_vec2, temp_vec3, 3);
        mju_mulMatVec(eta, Q, temp_vec_res, 2, 3);

        // restore gravity
        mju_copy(m->opt.gravity, gravity, 3);

        // begin controller formulation
        // define B; the matrix that multiplies with the atuated 6x1 torques to give joint accel
        mjtNum B[m->nv * (m->nv - 1)];
        mju_zero(B, m->nv * (m->nv - 1)); // zero all entries
        for (int i = 0; i < m->nv - 1; i++)
        {
            B[i * m->nv] = 1.0;
        }

        // get M_inv
        mjtNum M_inv[m->nv * m->nv];
        mju_zero(M_inv, m->nv * m->nv);

        mjtNum eye[m->nv * m->nv] = {0};
        mju_zero(eye, m->nv * m->nv); // zero all entries
        for (int i = 0; i < m->nv; i++)
        {
            eye[i * (m->nv + 1)] = 1.0;
        }
        mj_solveM(m, d, M_inv, eye, m->nv);

        mjtNum temp_mat1[m->nv * (m->nv - 1)];
        mju_mulMatMat(temp_mat1, M_inv, B, m->nv, m->nv, m->nv - 1);
        mju_mulMatMat(A, J_task_pend, temp_mat1, 2, m->nv, m->nv - 1);

        // Calculate output and output derivatives and formulate b vector; b = y_ddot +J_task*M_inv*(C*q_dot+G)  + eta
        mjtNum temp_vec4[m->nv * 1];
        mjtNum temp_vec5[2];
        mju_add(d->qfrc_bias, d->qfrc_bias, d->qfrc_passive, m->nv);
        mju_mulMatVec(temp_vec4, M_inv, d->qfrc_bias, m->nv, m->nv); // M_inv*(C*q_dot+G)
        mju_mulMatVec(temp_vec5, J_task_pend, temp_vec4, 2, m->nv);  // J_task*M_inv*(C*q_dot+G)

        mju_mulMatVec(y, Q, u_global, 2, 3);
        mju_mulMatVec(dy, J_task_pend, dq_current, 2, m->nv);

        mjtNum Kp_term[2];
        mju_mulMatVec(Kp_term, Kp, y, 2, 2);

        mjtNum Kd_term[2];
        mju_mulMatVec(Kd_term, Kd, dy, 2, 2);

        mjtNum ddy_temp[2];
        mju_add(ddy_temp, Kp_term, Kd_term, 2);
        mju_scl(nu, ddy_temp, -1, 2);

        mjtNum b_temp[2];
        mju_add(b_temp, nu, temp_vec5, 2);
        mju_sub(b, b_temp, eta, 2);

        // Map A (2x6) - specify RowMajor to match MuJoCo
        Eigen::Map<Eigen::Matrix<mjtNum, 2, 6, Eigen::RowMajor>> A_eig(A);

        // Map b (2x1)
        Eigen::Map<Eigen::Vector<mjtNum, 2>> b_eig(b);

        // Map d->ctrl (Output, 6x1)
        // This allows us to write the result directly into the physics data
        Eigen::Map<Eigen::Vector<mjtNum, 6>> ctrl_eig(d->ctrl);

        // Damped Pseudo-Inverse
        double lambda = 1e-4; // Damping factor to address singularity issues

        // Compute Gram Matrix: G = A * A^T (Result is 2x2)
        Eigen::Matrix<mjtNum, 2, 2> G = A_eig * A_eig.transpose();

        // Add damping to the diagonal for stability
        G.diagonal().array() += lambda;

        // --- Control Block ---
        // Logic: tau = A^T * (G^-1 * b)
        // We use .ldlt().solve() which is the fast/robust way to do "Inverse * Vector"
        ctrl_eig = A_eig.transpose() * G.ldlt().solve(b_eig);

        for (int i = 0; i < m->nu; i++)
        {
            std::cout << d->ctrl[i] << ",";
        }
        std::cout << std::endl;

        // --- NULL SPACE POSTURE CONTROL ---
        // 1. Define Desired Posture (e.g., standard "Up" pose for UR5)
        Eigen::Vector<double, 6> q_des;
        q_des << 0, -1.57, 1.57, -1.57, -1.57, 0;

        // 2. PD Control for Posture
        Eigen::Vector<double, 6> tau_posture;
        Eigen::Map<Eigen::Vector<double, 7>> q_map(q_current);
        Eigen::Map<Eigen::Vector<double, 7>> dq_map(dq_current);

        // Stiffness to hold the arm up
        double kp_posture = 100.0;
        double kd_posture = 20.0;

        for (int i = 0; i < 6; i++)
        {
            tau_posture(i) = -kp_posture * (q_map(i) - q_des(i)) - kd_posture * dq_map(i);
        }

        // 3. Project into Null Space: N = I - A_dagger * A
        // We need A_dagger explicitly.
        // A_dagger = A^T * (A A^T + lambda I)^-1
        Eigen::Matrix<double, 6, 2> A_dagger = A_eig.transpose() * G.ldlt().solve(Eigen::Matrix<double, 2, 2>::Identity());

        Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, 6, 6> N = I6 - (A_dagger * A_eig);

        // 4. ADD to the Control Output
        ctrl_eig = ctrl_eig + (N * tau_posture);

        // --- MuJoCo Physics Step ---
        mj_step(m, d);

        // --- Rendering Block ---
        // Inside the while (d->time < sim_duration) loop:

        // --- Rendering Block ---
        if (!glfwWindowShouldClose(window))
        {
            // 1. SET and CLEAR the background color using OpenGL
            // The RGBA values must be float (0.0f to 1.0f)
            glClearColor(0.4f, 0.42f, 0.44f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the buffer

            // 2. Update scene geometry to current mjData state
            mjv_updateScene(m, d, &vopt, &pert, &cam, ~0, &scn);

            // 3. Render the scene (Draws on top of the cleared background)
            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
            mjr_render(viewport, &scn, &con);

            // 4. Swap buffers (display the new frame)
            glfwSwapBuffers(window);

            // 5. Handle window events (input, closing, etc.)
            glfwPollEvents();
        }
        // ...
        else
        {
            break; // Exit loop if the window is closed
        }

        step_count++;

        // --- Data Logging/Printing (Optional) ---
        if (step_count % 1000 == 0)
        {
            printf("Time: %.3f s, Joint 1 Pos (rad): %.3f\n", d->time, d->qpos[0]);
        }
    }

    std::cout << "\nSimulation finished. Total steps: " << step_count << std::endl;

    // 5. Clean up
    mjr_freeContext(&con);
    mjv_freeScene(&scn);

    glfwDestroyWindow(window);
    glfwTerminate();

    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods)
{
    // Check for "press" action (GLFW_PRESS)
    if (act == GLFW_PRESS)
    {
        switch (key)
        {
        // Quit on ESC
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;

        // Reset simulation on SPACE
        case GLFW_KEY_SPACE:
            mj_resetData(m, d);
            // Optional: reload model to reset visualization state
            // m = mj_loadXML(xmlpath, 0, 0, 0);
            break;

        // Example: Toggle contact visualization (F1 key)
        case GLFW_KEY_F1:
            vopt.flags[mjVIS_CONTACTPOINT] = 1 - vopt.flags[mjVIS_CONTACTPOINT];
            break;

        default:
            break;
        }
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // Update global state of mouse buttons
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // Update the last position when a button is pressed
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // No buttons pressed, do nothing
    if (!button_left && !button_middle && !button_right)
        return;

    // FIX: Change mjvGeomType to mjtMouse, and use correct enum members
    mjtMouse action;

    if (button_right)
        action = mjMOUSE_ROTATE_V; // Rotate
    else if (button_left)
        action = mjMOUSE_MOVE_V; // Translate/Pan
    else
        action = mjMOUSE_ZOOM; // Zoom

    // Calculate movement difference
    double dx = xpos - lastx;
    double dy = ypos - lasty;

    // Get the current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Apply camera movement
    // The correct call for your MuJoCo version requires 6 arguments.
    // We add &scn as the final parameter.
    mjv_moveCamera(m, action, dx / width, dy / height, &scn, &cam); // FIXED LINE

    // Update last position
    lastx = xpos;
    lasty = ypos;
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    mjtMouse action = mjMOUSE_ZOOM;

    // ADD &scn as the final argument here as well
    mjv_moveCamera(m, action, 0, -0.05 * yoffset, &scn, &cam); // FIXED LINE
}