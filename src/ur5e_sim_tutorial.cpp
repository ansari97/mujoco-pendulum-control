#include <iostream>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// --- Global Variables (from previous code) ---
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

// ... (rest of the main function setup) ...

int main(int argc, const char **argv)
{
    // 1. Check for the XML file path
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

    if (m->nkey > 0)
    {
        mju_copy(d->qpos, m->key_qpos, m->nq);
        mj_forward(m, d);
    }

    // 1. Initialize GLFW library
    if (!glfwInit())
    {
        std::cerr << "Could not initialize GLFW" << std::endl;
        return 1;
    }

    // 2. Create the GLFW window
    window = glfwCreateWindow(1200, 900, "UR5e MuJoCo Simulation", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        std::cerr << "Could not create GLFW window" << std::endl;
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-sync

    // 3. Initialize MuJoCo visualization structures
    mjv_defaultCamera(&cam);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // NEW: Initialize visualization options (vopt) and perturbation (pert)
    mjv_defaultOption(&vopt);
    vopt.frame = mjFRAME_BODY;
    // vopt.frame = mjFRAME_;
    mjv_defaultPerturb(&pert);

    // 4. Create the scene and context based on the model
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Optional: Set initial camera position for a good view (e.g., side view)
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

    // 4. Zero Input Simulation Loop
    double sim_duration = 15.0;         // seconds
    double time_step = m->opt.timestep; // Time step defined in the XML (e.g., 0.002s)
    int step_count = 0;

    std::cout << "Starting UR5e zero-input simulation for " << sim_duration << "s..." << std::endl;
    std::cout << "Time step: " << time_step << "s" << std::endl;

    while (d->time < sim_duration)
    {
        // --- Zero-Input Control Block ---
        for (int i = 0; i < m->nu; ++i)
        {
            d->ctrl[i] = 0.0;
        }

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

    std::cout << "Simulation finished. Total steps: " << step_count << std::endl;

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