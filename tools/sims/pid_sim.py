# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # Define PID parameters
# # # kp = 0.5
# # # ki = 0.2
# # # kd = 0.1

# # # # Define setpoint and initial conditions
# # # setpoint = 0
# # # current = 0
# # # prev_error = 0
# # # integral = 0

# # # # Define simulation parameters
# # # dt = 0.01  # Time step
# # # total_time = 5  # Total simulation time
# # # time = np.arange(0, total_time, dt)
# # # output = np.zeros(len(time))

# # # # Simulate PID control
# # # for i in range(1, len(time)):
# # #     error = setpoint - current
# # #     integral += error * dt
# # #     derivative = (error - prev_error) / dt
# # #     output[i] = kp * error + ki * integral + kd * derivative
# # #     current += output[i] * dt  # Update current based on PID output
# # #     prev_error = error

# # # # Plot the results
# # # plt.figure()
# # # plt.plot(time, output, label='PID Output')
# # # plt.xlabel('Time')
# # # plt.ylabel('Output')
# # # plt.title('PID Control Simulation')
# # # plt.legend()
# # # plt.show()
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.animation import FuncAnimation

# # # Define PID parameters
# # kp = 0.5
# # ki = 0.2
# # kd = 0.1

# # # Define initial conditions
# # setpoint = 0
# # current = 0
# # prev_error = 0
# # integral = 0

# # # Create a figure and axis for real-time plotting
# # fig, ax = plt.subplots()
# # ax.set_xlim(0, 5)
# # ax.set_ylim(-1, 1)
# # line, = ax.plot([], [], lw=2)

# # # Initialize data lists for plotting
# # time_list = []
# # output_list = []

# # # Define the update function for real-time plot
# # def update(frame):
# #     global current, integral, prev_error
# #     time_list.append(frame)

# #     # Simulate face movement
# #     face_position = np.sin(2 * np.pi * frame)  # Example face position (sine wave for demonstration)

# #     # Implement PID control
# #     error = setpoint - current
# #     integral += error
# #     derivative = (error - prev_error)
# #     output = kp * error + ki * integral + kd * derivative
# #     current += output

# #     output_list.append(output)
# #     line.set_data(time_list, output_list)
# #     return line,

# # # Create the animation
# # ani = FuncAnimation(fig, update, frames=np.linspace(0, 5, 100), blit=True)

# # # Show the real-time plot
# # plt.xlabel('Time')
# # plt.ylabel('Output')
# # plt.title('Real-time PID Control Response')
# # plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import ipywidgets as widgets
# from IPython.display import display

# # Initialize the figure and axis
# fig, ax = plt.subplots()
# ax.set_xlim(0, 5)
# ax.set_ylim(-1, 1)
# line, = ax.plot([], [], lw=2)

# # Initialize data lists
# time_list = []
# output_list = []

# # Define the PID control function
# def pid_control(setpoint, kp, ki, kd):
#     global current, integral, prev_error
#     error = setpoint - current
#     integral += error
#     derivative = (error - prev_error)
#     output = kp * error + ki * integral + kd * derivative
#     current += output
#     return output

# # Define the update function for real-time plot
# def update(frame, setpoint, kp, ki, kd):
#     time_list.append(frame)

#     # Simulate face movement
#     face_position = np.sin(2 * np.pi * frame)  # Example face position (sine wave for demonstration)

#     # Implement PID control
#     output = pid_control(setpoint, kp, ki, kd)

#     output_list.append(output)
#     line.set_data(time_list, output_list)
#     return line,

# # Create the animation
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 5, 100), fargs=(0, 0.5, 0.2, 0.1), blit=True)

# # Create interactive sliders for parameters
# setpoint_slider = widgets.FloatSlider(value=0, min=-1, max=1, step=0.1, description='Setpoint')
# kp_slider = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.1, description='Kp')
# ki_slider = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.1, description='Ki')
# kd_slider = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.1, description='Kd')

# # Define the function for updating the sliders
# def update_sliders(change):
#     setpoint = setpoint_slider.value
#     kp = kp_slider.value
#     ki = ki_slider.value
#     kd = kd_slider.value
#     ani.event_source.stop()
#     ani.event_source.start()

# # Observe changes in sliders
# setpoint_slider.observe(update_sliders, 'value')
# kp_slider.observe(update_sliders, 'value')
# ki_slider.observe(update_sliders, 'value')
# kd_slider.observe(update_sliders, 'value')

# # Display the sliders and plot
# display(widgets.VBox([setpoint_slider, kp_slider, ki_slider, kd_slider]))
# plt.xlabel('Time')
# plt.ylabel('Output')
# plt.title('Real-time PID Control Response')
# plt.show()
 
import numpy as np
import matplotlib.pyplot as plt

# Define PID parameters
kp = 0.1
ki = 0.01
kd = 0.5

# Define initial conditions
setpoint_x = 0
setpoint_y = 0
current_x = 0
current_y = 0
prev_error_x = 0
prev_error_y = 0
integral_x = 0
integral_y = 0

# Simulation parameters
time = np.linspace(0, 10, 100)
face_position_x = np.sin(2 * np.pi * time)
face_position_y = np.cos(2 * np.pi * time)
camera_position_x = np.zeros_like(time)
camera_position_y = np.zeros_like(time)

# Simulate PID control
for i in range(len(time)):
    error_x = setpoint_x - current_x
    integral_x += error_x
    derivative_x = error_x - prev_error_x
    output_x = kp * error_x + ki * integral_x + kd * derivative_x
    current_x += output_x
    camera_position_x[i] = current_x

    error_y = setpoint_y - current_y
    integral_y += error_y
    derivative_y = error_y - prev_error_y
    output_y = kp * error_y + ki * integral_y + kd * derivative_y
    current_y += output_y
    camera_position_y[i] = current_y

    prev_error_x = error_x
    prev_error_y = error_y

# Plot the simulation results
plt.figure(figsize=(10, 6))
plt.plot(time, face_position_x, label='Face Position X', linestyle='--')
plt.plot(time, camera_position_x, label='Camera Position X')
plt.plot(time, face_position_y, label='Face Position Y', linestyle='--')
plt.plot(time, camera_position_y, label='Camera Position Y')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Simulation of Face Position and Camera PID Control')
plt.legend()
plt.show()
