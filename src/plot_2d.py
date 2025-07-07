import matplotlib.pyplot as plt


time_labels = ["0:00", "3:48", "6:56", "10:00", "13:00"]
device_0_errors = [0.10, 0.17, 0.23, 0.29, 0.35]
device_1_errors = [0.10, -0.51, -0.98, -0.92, -1.80]
device_2_errors = [-0.53, -0.63, -0.79, -0.90, -1.02]

# Create the plot with larger font sizes and annotated y-values
plt.figure(figsize=(10, 6))
plt.plot(time_labels, device_0_errors, marker='o', label='Device_0')
plt.plot(time_labels, device_1_errors, marker='o', label='Device_1')
plt.plot(time_labels, device_2_errors, marker='o', label='Device_2')

# Annotate each point with its y-value using a larger font size
for i, time in enumerate(time_labels):
    plt.text(time, device_0_errors[i] + 0.05, f"{device_0_errors[i]:.2f}", ha='center', va='bottom', fontsize=12)
    plt.text(time, device_1_errors[i] - 0.05, f"{device_1_errors[i]:.2f}", ha='center', va='top', fontsize=12)
    plt.text(time, device_2_errors[i] - 0.05, f"{device_2_errors[i]:.2f}", ha='center', va='top', fontsize=12)

# Add labels and title with larger font sizes
plt.xlabel('Time', fontsize=14)
plt.ylabel('Phase Error (ms)', fontsize=14)
plt.title('Phase Error Over Time for Three Devices', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()
