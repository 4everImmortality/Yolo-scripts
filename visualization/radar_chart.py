import matplotlib.pyplot as plt
import numpy as np

def plot_yolo_radar_chart(metrics_names, metrics_values, title="YOLO Performance Radar Chart"):
    num_metrics = len(metrics_names)

    # Ensure metrics_names and metrics_values have the same length
    if num_metrics != len(metrics_values):
        raise ValueError("Number of metric names must match number of metric values.")

    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

    # The plot is cyclic, so we need to add the first value and angle again to close the circle
    values = metrics_values + [metrics_values[0]]
    angles = angles + [angles[0]]
    names = metrics_names + [metrics_names[0]] # Add name for the extra angle for labelling consistency

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)

    # Fill area
    ax.fill(angles, values, alpha=0.25)

    # Set the labels for each axis
    # We need to use the original angles and names for labelling, not the extended ones
    ax.set_xticks(angles[:-1]) # Use original angles
    ax.set_xticklabels(metrics_names) # Use original names

    # Set the range of the radar chart (e.g., from 0 to 1 or slightly more)
    # Assuming metrics are between 0 and 1
    ax.set_rlabel_position(0) # Position the radial labels
    ax.set_ylim(0, 1.1) # Set radial limits, slightly above 1 for padding
    ax.set_yticks(np.arange(0, 1.1, 0.2)) # Optional: set specific radial tick marks
    ax.set_yticklabels([str(round(y, 1)) for y in np.arange(0, 1.1, 0.2)]) # Optional: label radial ticks

    # Add title
    ax.set_title(title, va='bottom')

    # Add grid
    ax.grid(True)

    # Optional: Adjust angular offset and direction for better appearance
    # ax.set_theta_offset(np.pi / 2) # Start the first axis at the top
    # ax.set_theta_direction(-1) # Go clockwise

    plt.show()

# --- Example Usage ---
metrics_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-score']
# Replace with your actual model's performance values (between 0 and 1)
metrics_values = [0.85, 0.70, 0.88, 0.82, 0.85]

plot_yolo_radar_chart(metrics_names, metrics_values, title="YOLO Model Performance")

# Example with slightly different values
# metrics_values_model2 = [0.82, 0.68, 0.85, 0.80, 0.83]
# plot_yolo_radar_chart(metrics_names, metrics_values_model2, title="YOLO Model 2 Performance")
