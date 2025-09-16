import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# Load the trained model (SavedModel)
# --------------------------------------
@st.cache_resource  # cache so it doesn't reload every interaction
def load_model():
    net_u = tf.saved_model.load("saved_net_u")
    infer = net_u.signatures["serving_default"]
    return infer

infer = load_model()

# --------------------------------------
# App title and description
# --------------------------------------
st.title("PINN Concentration Predictor")
st.markdown("""
This application predicts PFAS concentration change over time at a chosen depth `z` using a trained PINN model.
""")

# --------------------------------------
# User inputs
# --------------------------------------
t_max_days = 40  # total simulation time in days
n_points = 200   # number of time points for plotting

depth = st.slider("Select depth z (normalized 0–1)", 0.0, 1.0, 0.4, 0.01)

# --------------------------------------
# Generate input for prediction
# --------------------------------------
time_days = np.linspace(0, t_max_days, n_points)
t_max_sec = t_max_days * 24 * 3600
t_norm = time_days * 86400 / t_max_sec  # normalize to 0-1
y_vals = np.full_like(t_norm, depth, dtype=np.float32)
xy = np.stack([t_norm, y_vals], axis=1).astype(np.float32)

# --------------------------------------
# Predict concentration
# --------------------------------------
xy_tf = tf.convert_to_tensor(xy)
out = infer(xy_tf)
c_pred = list(out.values())[0].numpy().flatten()

# Optional rescale (depends on how you normalized during training)
P_MAX = 1.0
c_pred = c_pred * P_MAX

# --------------------------------------
# Plot results
# --------------------------------------
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time_days, c_pred, 'b-', linewidth=2)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Concentration (ppm)")
ax.set_title(f"Concentration vs Time at z={depth} m")
ax.set_ylim(0, 1)   # <-- fix Y axis from 0 to 1
ax.grid(True)
st.pyplot(fig)

# Display last concentration value

st.write(f"Predicted concentration at final time ({t_max_days} days): {c_pred[-1]:.4f} ppm")

