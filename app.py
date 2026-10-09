import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from engine.layers import DenseLayer, ReLUActivation
from engine.models import SequentialModel

st.set_page_config(page_title="NumPy Neural Engine", layout="wide")
st.title("🛰️ Deep NumPy Neural Framework Engine")
st.caption("Coded Natively from Scratch Using Raw Matrix Mathematics (No PyTorch/TensorFlow)")

# User Interface Sidebar Calibration Sliders
learning_rate = st.sidebar.slider("Calibration Learning Rate (α)", 0.001, 0.5, 0.05)
epochs = st.sidebar.slider("Training Epoch Execution Cycles", 50, 1000, 200)

# Simulated Satellite Data Matrices Creation
np.random.seed(42)
X_train = np.random.rand(100, 4)
y_train = np.zeros(100, dtype=int)
y_train[X_train[:, 3] > 0.6] = 1  
y_train[(X_train[:, 0] > 0.5) & (X_train[:, 2] > 0.5)] = 2
Y_hot = np.eye(3)[y_train]

if st.button("🚀 Compile and Initialize Framework Pipeline", use_container_width=True):
    model = SequentialModel()
    model.add(DenseLayer(4, 8))
    model.add(ReLUActivation())
    model.add(DenseLayer(8, 3))
    
    loss_history = []
    
    # Run the iterative structural loop
    for epoch in range(epochs):
        # Softmax evaluation
        raw_scores = model.forward(X_train)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Calculate Categorical Cross-Entropy loss constants
        loss = -np.sum(Y_hot * np.log(probs + 1e-15)) / 100
        loss_history.append(loss)
        
        # Derivative calculations pass backpropagation tracking
        d_output = (probs - Y_hot) / 100
        model.backward(d_output, learning_rate)
        
    st.success("🎯 Modular system layers calibrated online. Optimization complete.")
    
    # Plot the optimization chart magnitudes
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(loss_history, color="#FF4B4B", linewidth=2)
    ax.set_title("Framework Error Optimization Track (Loss Curve)", fontsize=10)
    ax.set_xlabel("Epoch Cycles", fontsize=8)
    ax.set_ylabel("Loss Magnitude", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        final_preds = np.argmax(probs, axis=1)
        accuracy = np.mean(final_preds == y_train) * 100
        st.metric("Framework Classification Accuracy", f"{accuracy:.2f}%")

