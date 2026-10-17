import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="AdaGrad Visual Lab", layout="wide")
st.title("AdaGrad: Interactive Optimization Visualizer")
st.caption("Explore coordinate-wise adaptive step sizes on a convex quadratic objective.")

st.markdown(
    r"""
We minimize **\(f(x,y)=\frac{1}{2}(x^2+10y^2)\)**, whose gradient is
\(\nabla f(x,y)=(x,10y)\). The unequal curvature makes the two coordinates
experience different gradient scales.
"""
)

with st.sidebar:
    st.header("Experiment controls")
    x0 = st.slider("Initial x", -5.0, 5.0, 4.0, 0.25)
    y0 = st.slider("Initial y", -5.0, 5.0, 3.0, 0.25)
    eta = st.slider("Base learning rate η", 0.01, 1.0, 0.30, 0.01)
    eps = st.select_slider("Epsilon", options=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4], value=1e-8,
                           format_func=lambda z: f"{z:.0e}")
    T = st.slider("Total iterations", 1, 100, 30, 1)
    frame = st.slider("Show through iteration", 0, T, T, 1)
    st.markdown("---")
    st.write("**Update convention**")
    st.latex(r"G_t=G_{t-1}+g_t\odot g_t")
    st.latex(r"\theta_{t+1}=\theta_t-\eta\,g_t/(\sqrt{G_t}+\epsilon)")

def run_adagrad(x_init, y_init, learning_rate, epsilon, steps):
    theta = np.array([x_init, y_init], dtype=float)
    G = np.zeros(2, dtype=float)
    rows = [{
        "iteration": 0, "x": theta[0], "y": theta[1],
        "loss": 0.5 * (theta[0]**2 + 10 * theta[1]**2),
        "grad_x": np.nan, "grad_y": np.nan,
        "G_x": G[0], "G_y": G[1],
        "lr_x": np.nan, "lr_y": np.nan,
        "step_x": np.nan, "step_y": np.nan
    }]
    points = [theta.copy()]
    for t in range(steps):
        grad = np.array([theta[0], 10.0 * theta[1]])
        G = G + grad**2
        effective_lr = learning_rate / (np.sqrt(G) + epsilon)
        step = effective_lr * grad
        theta = theta - step
        loss = 0.5 * (theta[0]**2 + 10 * theta[1]**2)
        rows.append({
            "iteration": t + 1, "x": theta[0], "y": theta[1], "loss": loss,
            "grad_x": grad[0], "grad_y": grad[1],
            "G_x": G[0], "G_y": G[1],
            "lr_x": effective_lr[0], "lr_y": effective_lr[1],
            "step_x": step[0], "step_y": step[1]
        })
        points.append(theta.copy())
    return pd.DataFrame(rows), np.array(points)

df, points = run_adagrad(x0, y0, eta, eps, T)
shown = df.iloc[:frame + 1].copy()
shown_points = points[:frame + 1]

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Parameter trajectory on the loss surface")
    grid = np.linspace(-5, 5, 180)
    X, Y = np.meshgrid(grid, grid)
    Z = 0.5 * (X**2 + 10 * Y**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    levels = np.geomspace(0.03, 140, 22)
    ax.contour(X, Y, Z, levels=levels)
    if len(shown_points) > 1:
        ax.plot(shown_points[:, 0], shown_points[:, 1], marker="o", markersize=3)
    ax.scatter([x0], [y0], marker="s", s=70, label="Start")
    ax.scatter([shown_points[-1, 0]], [shown_points[-1, 1]], marker="*", s=150, label=f"Iteration {frame}")
    ax.scatter([0], [0], marker="x", s=80, label="Minimum (0, 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Contour lines of f(x, y)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("2. Loss over iterations")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(shown["iteration"], shown["loss"], marker="o", markersize=3)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss f(x, y)")
    ax2.set_title("Objective value after each update")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

st.subheader("3. Coordinate-wise effective learning rates")
rates = df.iloc[1:frame + 1]
if not rates.empty:
    fig3, ax3 = plt.subplots(figsize=(9, 3.8))
    ax3.plot(rates["iteration"], rates["lr_x"], marker=".", label="Effective LR for x")
    ax3.plot(rates["iteration"], rates["lr_y"], marker=".", label="Effective LR for y")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Effective learning rate")
    ax3.set_title("η / (sqrt(accumulated squared gradient) + ε)")
    ax3.legend()
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("Move 'Show through iteration' above zero to display learning-rate history.")

st.subheader("4. Current state")
current = shown.iloc[-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Iteration", int(current["iteration"]))
m2.metric("x", f'{current["x"]:.6f}')
m3.metric("y", f'{current["y"]:.6f}')
m4.metric("Loss", f'{current["loss"]:.8f}')

st.subheader("5. Step-by-step numerical trace")
st.write("The gradient and learning rates in a row are those used to produce that row's updated parameters.")
display_cols = ["iteration", "x", "y", "loss", "grad_x", "grad_y", "G_x", "G_y", "lr_x", "lr_y", "step_x", "step_y"]
st.dataframe(shown[display_cols].round(8), use_container_width=True, hide_index=True)

csv = shown.to_csv(index=False).encode("utf-8")
st.download_button("Download displayed trace as CSV", csv, "adagrad_trace.csv", "text/csv")

with st.expander("Interpretation prompts for your report"):
    st.markdown("""
    - Which coordinate starts with the larger gradient magnitude, and why?
    - How does each accumulated squared-gradient term change over time?
    - Do the effective learning rates increase, decrease, or stay constant?
    - Does the loss decrease at every displayed iteration for your chosen settings?
    - How does changing the base learning rate affect the trajectory?
    - Explain why this visualization is an illustration, not proof of universal convergence.
    """)
