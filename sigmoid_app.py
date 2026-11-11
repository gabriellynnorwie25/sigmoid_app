
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function")

st.title("Sigmoid Activation Function")

st.write("""
The Sigmoid function converts an input value
into an output between 0 and 1.
""")

st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")

# Input slider
x_value = st.slider(
    "Choose an input value",
    -10.0, 10.0, 0.0, 0.5
)

# Calculate output
y_value = 1 / (1 + np.exp(-x_value))

st.metric("Sigmoid Output", f"{y_value:.4f}")

# Create graph
x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots()

ax.plot(x, y, label="Sigmoid")
ax.scatter(x_value, y_value, color="red",
           label="Selected Input")

ax.set_title("Sigmoid Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.subheader("Explanation")

st.write("1. The output is always between 0 and 1.")
st.write("2. When the input is 0, the output is 0.5.")
st.write("3. Sigmoid is useful for binary classification.")