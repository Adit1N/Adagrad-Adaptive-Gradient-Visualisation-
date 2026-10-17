# AdaGrad Optimization Visualization

## Overview
This project demonstrates the AdaGrad adaptive optimization algorithm using an interactive Python visualization.

The objective function is:

\[
f(x,y)=\frac{1}{2}(x^2+10y^2)
\]

The visualization shows the parameter trajectory, loss over iterations, and coordinate-wise effective learning rates.

## Features
- Interactive optimization controls
- Contour plot of the objective function
- Loss-versus-iteration plot
- Coordinate-wise effective learning-rate plots
- Numerical iteration trace and CSV export

## Technologies
- Python
- Streamlit
- NumPy
- Matplotlib
- Pandas

## How to Run

1. Install Python.
2. Open a terminal in the `animation` folder.
3. Install the dependencies:

   `python -m pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Repository Contents
- `main.tex`: LaTeX report source
- `references.bib`: Bibliography
- `report.pdf`: Compiled report
- `animation/`: Python visualization
- `figures/`: Report figures and screenshots

## Academic Note
The report discusses the algorithm, its computational complexity, applications, and limitations. Experimental observations should be verified against the actual program output.