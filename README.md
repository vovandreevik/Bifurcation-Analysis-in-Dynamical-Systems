Sure, here is the README in Markdown syntax for GitHub:

---

# Bifurcation Analysis in Dynamical Systems

## Overview

This project involves analyzing the stationary solutions of a system of differential equations depending on the parameter \( p4 \). The primary goal is to plot the dependence of the stationary solutions \( x1(p4) \), \( x2(p4) \), and \( x3(p4) \) on \( p4 \), and mark stable and unstable stationary points, as well as points of real and complex bifurcation if they exist.

## Model Description

The differential equations describing the system are:

$\frac{dx1}{dt} = \frac{(p1 \cdot x2 - x1 \cdot x2 + x1 - x1^2)}{p2} - p4 \cdot x1$

$\frac{dx2}{dt} = \frac{(-p1 \cdot x2 - x1 \cdot x2 + p5 \cdot x3)}{p3} + p4 \cdot (p6 - x2)$

$\frac{dx3}{dt} = x1 - x3 - p4 \cdot x3$

Where:
- $p1 = 8.4 \times 10^{-6}$
- $p2 = 6.6667 \times 10^{-4} $
- $p4$ is the varying parameter
- $p5 = 2 $
- $p6 = 10$  or $p6 = 100$

## How to Run

1. **Install dependencies:**
   Ensure you have the necessary Python packages installed. You can do this via pip:
   ```sh
   pip install numpy matplotlib
   ```

2. **Run the script:**
   Execute the main script to perform the bifurcation analysis.
   ```sh
   python bifurcation_analysis.py
   ```

3. **Output:**
   The script will generate and display several plots:
   - Plots for \( x1 \), \( x2 \), and \( x3 \) as functions of \( p4 \).
   - Stability of stationary points will be indicated with different colors (e.g., green for stable, red for unstable).

## Functions

### `JacobyMatrix(x1, x2, x3, p4)`
Calculates the Jacobian matrix for the system at given values of \( x1 \), \( x2 \), \( x3 \), and \( p4 \).

### `check(x1, x2, x3, p4)`
Checks if the provided stationary points satisfy the system of equations.

### `calculate_eigenvalues(p4, p1, p2, p3, p5, p6)`
Main function to compute and plot the stationary solutions and their stability. It iterates over a range of \( p4 \) values, computes the stationary points and their stability, and then generates the plots.

## Results

The results of the analysis are saved in text files and plotted using matplotlib. The plots show how the stationary solutions \( x1 \), \( x2 \), and \( x3 \) vary with \( p4 \), and they also indicate the stability of these solutions.
