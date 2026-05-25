# Wrinkle-Formation-and-Elastica-Tracking-Dynamics
# Wrinkle Formation and Elastica Tracking Dynamics

## Overview
This repository contains numerical solvers and visualization tools developed to model wrinkle formation and adiabatic tracking limitations in constrained physical hamiltonian systems. By reducing complex material behavior into solvable elliptical ODEs, this project maps phase space dynamics, structural evolution, and the specific conditions leading to numerical collapse (separatrix) in tracking systems.

## Project Structure

### 1. 2D Complex Tracking (`/2d_complex`)
The core analytical component of the repository. It contains algorithms that solve for local tracking energy angular coordiantes.
* **Energy Solvers:** Computes local energy $E(s)$ across a varying tension profile $\lambda(s)$.
* **Spatial ODE System:** Solves the kinematics of the system mapping the geometry of the wrinkles based on velocity parameters ($A/L$).
* **Phase Space Analysis:** Identifies the separatrix where the tracker can no longer adiabatically follow the target trajectory.

### 2. 1D Simplified Tracking (`/1d_simplified`)
Foundational scripts demonstrating the basic geometric tracking problem. It models an irregular tracker path (sum of sine waves) oscillating around a fixed target path, calculating local tangent vectors and tracking angles $\theta(s)$.

### 3. Adiabatic Piston Visualizations (`/adiabatic_piston`)
Supplementary visualization tools generating diagrams that map the physical state and boundary constraints of the tracking environment.

## Mathematical Framework
The 2D spatial marching algorithms map physical constraints directly to Hamiltonian phase space. The action integral $I(E, \lambda)$ is defined using complete elliptic integrals of the first and second kind:
* $K(m)$: `scipy.special.ellipk`
* $E(m)$: `scipy.special.ellipe`

Where the parameter $m$ is driven by the ratio of local energy to uniform pressure. The scripts compute the roots of the objective functions to dictate the material heading $\theta(s)$ and reconstruct the Cartesian coordinates $(\xi, \eta)$.

## Dependencies
This project requires Python 3.x and the following libraries:
* `numpy` (Array operations and data structuring)
* `scipy` (Root-finding `optimize`, integration `solve_ivp`, and special functions)
* `matplotlib` (Data visualization and inset mapping)

## Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/wrinkle-formation-dynamics.git](https://github.com/yourusername/wrinkle-formation-dynamics.git)
cd wrinkle-formation-dynamics
pip install numpy scipy matplotlib
