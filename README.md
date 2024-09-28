# Plant-Height-Simulation-and-Parameter-Estimation-Using-Active-Learning

# Overview
This project simulates plant height data from a Gamma distribution, estimates its parameters using both the method of moments and Maximum Likelihood Estimation (MLE), and implements an active learning approach to improve parameter estimation by querying additional data based on uncertainty measures.

# Dependencies
To run this code, you need the following Python packages:

numpy
matplotlib
scipy
You can install these packages using pip:


Copy code
pip install numpy matplotlib scipy

# Code Description
## 1. Simulate Plant Height Data

The simulate_plant_heights(alpha, beta, size) function simulates plant height data from a Gamma distribution, where:

alpha is the shape parameter.
beta is the scale parameter.
size is the number of samples to generate.

## 2. Method of Moments
The initial_params_moments(data) function estimates initial parameters (alpha, beta) of the Gamma distribution using the method of moments, based on the mean and variance of the data.

## 3. Maximum Likelihood Estimation (MLE)
The gamma_mle(data) function estimates the parameters of the Gamma distribution using MLE. It minimizes the negative log-likelihood function using the SciPy optimizer:

It first derives initial guesses from the method of moments.
It uses bounds to prevent invalid parameter values.

## 4. Fisher Information Matrix (FIM)
The fisher_information_matrix(alpha, beta) function computes the Fisher Information Matrix for the Gamma distribution, which is essential for quantifying the uncertainty in the parameter estimates.

## 5. Uncertainty Measure
The uncertainty_measure(fim) function calculates the uncertainty measure as the determinant of the Fisher Information Matrix.

## 6. Active Learning Function
The active_learning(initial_samples, total_samples, uncertainty_threshold, alpha_true, beta_true) function implements an active learning loop:

It initializes the sample data and queries additional samples based on the uncertainty measure.
If the uncertainty is below a specified threshold, the next sample is added to the dataset; otherwise, the process queries the next sample.

## Usage
You can run the active learning process by setting the following parameters:

-alpha_true: The true shape parameter for the Gamma distribution.
-beta_true: The true scale parameter for the Gamma distribution.
-initial_samples: The initial number of samples to start with.
-total_samples: The total number of samples to collect.
-uncertainty_threshold: The threshold for deciding whether to include the next sample based on uncertainty.


## Example
python
Copy code

# Parameters for simulation
alpha_true = 0.8
beta_true = 1.0
initial_samples = 100
total_samples = 500
uncertainty_threshold = 1.0

# Run active learning
heights = active_learning(initial_samples, total_samples, uncertainty_threshold, alpha_true, beta_true)

# Final estimation after active learning
estimated_alpha, estimated_beta = gamma_mle(heights)
print(f"Final Estimated Parameters: Alpha = {estimated_alpha:.4f}, Beta = {estimated_beta:.4f}")
Visualization
At the end of the simulation, the code visualizes the histogram of the simulated plant heights and overlays the fitted Gamma probability density function.

## Results
The code prints:

Final estimated parameters of the Gamma distribution.
The final Fisher Information Matrix.
The final uncertainty measure.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
Special thanks to the developers of NumPy, Matplotlib, and SciPy for their excellent libraries that made this project possible.
