import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import minimize

# Step 1: Simulate Plant Height Data from Gamma Distribution
def simulate_plant_heights(alpha, beta, size):
    """ Simulate plant height data from a Gamma distribution. """
    return gamma.rvs(a=alpha, scale=1/beta, size=size)

# Step 2: Method of Moments to estimate initial parameters
def initial_params_moments(data):
    """ Estimate parameters (alpha, beta) of Gamma distribution using method of moments. """
    mu = np.mean(data)
    var = np.var(data)
    
    alpha = mu**2 / var
    beta = mu / var
    return alpha, beta

# Step 3: Estimate Parameters using Maximum Likelihood Estimation (MLE)
def gamma_mle(data):
    """ Estimate parameters (alpha, beta) of Gamma distribution using MLE. """
    def negative_log_likelihood(params):
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf  # Return infinity if parameters are invalid
        return -np.sum(gamma.logpdf(data, a=alpha, scale=1/beta))

    # Use method of moments for initial guess
    initial_alpha, initial_beta = initial_params_moments(data)
    initial_params = [initial_alpha, initial_beta]
    bounds = [(1e-5, None), (1e-5, None)]  # Prevent parameters from being negative
    result = minimize(negative_log_likelihood, initial_params, bounds=bounds)

    if not result.success:
        print("Optimization failed: ", result.message)
    return result.x  # Return estimated parameters (alpha, beta)

# Step 4: Compute Fisher Information Matrix (FIM) for Gamma distribution
def fisher_information_matrix(alpha, beta):
    """ Calculate the Fisher Information Matrix for Gamma distribution. """
    I11 = 1 / alpha
    I12 = -1 / beta
    I21 = -1 / beta
    I22 = 1 / (beta**2)
    return np.array([[I11, I12], [I21, I22]])

# Step 5: Compute Uncertainty Measure
def uncertainty_measure(fim):
    """ Compute uncertainty measure as the determinant of the Fisher Information Matrix. """
    return np.linalg.det(fim)

# Active Learning Function
def active_learning(initial_samples, total_samples, uncertainty_threshold, alpha_true, beta_true):
    """ Active learning loop based on uncertainty measure. """
    # Initial sample data
    heights = simulate_plant_heights(alpha_true, beta_true, initial_samples)

    # Simulate additional data points to query
    additional_heights = simulate_plant_heights(alpha_true, beta_true, total_samples - initial_samples)

    # Active learning loop
    for i in range(total_samples - initial_samples):
        # Estimate parameters and calculate FIM
        estimated_alpha, estimated_beta = gamma_mle(heights)
        fim = fisher_information_matrix(estimated_alpha, estimated_beta)
        uncertainty = uncertainty_measure(fim)

        # Print current uncertainty measure
        print(f"Iteration {i + 1}: Current Uncertainty Measure: {uncertainty:.4f}")

        # Decide whether to include the next sample based on uncertainty
        if uncertainty < uncertainty_threshold:
            # Add the next sample from additional_heights
            heights = np.append(heights, additional_heights[i])
        else:
            print("Uncertainty is high, querying next sample...")

    return heights

# Parameters for simulation
alpha_true = 0.8  # True shape parameter
beta_true = 1.0   # True scale parameter
initial_samples = 100   # Initial number of samples
total_samples = 500      # Total number of samples
uncertainty_threshold = 1.0  # Threshold for active learning

# Run active learning
heights = active_learning(initial_samples, total_samples, uncertainty_threshold, alpha_true, beta_true)

# Final estimation after active learning
estimated_alpha, estimated_beta = gamma_mle(heights)
fim = fisher_information_matrix(estimated_alpha, estimated_beta)
uncertainty = uncertainty_measure(fim)

# Print results
print(f"Final Estimated Parameters: Alpha = {estimated_alpha:.4f}, Beta = {estimated_beta:.4f}")
print(f"Final Fisher Information Matrix:\n{fim}")
print(f"Final Uncertainty Measure: {uncertainty:.4f}")

# Visualization
plt.hist(heights, bins=15, density=True, alpha=0.6, color='g', label='Simulated Heights')
x = np.linspace(0, 15, 100)
plt.plot(x, gamma.pdf(x, a=estimated_alpha, scale=1/estimated_beta), 'r-', lw=2, label='Fitted Gamma PDF')
plt.xlabel('Plant Height')
plt.ylabel('Density')
plt.title('Plant Height Distribution after Active Learning')
plt.legend()
plt.grid()
plt.show()

