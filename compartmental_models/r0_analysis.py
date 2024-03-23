import numpy as np
import matplotlib.pyplot as plt


# Setting the recovery rate to be constant
gamma = 1/7


def get_r0_value(beta, mu_D, sigma):
    
    # Returning the R0 value for the SIM model (with one variant)
    return (beta/(mu_D + gamma)) * (1 + sigma)


# Creating a list of test transmission and death rates
betas = np.linspace(start=0, stop=1, num=1000)
death_rates = np.linspace(start=0, stop=1, num=1000)

# Creating lists to store the results
r0_sigma_0 = np.zeros(shape=(len(betas), len(death_rates)))
r0_sigma_05 = np.zeros(shape=(len(betas), len(death_rates)))
r0_sigma_1 = np.zeros(shape=(len(betas), len(death_rates)))

# Looping through combinations
for j, b in enumerate(betas):
    for i, mu in enumerate(death_rates):

        # Determining the results
        r0_sigma_0[j, i] = get_r0_value(b, mu, sigma=0)
        r0_sigma_05[j, i] = get_r0_value(b, mu, sigma=0.5)
        r0_sigma_1[j, i] = get_r0_value(b, mu, sigma=1)


# Creating three subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Displaying the first plot
axs[0].set_title('R0 for various rates and sigma=0')
axs[0].contourf(betas, death_rates, r0_sigma_0)
axs[0].set_xlabel('Transmission rate (1/day)')
axs[0].set_ylabel('Death rate (1/day)')

# Displaying the second plot
axs[1].set_title('R0 for various rates and sigma=0.5')
axs[1].contourf(betas, death_rates, r0_sigma_05)
axs[1].set_xlabel('Transmission rate (1/day)')
axs[1].set_ylabel('Death rate (1/day)')

# Displaying the third plot
axs[2].set_title('R0 for various rates and sigma=1')
axs[2].contourf(betas, death_rates, r0_sigma_1)
axs[2].set_xlabel('Transmission rate (1/day)')
axs[2].set_ylabel('Death rate (1/day)')

# Setting the layout
plt.tight_layout()
plt.show()
