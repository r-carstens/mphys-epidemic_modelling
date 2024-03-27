import numpy as np
import matplotlib.pyplot as plt

########## BIRTH DYNAMICS

mu_B = 0.15       # birth rate
N_star = 1.0      # carrying capacity of system

########## EPIDEMIOLOGY DYNAMICS

# Disease 1
beta = 1.5        # infection rate
gamma = 1/10      # recovery (to partial immunity)
sigma = 0         # "breakthrough" rate of partial immunity (between 0 and 1)

########## BOTTLENECK DYNAMICS

mu_D_star = 0.09  # death rate baseline
lam = 0.07        # noise element of death rate evolution
nu = 1.0          # death rate reversion to baseline rate
omega = 0.2       # size of shock
kappa = 0.2       # shock frequency


########## CLASS TO STORE COMPARTMENTS
class Compartments:
    def __init__(self, num_S, num_I, num_M, mu_D_star, sim_size):

        # Compartment data
        self.S_data = np.zeros(shape=sim_size)
        self.S_data[0] = num_S

        self.I_data = np.zeros(shape=sim_size)
        self.I_data[0] = num_I

        self.M_data = np.zeros(shape=sim_size)
        self.M_data[0] = num_M

        # Death rate data
        self.death_rates = np.zeros(shape=len(t_range))
        self.death_rates[0] = mu_D_star

        # Additional data
        self.birth_data = np.zeros(shape=sim_size)
        self.infection_data = np.zeros(shape=sim_size)
        self.recovery_data = np.zeros(shape=sim_size)
        self.death_data = np.zeros(shape=sim_size)


########## POPULATION DATA
N = 1                  # total population size
E, I, M = 0, 0.001, 0  # exposed, infected, immune
S = N - I - M          # susceptible

########## TIME DATA
t_max, dt = 250, 0.2
t_range = np.arange(start=0, stop=t_max, step=dt)

########## INITIALISING SIMULATION
compartments = Compartments(num_S=S, num_I=I, num_M=M, sim_size=len(t_range), mu_D_star=mu_D_star)

########## RUNNING SIMULATION
for i in range(len(t_range) - 1):

    # Determining the current population parameters
    S_i, I_i, M_i = compartments.S_data[i], compartments.I_data[i], compartments.M_data[i]
    N_i = S_i + I_i + M_i

    # Determining the current death rate
    mu_D_i = compartments.death_rates[i]

    # Determining the change in the susceptible compartment
    dS_i = dt * (- beta * S_i * I_i                   # infection process (fully susceptible)
                 + mu_B * N_i * (1 - (N_i / N_star))  # birth process
                 - mu_D_i * S_i)                      # natural death process

    # Determining the change in the infected compartment
    dI_i = dt * (beta * S_i * I_i                     # infection process (fully susceptible)
                 + sigma * beta * M_i * I_i           # infection process (partially immune)
                 - gamma * I_i                        # recovery process
                 - mu_D_i * I_i)                      # natural death process

    # Determining the change in the partially immune compartment
    dM_i = dt * (-sigma * beta * M_i * I_i            # infection process (partially immune)
                 + gamma * I_i                        # recovery process
                 - mu_D_i * M_i)                      # natural death process

    # Determining the change in the current death rate
    check_for_shock = int(np.random.uniform() < (kappa * dt))
    dp_death_rate = dt * (lam * np.random.normal() - nu * (mu_D_i - mu_D_star)) \
                    + check_for_shock * omega

    # Determining the number of new deaths
    dDeaths = dt * (mu_D_i * N_i)

    # Determining the number of new births
    dBirths = dt * (mu_B * N_i * (1 - (N_i / N_star)))

    # Determining the number of new infections (partially and fully)
    dInfected = dt * (sigma * beta * M_i * I_i + beta * M_i * I_i)

    # Determining the number of new recoveries
    dRecovered = dt * (gamma * I_i)

    # Updating the compartments
    compartments.S_data[i + 1] = S_i + dS_i
    compartments.I_data[i + 1] = I_i + dI_i
    compartments.M_data[i + 1] = M_i + dM_i

    # Updating the remaining useful data
    compartments.death_data[i] = dDeaths
    compartments.infection_data[i] = dInfected
    compartments.birth_data[i] = dBirths
    compartments.recovery_data[i] = dRecovered

    # Updating the death rate
    compartments.death_rates[i + 1] = min(max(mu_D_i + dp_death_rate, 0), max(1, omega))


########## PLOTTING RESULTS
plt.clf()
plt.title('Evolution of compartment sizes over time for SIM compartmental model \nwith one disease')
plt.xlabel('Time')
plt.ylabel('Total')
plt.plot(t_range, compartments.S_data, label='Susceptible')
plt.plot(t_range, compartments.I_data, label='Infected')
plt.plot(t_range, compartments.M_data, label='Immune')
plt.legend()
plt.show()

plt.clf()
plt.title('Evolution of death rates over time for SIM compartmental model \nwith one disease')
plt.xlabel('Time')
plt.ylabel('Death Rate')
plt.plot(t_range, compartments.death_rates)
plt.show()
