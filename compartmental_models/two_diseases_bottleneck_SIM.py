import numpy as np
import matplotlib.pyplot as plt

########## BIRTH DYNAMICS

mu_B = 0.15       # birth rate
N_star = 1.0      # carrying capacity of system

########## BOTTLENECK DYNAMICS

mu_D_star = 0.09  # death rate baseline
lam = 0.07        # noise element of death rate evolution
nu = 1.0          # death rate reversion to baseline rate
omega = 1         # size of shock
kappa = 0.01      # shock frequency

########## EPIDEMIOLOGICAL DYNAMICS

# Disease 1
beta_1 = 3        # infection rate
gamma_1 = 0.4     # recovery (to partial immunity)
sigma_1 = 0.6     # "breakthrough" rate of partial immunity (between 0 and 1)

# Disease 2
beta_2 = 2.5      # infection rate
gamma_2 = 0.5     # recovery (to partial immunity)
sigma_2 = 0.8     # "breakthrough" rate of partial immunity (between 0 and 1)


########## CLASS TO STORE COMPARTMENTS
class Compartments:

    def __init__(self, initial_conditions, mu_D_star, t_max, dt):

        # Simulation duration
        self.dt = dt
        self.t_range = np.arange(start=0, stop=t_max, step=self.dt)

        # Unpacking the initial compartment conditions
        num_S1S2, num_I1S2, num_M1S2, num_S1I2, num_I1I2, num_M1I2, num_S1M2, num_I1M2, num_M1M2 = initial_conditions

        # Possible compartments
        self.S1S2_data = np.zeros(shape=self.t_range.shape)
        self.S1S2_data[0] = num_S1S2

        self.I1S2_data = np.zeros(shape=self.t_range.shape)
        self.I1S2_data[0] = num_I1S2

        self.M1S2_data = np.zeros(shape=self.t_range.shape)
        self.M1S2_data[0] = num_M1S2

        self.S1I2_data = np.zeros(shape=self.t_range.shape)
        self.S1I2_data[0] = num_S1I2

        self.I1I2_data = np.zeros(shape=self.t_range.shape)
        self.I1I2_data[0] = num_I1I2

        self.M1I2_data = np.zeros(shape=self.t_range.shape)
        self.M1I2_data[0] = num_M1I2

        self.S1M2_data = np.zeros(shape=self.t_range.shape)
        self.S1M2_data[0] = num_S1M2

        self.I1M2_data = np.zeros(shape=self.t_range.shape)
        self.I1M2_data[0] = num_I1M2

        self.M1M2_data = np.zeros(shape=self.t_range.shape)
        self.M1M2_data[0] = num_M1M2

        # Death rate data
        self.death_rates = np.zeros(shape=self.t_range.shape)
        self.death_rates[0] = mu_D_star

    def euler_step(self, i):

        # Accessing the current compartment sizes
        S1S2, I1S2, M1S2, S1I2, I1I2, M1I2, S1M2, I1M2, M1M2 = self.S1S2_data[i], self.I1S2_data[i], self.M1S2_data[i], \
                                                               self.S1I2_data[i], self.I1I2_data[i], self.M1I2_data[i], \
                                                               self.S1M2_data[i], self.I1M2_data[i], self.M1M2_data[i]

        # Determining the current population size
        N_i = S1S2 + I1S2 + M1S2 + S1I2 + I1I2 + M1I2 + S1M2 + I1M2 + M1M2

        # Determining the current death rate
        mu_D_i = self.death_rates[i]

        # Updating the S1S2 compartment (susceptible to both diseases)
        dS1S2_i = self.dt * (mu_B * N_i * (1 - N_i / N_star)                   # birth (assumes total susceptibility)
                             - beta_1 * S1S2 * (I1S2 + I1I2 + I1M2)            # infection by disease 1
                             - beta_2 * S1S2 * (S1I2 + I1I2 + M1I2)            # infection by disease 2
                             - mu_D_i * S1S2)                                  # death

        # Updating the I1S2 compartment (infected by 1, susceptible to 2)
        dI1S2_i = self.dt * (beta_1 * S1S2 * (I1S2 + I1I2 + I1M2)              # full infection by 1
                             + sigma_1 * beta_1 * M1S2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             - gamma_1 * I1S2                                  # recovery from 1
                             - beta_2 * I1S2 * (S1I2 + I1I2 + M1I2)            # full infection by 2
                             - mu_D_i * I1S2)                                  # death

        # Updating the M1S2 compartment (immune to 1, susceptible to 2)
        dM1S2_i = self.dt * (- sigma_1 * beta_1 * M1S2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             + gamma_1 * I1S2                                  # recovery from 1
                             - beta_1 * M1S2 * (I1S2 + I1I2 + I1M2)            # full infection by 2
                             - mu_D_i * M1S2)                                  # death

        # Updating the S1I2 compartment (susceptible to 1, immune to 2)
        dS1I2_i = self.dt * (- beta_1 * S1I2 * (I1S2 + I1I2 + I1M2)            # full infection by 1
                             + beta_2 * S1S2 * (S1I2 + I1I2 + M1I2)            # full infection by 2
                             + sigma_2 * beta_2 * S1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             - gamma_2 * S1I2                                  # recovery from 2
                             - mu_D_i * S1I2)                                  # death

        # Updating the I1I2 compartment (infected by both)
        dI1I2_i = self.dt * (beta_1 * S1I2 * (I1S2 + I1I2 + I1M2)              # full infection by 1
                             + beta_1 * sigma_1 * M1I2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             - gamma_1 * I1I2                                  # recovery from 1
                             + beta_2 * I1S2 * (S1I2 + I1I2 + M1I2)            # full infection by 2
                             - sigma_2 * beta_2 * I1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             - gamma_2 * I1I2                                  # recovery from 2
                             - mu_D_i * I1I2)                                  # death

        # Updating the M1I2 compartment (immune to 1, infected by 2)
        dM1I2_i = self.dt * (- sigma_1 * beta_1 * M1I2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             + gamma_1 * I1I2                                  # recovery from 1
                             + beta_2 * M1S2 * (S1I2 + I1I2 + M1I2)            # infection by 2
                             - sigma_2 * beta_2 * M1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             - gamma_2 * M1I2                                  # recovery from 2
                             - mu_D_i * M1I2)                                  # death

        # Updating the S1M2 compartment (susceptible to 1, immune to 2)
        dS1M2_i = self.dt * (- beta_1 * S1M2 * (I1S2 + I1I2 + I1M2)            # full infection by 1
                             - sigma_2 * beta_2 * S1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             + gamma_2 * S1I2                                  # recovery from 2
                             - mu_D_i * S1M2)                                  # death

        # Updating the I1M2 compartment (infected by 1, immune to 2)
        dI1M2_i = self.dt * (beta_1 * S1M2 * (I1S2 + I1I2 + I1M2)              # full infection by 1
                             + sigma_1 * beta_1 * M1M2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             - gamma_1 * I1M2                                  # recovery from 1
                             - sigma_2 * beta_2 * I1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             + gamma_2 * I1I2                                  # recovery from 2
                             - mu_D_i * I1M2)                                  # death

        # Updating the M1M2 compartment (immune to both)
        dM1M2_i = self.dt * (- sigma_1 * beta_1 * M1M2 * (I1S2 + I1I2 + I1M2)  # partial infection by 1
                             + gamma_1 * I1M2                                  # recovery from 1
                             - sigma_2 * beta_2 * M1M2 * (S1I2 + I1I2 + M1I2)  # partial infection by 2
                             + gamma_2 * M1I2                                  # recovery from 2
                             - mu_D_i * M1M2)                                  # death

        # Updating the compartments
        self.S1S2_data[i + 1] = S1S2 + dS1S2_i
        self.I1S2_data[i + 1] = I1S2 + dI1S2_i
        self.M1S2_data[i + 1] = M1S2 + dM1S2_i
        self.S1I2_data[i + 1] = S1I2 + dS1I2_i
        self.I1I2_data[i + 1] = I1I2 + dI1I2_i
        self.M1I2_data[i + 1] = M1I2 + dM1I2_i
        self.S1M2_data[i + 1] = S1M2 + dS1M2_i
        self.I1M2_data[i + 1] = I1M2 + dI1M2_i
        self.M1M2_data[i + 1] = M1M2 + dM1M2_i

        # Determining the change in the current death rate
        check_for_shock = int(np.random.uniform() < (kappa * dt))
        dp_death_rate = dt * (lam * np.random.normal() - nu * (mu_D_i - mu_D_star)) + check_for_shock * omega

        # Updating the death rate
        self.death_rates[i + 1] = min(max(mu_D_i + dp_death_rate, 0), max(1, omega))

    def run_sim(self):

        # Euler update at every time step
        for i in range(len(self.t_range) - 1):
            self.euler_step(i)


########## POPULATION DATA
N = 1
I1S2 = 0.001
S1S2 = N - I1S2

########## INITIAL COMPARTMENT SIZES
initial_conditions = np.zeros(shape=9, dtype=float)  # S1S2, I1S2, M1S2, S1I2, I1I2, M1I2, S1M2, I1M2, M1M2
initial_conditions[0] = S1S2
initial_conditions[1] = I1S2

########## TIME DATA
t_max = 250
dt = 0.2

########## RUNNING SIMULATION
multiple_diseases = Compartments(initial_conditions, mu_D_star, t_max, dt)
multiple_diseases.run_sim()

########## PLOTTING RESULTS
plt.clf()
plt.plot(multiple_diseases.t_range, multiple_diseases.S1S2_data, label='S1S1')  # susceptible to both disease 1 and 2
plt.plot(multiple_diseases.t_range, multiple_diseases.I1S2_data, label='I1S2')  # infected by 1, susceptible to 2
plt.plot(multiple_diseases.t_range, multiple_diseases.M1S2_data, label='M1S2')  # immune to 1, susceptible to 2
plt.plot(multiple_diseases.t_range, multiple_diseases.S1I2_data, label='S1I2')  # susceptible to 1, infected by 2
plt.plot(multiple_diseases.t_range, multiple_diseases.I1I2_data, label='I1I2')  # immune to both disease 1 and 2
plt.plot(multiple_diseases.t_range, multiple_diseases.M1I2_data, label='M1I2')  # immune to 1, infected by 2
plt.plot(multiple_diseases.t_range, multiple_diseases.S1M2_data, label='S1M2')  # susceptible to 1, immune to 2
plt.plot(multiple_diseases.t_range, multiple_diseases.I1M2_data, label='I1M2')  # infected by 1, immune to 2
plt.plot(multiple_diseases.t_range, multiple_diseases.M1M2_data, label='M1M2')  # immune to both disease 1 and 2
plt.title('Evolution of compartment sizes over time for SIM compartmental model \nwith two diseases')
plt.xlabel('Time')
plt.ylabel('Total')
plt.legend()
plt.show()

plt.clf()
plt.title('Evolution of death rates over time for SIM compartmental model \nwith two diseases')
plt.xlabel('Time')
plt.ylabel('Death Rate')
plt.plot(multiple_diseases.t_range, multiple_diseases.death_rates)
plt.show()
