import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Epidemiological Parameters
gamma = 0.1
sigma = 0


##### NETWORK INITIALISATION

def get_mosquito_transmission():

    a = np.random.uniform(low=0.1, high=2)     # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0.1, high=1)     # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0.1, high=1)     # transmission efficiency from humans to mosquitoes
    m = np.random.uniform(low=0, high=50)      # number of mosquitoes in the region per human
    mu = np.random.uniform(low=10, high=60)    # life expectancy of mosquitoes

    # return (a ** 2 * b * c * m) / mu
    return 0.5


def initialise_graph(n_nodes):

    # Initialising the network structure so that each node is connected to all other nodes
    G = nx.complete_graph(n=n_nodes)

    # Looping through all edges
    for u, v in G.edges():

        # Adding weights to the edges to represent mosquito transmission
        G[u][v]['weight'] = get_mosquito_transmission()

    return G


def initialise_infections(G, n_nodes, n_infected):

    # Creating a list of infection statuses
    node_states = np.array([susceptible for node in G.nodes()])

    # Randomly choosing infected individuals and setting those individuals to infected
    random_infected_nodes = np.random.choice(np.arange(n_nodes), size=n_infected, replace=False)
    node_states[random_infected_nodes] = infected

    # Adding node states
    for n in range(G.number_of_nodes()):
        G.nodes()[n]['state'] = node_states[n]

    return G


##### RUNNING THE SIMULATION

def check_for_infection(G, source_label, target_label):

    infection_event = False

    # Determining the probability of transmission along the node edge
    p_mosquito_transmission = G[source_label][target_label]['weight']

    # Checking if an infection will occur
    if np.random.uniform() < p_mosquito_transmission:
        infection_event = True

    return infection_event


def check_for_recovery():

    # Checking if the infected individual has recovery with probability gamma
    return np.random.uniform() < gamma


def complete_step(G):

    # Choosing two neighbours at random within the population
    source_label, target_label = np.random.choice(np.arange(G.number_of_nodes()), size=2, replace=False)

    # Determining the states of the target and source
    target_before, source = G.nodes()[target_label]['state'], G.nodes()[source_label]['state']
    target_after = target_before

    if target_before == susceptible and source == infected:

        # Determining if the node being interacted with is infected
        if check_for_infection(G, source_label, target_label):
            target_after = infected

    elif target_before == infected:

        # Checking for recovery
        if check_for_recovery():
            target_after = immune

    return G, target_label, source_label, target_before, target_after


def get_state_totals(G):

    # Counting the number of individuals within each state
    S_total = np.sum([G.nodes()[node]['state'] == susceptible for node in G.nodes()])
    I_total = np.sum([G.nodes()[node]['state'] == infected for node in G.nodes()])
    M_total = np.sum([G.nodes()[node]['state'] == immune for node in G.nodes()])

    return S_total, I_total, M_total


def run_simulation(G, sim_time, out_file):

    # Creating a file to store the results to
    infection_outfile = open(out_file, 'w')
    infection_outfile.write('timestep,target_label,source_label,target_before,target_after,total_S,total_I,total_M')

    for t in sim_time:

        # Completing an iteration step
        G, target_label, source_label, target_before, target_after = complete_step(G)

        # Updating network if required
        if target_before != target_after:
            G.nodes()[target_label]['state'] = target_after

        # Counting the number of individuals in each state
        S_total, I_total, M_total = get_state_totals(G)

        # Logging the results
        infection_outfile.write('\n%s,%s,%s,%s,%s,%s,%s,%s' % (
        t, target_label, source_label, target_before, target_after, S_total, I_total, M_total))

        # Displaying the progress
        if t % 1000 == 0:
            print(t)

    # Closing the file
    infection_outfile.close()


##### MAIN

# Setting simulation data
N = 1000
I0 = 100
t_data = np.arange(100000)
out_file_name = 'individual_sim_outfile.txt'

# Initialising the simulation
G = initialise_graph(n_nodes=N)
G = initialise_infections(G=G, n_nodes=N, n_infected=I0)

# Running the simulation
run_simulation(G=G, sim_time=t_data, out_file=out_file_name)

# Importing transmission data
df = pd.read_csv(out_file_name, delimiter=',')

# Plotting the results
plt.title('State Totals versus Time for Individual SIM Model')
plt.plot(df['timestep'], df['total_S'], label=susceptible)
plt.plot(df['timestep'], df['total_I'], label=infected)
plt.plot(df['timestep'], df['total_M'], label=immune)
plt.legend()
plt.show()





