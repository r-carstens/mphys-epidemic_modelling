import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setting filenames to be used for storing and reading data
event_path = 'vector_event_outfile'
complete_path = 'complete_vector_event_data'

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting simulation data
N = 1000
I0 = 1
t_max = 100000
dt = 0.2

# Epidemiological Parameters
gamma = 0.1
sigma = 0

# Bottleneck dynamics
baseline_transmission = 0.4  # setting the baseline mosquito transmission probability
lam = 0.07                   # noise element of death rate evolution
nu = 1.0                     # death rate reversion to baseline rate
omega = 0.5                  # size of shock
kappa = 0.0002               # shock frequency


##### NETWORK INITIALISATION

def get_mosquito_transmission():
    a = np.random.uniform(low=0.1, high=2)   # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0.1, high=1)   # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0.1, high=1)   # transmission efficiency from humans to mosquitoes
    m = np.random.uniform(low=0, high=50)    # number of mosquitoes in the region per human
    mu = np.random.uniform(low=10, high=60)  # life expectancy of mosquitoes

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


##### SIMULATING CATASTROPHIC EVENTS

def get_event_times(sim_time):

    # Checking if an event occurs at each timestep
    return np.random.uniform(size=sim_time) < (kappa * dt)


def get_event_impact(sim_time, event_times, baseline_rate):

    # Initialising an array to store the event impact at each step
    event_impact = np.zeros(shape=sim_time)
    event_impact[0] = baseline_rate

    for t in range(sim_time - 1):

        # Determining the mosquito transmission at the current timestep
        check_for_event = int(event_times[t])

        # Determining the new transmission probability
        new_impact = dt * (lam * np.random.normal() - nu * (event_impact[t] - baseline_rate)) + check_for_event * omega

        # Updating the event impact and ensuring it lies between 0 and 1
        event_impact[t+1] = min(max(event_impact[t] + new_impact, 0), max(1, omega))

    return event_impact


##### SIMULATING DISEASE TRANSMISSION

def check_for_infection(G, source_label, target_label, event_impact, reinfection):

    infection_event = False

    # Determining the probability of transmission along the node edge
    p_mosquito_transmission = event_impact # * G[source_label][target_label]['weight']

    # Checking if the event is a reinfection (in which immunity loss is taken into account)
    if reinfection:
        p_mosquito_transmission *= sigma

    # Checking if an infection will occur
    if np.random.uniform() < p_mosquito_transmission:
        infection_event = True

    return infection_event


def check_for_recovery():

    # Checking if the infected individual has recovery with probability gamma
    return np.random.uniform() < gamma


def complete_step(G, event_impact):

    # Choosing two neighbours at random within the population
    all_nodes = np.arange(G.number_of_nodes())
    source_node, target_node = np.random.choice(all_nodes, size=2, replace=False)

    # Determining the states of the target and source
    target_before, source_during = G.nodes()[target_node]['state'], G.nodes()[source_node]['state']
    target_after = target_before

    if target_before == susceptible and source_during == infected:

        # Determining if the node being interacted with is infected and infection is transmitted
        if check_for_infection(G, source_label=source_node, target_label=target_node, event_impact=event_impact, reinfection=False):
            target_after = infected

    elif target_before == infected:

        # Checking for recovery
        if check_for_recovery():
            target_after = immune

    elif target_before == immune and source_during == infected:

        # Determining if the node being interacted with is infected and infection is transmitted
        if check_for_infection(G, source_label=source_node, target_label=target_node, event_impact=event_impact, reinfection=True):
            target_after = infected

    return G, source_node, target_node, source_during, target_before, target_after


def get_state_totals(G):

    # Counting the number of individuals within each state
    S_total = np.sum([G.nodes()[node]['state'] == susceptible for node in G.nodes()])
    I_total = np.sum([G.nodes()[node]['state'] == infected for node in G.nodes()])
    M_total = np.sum([G.nodes()[node]['state'] == immune for node in G.nodes()])

    return S_total, I_total, M_total


def run_simulation_iteration(G, n_nodes, I0, event_impact, sim_time, iter_num):

    # Creating a file to store the results to
    complete_outfile = open(complete_path + '_%s.txt' % (iter_num + 1), 'w')
    complete_outfile.write('N=%s,I0=%s,t_max=%s,gamma=%s,sigma=%s' % (n_nodes, I0, sim_time,gamma,sigma))
    complete_outfile.write('\ntimestep,source_label,target_label,source_during,target_before,target_after,S_total,I_total,M_total')

    # Looping through timesteps
    for t in tqdm(range(sim_time)):
        
        # Determining the current mosquito transmission potential
        current_event_impact = event_impact[t]

        # Completing an iteration step
        G, source_label, target_label, source_during, target_before, target_after = complete_step(G, event_impact=current_event_impact)

        # Updating network if required
        if target_before != target_after:
            G.nodes()[target_label]['state'] = target_after

        # Counting the number of individuals in each state
        S_total, I_total, M_total = get_state_totals(G)
    
        # Logging totals result
        complete_outfile.write('\n%s,%s,%s,%s,%s,%s,%s,%s,%s' % (
        t, source_label, target_label, source_during, target_before, target_after, S_total, I_total, M_total))

    # Closing the data file
    complete_outfile.close()


def repeat_simulation(N, I0, t_max, num_iterations=1):

    # Simulating when catastrophic events will occur
    event_times = get_event_times(sim_time=t_max)
    event_impact = get_event_impact(sim_time=t_max, event_times=event_times, baseline_rate=baseline_transmission)

    # Creating file to write catastrophic event data to
    with open(event_path + '.txt', 'w') as event_outfile:
        event_outfile.write('event_occurred,event_impact')

        # Writing catastrophic event data
        for event_data in list(zip(event_times, event_impact)):
            event_outfile.write('\n%s,%s' % (event_data))

    # Repeating entire simulation required number of times
    for n in range(num_iterations):

        # Initialising the simulation
        G = initialise_graph(n_nodes=N)
        G = initialise_infections(G, n_nodes=N, n_infected=I0)

        # Running the simulation
        run_simulation_iteration(G, n_nodes=N, I0=I0, event_impact=event_impact, sim_time=t_max, iter_num=n)


##### MAIN

# Repeating the simulation for a required number of iterations
n_iterations = 1
repeat_simulation(N=N, I0=I0, t_max=t_max, num_iterations=n_iterations)