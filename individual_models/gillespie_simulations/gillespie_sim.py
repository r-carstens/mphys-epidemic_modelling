import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Initialising the system
N = 100

# Setting the initial conditions
I0, R0 = 1, 0
S0 = N - I0 - R0
t_max = 100

# Setting the epidemic parameters
gamma = 1/10
sigma = 0


##### NETWORK INITIALISATION

def get_mosquito_transmission():
    
    m = np.random.uniform(low=0, high=5)                # number of mosquitoes in the region per human
    a = np.random.uniform(low=0, high=0.5)              # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0, high=1)                # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0, high=1)                # transmission efficiency from humans to mosquitoes
    m_life_exp = np.random.uniform(low=1/10, high=1/7)  # life expectancy of mosquitoes

    return (m * a**2 * b * c) * m_life_exp


def initialise_graph(n_nodes):

    # Initialising the network structure so that each node is connected to all other nodes
    G = nx.complete_graph(n=n_nodes)

    # Looping through all edges
    for u, v in G.edges():

        # Adding weights to the edges to represent mosquito transmission
        G[u][v]['weight'] = get_mosquito_transmission()

    return G


def initialise_infections(G, n_infected):

    # Randomly infecting living nodes
    random_infected_nodes = np.random.choice(np.arange(G.number_of_nodes()), size=n_infected, replace=False)

    # Looping through all nodes
    for node in range(G.number_of_nodes()):

        # Checking if current node was chosen to be infected
        if node in random_infected_nodes:
            G.nodes()[node]['state'] = infected

        # Otherwise making node susceptible
        else:
            G.nodes()[node]['state'] = susceptible

    return G


##### DETERMINING THE RATES AT WHICH EVENTS OCCUR

def get_disease_populations(G):

    # Determining all node states with respect to the current disease
    susceptible_nodes = [node for node in G.nodes() if G.nodes()[node]['state'] == susceptible]
    infected_nodes = [node for node in G.nodes() if G.nodes()[node]['state'] == infected]
    recovered_nodes = [node for node in G.nodes() if G.nodes()[node]['state'] == immune]

    return susceptible_nodes, infected_nodes, recovered_nodes


def get_disease_recovery_data(infected_nodes):

    # Storing the potential recovery rates to the current disease
    return [gamma for infected_node in infected_nodes]


def get_disease_infection_data(G, susceptible_nodes, infected_nodes, reinfection):

    # Initialising arrays to store results
    sources, targets, infection_rates = [], [], []

    # Determining potential infections
    for sus_node in susceptible_nodes:

        # Finding infected neighbours and their transmission probabilities
        current_infected_nbs = set(G.neighbors(sus_node)) & set(infected_nodes)
        current_infection_rates = [G.get_edge_data(sus_node, nb)['weight'] for nb in current_infected_nbs]

        # Checking if a reinfection event (i.e. recovered individual loses immunity at rate sigma)
        if reinfection:
            current_infection_rates = [current_rate * sigma for current_rate in current_infection_rates]

        # Storing the results
        sources += current_infected_nbs
        targets += [sus_node for i in range(len(current_infected_nbs))]
        infection_rates += current_infection_rates

    return sources, targets, infection_rates


def get_event_rates(G):

    # Data to store event results (rates will be added for every possible disease by appending)
    sources, targets, rates, diseases = [], [], [], []

    # Sub-grouping into susceptible and infected nodes for the current disease
    S_nodes, I_nodes, R_nodes = get_disease_populations(G)

    # Determining all possible event data for the current disease
    recovery_rates = get_disease_recovery_data(I_nodes)
    infected_sources, susceptible_targets, infection_rates = get_disease_infection_data(G, S_nodes, I_nodes, reinfection=False)
    reinfected_sources, recovered_targets, recovered_infection_rates = get_disease_infection_data(G, R_nodes, I_nodes, reinfection=True)

    # Event sources include I_nodes which can recover and infected sources which can infect others
    sources += (I_nodes + infected_sources + reinfected_sources)

    # Event targets include I_nodes which can recover and infected sources which can infect others
    targets += (I_nodes + susceptible_targets + recovered_targets)

    # The I_nodes can recover or the susceptible targets can become infected
    rates += (recovery_rates + infection_rates + recovered_infection_rates)
    
    return sources, targets, rates


##### CHOOSING WHICH EVENT WILL OCCUR AND WHEN

def get_chosen_state_position(event_rates, sum_rates):

    # Determining the probabilities of each event and randomly choosing an event from a discrete probability function
    event_probabilities = event_rates / sum_rates
    random_position = np.random.choice(np.arange(len(event_probabilities)), size=1, p=event_probabilities)[0]

    return random_position


def get_chosen_time(sum_rates):

    # Generating a uniform random number and choosing timestep by drawing from exponential
    r1 = np.random.uniform()
    chosen_time = np.log(1.0 / r1) / sum_rates

    return chosen_time


def update_network(G, event_target):

    # Checking if the required disease state is susceptible and setting to infected
    if G.nodes()[event_target]['state'] == susceptible:
        G.nodes()[event_target]['state'] = infected

    # Checking if the required disease state is infected and setting to recovered
    elif G.nodes()[event_target]['state'] == infected:
        G.nodes()[event_target]['state'] = immune

    # Checking if the required disease state is recovered and setting to infected
    elif G.nodes()[event_target]['state'] == immune:
        G.nodes()[event_target]['state'] = infected

    return G


##### RUNNING THE SIMULATION

def get_current_totals(G):

    # Determining the population totals
    num_susceptible = np.sum([G.nodes()[node]['state'] == susceptible for node in G.nodes()])
    num_infected = np.sum([G.nodes()[node]['state'] == infected for node in G.nodes()])
    num_recovered = np.sum([G.nodes()[node]['state'] == immune for node in G.nodes()])

    return num_susceptible, num_infected, num_recovered


def run_gillespie(G, S0, I0, R0, t_max):

    # Creating a structure to store the totals
    S_totals, I_totals, M_totals, timesteps = [S0], [I0], [R0], [0]

    # Setting initial conditions
    t = 0

    while t < t_max:

        # Determining the possible events and their rates
        sources, targets, rates = get_event_rates(G)
        sum_rates = np.sum(rates)

        # Checking for convergence
        if len(rates) == 0:
            break

        # Choosing the event to occur by drawing from a discrete probability distribution
        chosen_position = get_chosen_state_position(rates, sum_rates)

        # Determining which event took place
        event_source = sources[chosen_position]  # node causing the event (e.g. I_j causes an infection to S_k)
        event_target = targets[chosen_position]  # node event is occurring to (e.g. S_k becomes infected)

        # Updating the system
        G = update_network(G, event_target)
        t += get_chosen_time(sum_rates)

        # Determining the current population totals
        S_total, I_total, M_total = get_current_totals(G)

        # Saving the results
        S_totals.append(S_total)
        I_totals.append(I_total)
        M_totals.append(M_total)
        timesteps.append(t)

        # Displaying progress
        print(t)

    return G, S_totals, I_totals, M_totals, timesteps


# Creating the graph and running the simulation
G = initialise_graph(N)
G = initialise_infections(G, I0)
G, S_totals, I_totals, M_totals, timesteps = run_gillespie(G, S0, I0, R0, t_max)

# Plotting the results
plt.title('Results from Gillespie Algorithm')
plt.plot(timesteps, S_totals, label='Susceptible')
plt.plot(timesteps, I_totals, label='Infectious')
plt.plot(timesteps, M_totals, label='Immune')
plt.xlabel('Time (days)')
plt.ylabel('Population Size')
plt.legend()
plt.show()