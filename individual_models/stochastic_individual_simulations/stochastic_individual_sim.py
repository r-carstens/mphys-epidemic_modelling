import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


################################################## SIMULATION CONSTANTS

gammas = [1/10, 1/5]  # human recovery times for diseases
sigmas = [0.5, 0]  # immunity breakthrough rates from diseases

susceptible = 'S'
infected = 'I'
recovered = 'R'


################################################## INITIALISING GRAPH

def get_mosquito_transmission_weight():
    a = np.random.uniform(low=0.1, high=2)  # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0.1, high=1)  # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0.1, high=1)  # transmission efficiency from humans to mosquitoes
    m = np.random.uniform(low=0, high=50)  # number of mosquitoes in the region per human
    mu = np.random.uniform(low=10, high=60)  # life expectancy of mosquitoes

    # return (a ** 2 * b * c * m) / mu
    return 0.5


def get_graph_nodes(G, all_nodes, num_diseases):

    # Randomly choosing infected nodes (currently only 1 infection per disease in the system)
    random_infected_nodes = np.random.choice(all_nodes, size=num_diseases, replace=True)

    for node in all_nodes:

        # Creating a structure to store each node's states with respect to the diseases
        initial_states = np.array([susceptible for i in range(num_diseases)])

        # Checking if the node is infected with any disease and altering the initial states to include the infection(s)
        if len(np.where(node == random_infected_nodes)[0]) > 0:
            initial_states[np.where(node == random_infected_nodes)] = infected

        # Adding the node
        G.add_node(node, states=initial_states)

    return G


def get_graph_edges(G):

    # Looping through all possible edges
    for patch in G.nodes():
        for other_patch in G.nodes():

            # Connecting node to all other nodes except itself and setting distance between
            if patch != other_patch:
                G.add_edge(patch, other_patch, weight=get_mosquito_transmission_weight())

    return G


def initialise_graph(num_nodes, num_diseases):

    # Initialising the network structure and the node labels
    G = nx.Graph()
    all_nodes = np.arange(num_nodes)

    # Creating the graph (these are functions to allow for more graphs to be easily constructured)
    G = get_graph_nodes(G, all_nodes, num_diseases)
    G = get_graph_edges(G)

    return G


##### DETERMINING THE RATES AT WHICH EVENTS OCCUR

def get_disease_populations(G, disease):

    # Determining all node states with respect to the current disease
    susceptible_nodes = [node for node in G.nodes() if susceptible in G.nodes()[node]['states'][disease]]
    infected_nodes = [node for node in G.nodes() if infected in G.nodes()[node]['states'][disease]]
    recovered_nodes = [node for node in G.nodes() if recovered in G.nodes()[node]['states'][disease]]

    return susceptible_nodes, infected_nodes, recovered_nodes


def get_disease_recovery_data(infected_nodes, disease):

    # Determining potential recovery rates to the current disease
    return [gammas[disease] for infected_node in infected_nodes]


def get_disease_infection_data(G, susceptible_nodes, infected_nodes, disease, reinfection):

    sources, targets, infection_rates = [], [], []

    # Determining potential infections
    for sus_node in susceptible_nodes:

        # Finding infected neighbours and their transmission probabilities
        current_infected_nbs = set(G.neighbors(sus_node)) & set(infected_nodes)
        current_infection_rates = [G.get_edge_data(sus_node, nb)['weight'] for nb in current_infected_nbs]

        # Checking if a reinfection event (i.e. recovered individual loses immunity at rate sigma)
        if reinfection:
            current_infection_rates = [current_rate * sigmas[disease] for current_rate in current_infection_rates]

        # Storing the results
        sources += current_infected_nbs
        targets += [sus_node for i in range(len(current_infected_nbs))]
        infection_rates += current_infection_rates

    return sources, targets, infection_rates


def get_event_rates(G, num_diseases):

    # Data to store event results (rates will be added for every possible disease by appending)
    sources, targets, rates, diseases = [], [], [], []

    # Determining event data for all diseases
    for disease in range(num_diseases):

        # Sub-grouping into susceptible and infected nodes for the current disease
        S_nodes, I_nodes, R_nodes = get_disease_populations(G, disease)

        # Determining all possible event data for the current disease
        recovery_rates = get_disease_recovery_data(I_nodes, disease)
        infected_sources, susceptible_targets, infection_rates = get_disease_infection_data(G, S_nodes, I_nodes, disease, reinfection=False)
        reinfected_sources, recovered_targets, recovered_infection_rates = get_disease_infection_data(G, R_nodes, I_nodes, disease, reinfection=True)

        # Event sources include I_nodes which can recover and infected sources which can infect others
        sources += (I_nodes + infected_sources + reinfected_sources)

        # Event targets include I_nodes which can recover and infected sources which can infect others
        targets += (I_nodes + susceptible_targets + recovered_targets)

        # The I_nodes can recover or the susceptible targets can become infected
        rates += (recovery_rates + infection_rates + recovered_infection_rates)

        # Adding the specific disease involved
        diseases += [disease for i in range(len(recovery_rates) + len(infection_rates) + len(recovered_infection_rates))]

    return sources, targets, rates, diseases


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


def update_network(G, event_target, event_disease):

    # Checking if the required disease state is susceptible and setting to infected
    if G.nodes()[event_target]['states'][event_disease] == susceptible:
        G.nodes()[event_target]['states'][event_disease] = infected

    # Checking if the required disease state is infected and setting to recovered
    elif G.nodes()[event_target]['states'][event_disease] == infected:
        G.nodes()[event_target]['states'][event_disease] = recovered

    # Checking if the required disease state is recovered and setting to infected
    elif G.nodes()[event_target]['states'][event_disease] == recovered:
        G.nodes()[event_target]['states'][event_disease] = infected

    return G


def get_current_totals(G, num_diseases):

    # Creating structures to store the current totals
    current_data = np.zeros(shape=(num_diseases, 3))

    # Looping through each possible disease
    for disease in range(num_diseases):

        # Determining the population totals
        num_susceptible = np.sum([G.nodes()[node]['states'][disease] == susceptible for node in G.nodes()])
        num_infected = np.sum([G.nodes()[node]['states'][disease] == infected for node in G.nodes()])
        num_recovered = np.sum([G.nodes()[node]['states'][disease] == recovered for node in G.nodes()])

        # Storing the current results
        current_data[disease] = num_susceptible, num_infected, num_recovered

    return current_data.reshape(num_diseases * 3)


##### RUNNING THE SIMULATION

def run_gillespie(G, num_diseases, t_max, file_name):

    # Creating file to store data
    out_file = open(file_name, 'w')
    out_file.write('time,' + ','.join([f'S_{disease},I_{disease},R_{disease}' for disease in range(num_diseases)]) + '\n')

    # Setting initial conditions
    t = 0

    while t < t_max:

        # Determining the possible events and their rates
        sources, targets, rates, diseases = get_event_rates(G, num_diseases)
        sum_rates = np.sum(rates)

        # Checking for convergence
        if len(rates) == 0:
            break

        # Choosing the event to occur by drawing from a discrete probability distribution
        chosen_position = get_chosen_state_position(rates, sum_rates)

        # Determining which event took place
        event_source = sources[chosen_position]  # node causing the event (e.g. I_j causes an infection to S_k)
        event_target = targets[chosen_position]  # node event is occurring to (e.g. S_k becomes infected)
        event_disease = diseases[chosen_position]  # specific disease involved

        # Updating the system
        G = update_network(G, event_target, event_disease)
        t += get_chosen_time(sum_rates)

        # Storing the results
        population_results = get_current_totals(G, num_diseases)
        out_file.write(str(t) + ',' + ','.join(population_results.astype(str)) + '\n')

        # Displaying progress
        print(t)

    out_file.close()
    return G


# Initialising the system
N = 100
num_diseases = 2

# Setting the initial conditions
I0, R0 = 1, 0
S0 = N - I0 - R0
t_max = 100

# Creating file to store data
file_name = 'stochastic_SIM_%s_diseases_data.txt' % num_diseases if num_diseases > 1 else 'stochastic_SIM_disease_data.txt'

# Creating the graph and running the simulation
G = initialise_graph(N, num_diseases)
G = run_gillespie(G, num_diseases, t_max, file_name)

# Reading the data
data = pd.read_csv(file_name)

# Plotting the data
for i in range(1, len(data.columns)):
    plt.plot(data[data.columns[0]], data[data.columns[i]], label=data.columns[i])

if num_diseases > 1:
    plt.title('Evolution of population sizes over time for an individual-based model \nwith %s diseases' % num_diseases)

else:
    plt.title('Evolution of population sizes over time for an individual-based model \nwith %s disease' % num_diseases)

plt.xlabel('Time (days)')
plt.ylabel('Population Size')
plt.legend()
plt.show()