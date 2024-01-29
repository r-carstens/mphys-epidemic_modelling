import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


################################################## SIMULATION CONSTANTS

gammas = [1/14, 1/10] # human recovery times for diseases
sigmas = [0, 0]       # immunity breakthrough rates from diseases

connection_prob = 0.3 # probability that a node is connected to another node


################################################## STATES

susceptible = 'S'
infected = 'I'
recovered = 'R'


################################################## INITIALISING GRAPH

def get_mosquito_transmission_weights():

    # Uses the concept of vectorial capacity to determine the likelihood of a transmission between nodes
    a = np.random.uniform(low=0.1, high=2)       # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0.5, high=0.5)     # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0.5, high=0.5)     # transmission efficiency from humans to mosquitoes
    m = np.random.uniform(low=0, high=5)        # number of mosquitoes in the region per human
    mu = np.random.uniform(low=0.14, high=0.23)  # life expectancy of mosquitoes

    # This will be used as a weight between nodes, giving the likelihood of transmission between humans along that node
    return (a ** 2 * b * c * m) / mu


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
            if patch != other_patch and np.random.uniform() < connection_prob:
                G.add_edge(patch, other_patch, weight=get_mosquito_transmission_weights())
                
    return G


def initialise_graph(num_nodes, num_diseases):

    # Initialising the network structure and the node labels
    G = nx.Graph()
    all_nodes = np.arange(num_nodes)
    
    # Creating the graph (these are functions to allow for more graphs to be easily constructured)
    G = get_graph_nodes(G, all_nodes, num_diseases)
    G = get_graph_edges(G)
                    
    return G


################################################## RUNNING SIMULATION

def get_disease_populations(G, disease):
    
    # Determining all nodes susceptible to and infected by the current disease
    susceptible_nodes = [node_label for node_label, node_data in G.nodes(data=True) if susceptible in node_data['states'][disease]]
    infected_nodes = [node_label for node_label, node_data in G.nodes(data=True) if infected in node_data['states'][disease]]

    return susceptible_nodes, infected_nodes
    
    
def get_disease_recovery_data(G, infected_nodes, disease):
    
    # Determining potential recoveries
    recovery_rates = [gammas[disease] for infected_node in infected_nodes]
    
    return recovery_rates


def get_disease_infection_data(G, susceptible_nodes, infected_nodes, disease):
    
    sources, targets, infection_rates = [], [], []
    
    # Determining potential infections
    for sus_node in susceptible_nodes:
            
        # Finding infected neighbours and their transmission probabilities
        current_infected_nbs = set(G.neighbors(sus_node)) & set(infected_nodes)
        current_infection_rates = [G.get_edge_data(sus_node, infected_nb)['weight'] for infected_nb in current_infected_nbs]
        
        # Storing the results
        sources += current_infected_nbs
        targets += [sus_node for i in range(len(current_infected_nbs))]
        infection_rates += current_infection_rates
        
    return sources, targets, infection_rates


def get_event_rates(G, num_diseases):
    
    # Data to store event results
    sources, targets, rates, diseases = [], [], [], []
    
    # Determining event data for all diseases
    for disease in range(num_diseases):

        # Subgrouping into susceptible and infected nodes for the current disease
        susceptible_nodes, infected_nodes = get_disease_populations(G, disease)
        
        # Determining all possible event data for the current disease
        disease_recovery_rates = get_disease_recovery_data(G, infected_nodes, disease)
        disease_infected_sources, disease_susceptible_targets, disease_infection_rates = get_disease_infection_data(G, susceptible_nodes, infected_nodes, disease)

        # Storing the results
        sources += (infected_nodes + disease_infected_sources)
        targets += (infected_nodes + disease_susceptible_targets)
        rates += (disease_recovery_rates + disease_infection_rates)
        diseases += [disease for i in range(len(disease_recovery_rates) + len(disease_infection_rates))]
        
    return sources, targets, rates, diseases
    
    
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


def update_network(G, event_source, event_target, event_disease):
            
    # Checking if the required disease state is susceptible
    if G.nodes()[event_target]['states'][event_disease] == susceptible:
        
        # Setting to infected
        G.nodes()[event_target]['states'][event_disease] = infected
        
    # Checking if the required disease state is infected
    elif G.nodes()[event_target]['states'][event_disease] == infected:
        
        # Setting the state to recovered
        G.nodes()[event_target]['states'][event_disease] = recovered
                
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
                                 
                                   
def run_gillespie(G, num_diseases, t_max, file_name):
    
    # Creating file to store data
    out_file = open(file_name, 'w')
    out_file.write('Time,' + ','.join([f'S_{disease},I_{disease},R_{disease}' for disease in range(num_diseases)]) + '\n')
    
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
        event_source = sources[chosen_position]    # node causing the event (e.g. I_j causes an infection to S_k)
        event_target = targets[chosen_position]    # node event is occurring to (e.g. S_k becomes infected)
        event_disease = diseases[chosen_position]  # specific disease involved
        
        # Updating the system
        G = update_network(G, event_source, event_target, event_disease)
        t += get_chosen_time(sum_rates)

        # Storing the results
        population_results = get_current_totals(G, num_diseases)
        out_file.write(str(t) + ',' + ','.join(population_results.astype(str)) + '\n')

    out_file.close()
    return G


# Initialising the system
N = 1000
num_diseases = 2

# Setting the initial conditions
I0, R0 = 1, 0
S0 = N - I0 - R0
t_max = 100

# Creating file to store data
file_name = 'individual_SIM_%s_diseases_data.txt' % num_diseases if num_diseases > 1 else 'individual_SIM_disease_data.txt'

# Creating the graph and running the simulation
G = initialise_graph(N, num_diseases)
G = run_gillespie(G, num_diseases, t_max, file_name)

# Reading the data
data = pd.read_csv(file_name)
data.columns

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
