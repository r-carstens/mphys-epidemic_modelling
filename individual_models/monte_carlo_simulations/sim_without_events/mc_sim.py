import os
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

def check_for_infection(G, source_label, target_label, reinfection):

    infection_event = False

    # Determining the probability of transmission along the node edge
    p_mosquito_transmission = G[source_label][target_label]['weight']

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


def complete_step(G):

    # Choosing two neighbours at random within the population
    source_node, target_node = np.random.choice(np.arange(G.number_of_nodes()), size=2, replace=False)

    # Determining the states of the target and source
    target_before, source_during = G.nodes()[target_node]['state'], G.nodes()[source_node]['state']
    target_after = target_before

    if target_before == susceptible and source_during == infected:

        # Determining if the node being interacted with is infected and infection is transmitted
        if check_for_infection(G, source_label=source_node, target_label=target_node, reinfection=False):
            target_after = infected

    elif target_before == infected:

        # Checking for recovery
        if check_for_recovery():
            target_after = immune

    elif target_before == immune and source_during == infected:

        # Determining if the node being interacted with is infected and infection is transmitted
        if check_for_infection(G, source_label=source_node, target_label=target_node, reinfection=True):
            target_after = infected

    return G, source_node, target_node, source_during, target_before, target_after


def get_state_totals(G):

    # Counting the number of individuals within each state
    S_total = np.sum([G.nodes()[node]['state'] == susceptible for node in G.nodes()])
    I_total = np.sum([G.nodes()[node]['state'] == infected for node in G.nodes()])
    M_total = np.sum([G.nodes()[node]['state'] == immune for node in G.nodes()])

    return S_total, I_total, M_total


def run_simulation_iteration(G, n_nodes, I0, sim_time, out_file_name):

    # Creating a file to store the results to
    infection_outfile = open(out_file_name, 'w')
    infection_outfile.write('N=%s, I0=%s, t_max=%s, gamma=%s, sigma=%s' % (n_nodes, I0, sim_time,gamma,sigma))
    infection_outfile.write('\ntimestep,source_label,target_label,source_during,target_before,target_after,S_total,I_total,M_total')

    for t in range(sim_time):

        # Completing an iteration step
        G, source_label, target_label, source_during, target_before, target_after = complete_step(G)

        # Updating network if required
        if target_before != target_after:
            G.nodes()[target_label]['state'] = target_after

        # Counting the number of individuals in each state
        S_total, I_total, M_total = get_state_totals(G)

        # Logging the results
        infection_outfile.write('\n%s,%s,%s,%s,%s,%s,%s,%s,%s' % (
        t, source_label, target_label, source_during, target_before, target_after, S_total, I_total, M_total))

        # Displaying the progress
        if t % 1000 == 0:
            print(t)

    # Closing the file
    infection_outfile.close()


def repeat_simulation(N, I0, t_max, out_file_prefix, num_iterations=1):

    # Repeating entire simulation required number of times
    for n in range(num_iterations):

        # Creating a new outfile name
        current_file_name = out_file_prefix + '_%s.txt' % (n + 1)

        # Initialising the simulation
        G = initialise_graph(n_nodes=N)
        G = initialise_infections(G, n_nodes=N, n_infected=I0)

        # Running the simulation
        run_simulation_iteration(G, n_nodes=N, I0=I0, sim_time=t_max, out_file_name=current_file_name)


##### ANALYSING AND DISPLAYING DATA

def get_results_dataframe(in_path):

    # Creating a dataframe to store all results from all files
    results_df = pd.DataFrame()

    # Locating all data files within the directory
    sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(in_path)]

    # Looping through each data file
    for counter, file in enumerate(sim_data_files):

        # Reading in the data
        current_df = pd.read_csv(file, delimiter=',', skiprows=1)

        # Changing column names to match their origin file (i.e. 'column_name_number')
        new_col_names = [current_df.columns[0]] + [col_name + '_%s' % counter for col_name in current_df.columns[1:]]
        current_df.columns = new_col_names

        # Concatenating the current results to the results dataframe
        results_df = pd.concat([results_df, current_df], axis=1)

    return results_df


def get_state_dataframes(results_df):

    # Extracting the state data
    susceptible_df = results_df.filter(regex='S_total')
    infected_df = results_df.filter(regex='I_total')
    immune_df = results_df.filter(regex='M_total')

    # Adding mean data
    susceptible_df = susceptible_df.assign(S_total_mean=susceptible_df.mean(axis=1))
    infected_df = infected_df.assign(I_total_mean=infected_df.mean(axis=1))
    immune_df = immune_df.assign(M_total_mean=immune_df.mean(axis=1))

    return susceptible_df, infected_df, immune_df


def plot_state_totals(susceptible_df, infected_df, immune_df):

    # Creating a list of the dataframes
    all_state_dfs = susceptible_df, infected_df, immune_df

    # Creating a list for legend labels and plot colors
    labels = ['Susceptible', 'Infectious', 'Immune']
    colors = ['navy', 'firebrick', 'g']

    # Initialising a figure
    fig, ax = plt.subplots()

    # Looping through the remaining results
    for counter, df in enumerate(all_state_dfs):

        # Plotting the rough data with transparent lines
        df[df.columns[:-1]].plot(ax=ax, alpha=0.1, color=colors[counter], legend=False)

        # Plotting the mean data with a solid line
        df[df.columns[-1]].plot(ax=ax, color=colors[counter], linewidth=2, label=labels[counter], legend=True)

    # Adding plot titles and legend
    plt.title('Population sizes versus time for individual-based SIM model\nwith gamma=%s and sigma=%s' % (gamma, sigma))
    plt.xlabel('Time (days)')
    plt.ylabel('Population sizes')
    plt.show()


##### MAIN

# Setting simulation data
N = 1000
I0 = 1
t_max = 100000

# Creating out file name to be used for storing and reading data
data_path = 'individual_sim_outfile'

# Repeating the simulation
repeat_simulation(N=N, I0=I0, t_max=t_max, out_file_prefix=data_path, num_iterations=1)

# Analysing the data
results_df = get_results_dataframe(in_path=data_path)
susceptible_df, infected_df, immune_df = get_state_dataframes(results_df=results_df)

# Plotting the data
plot_state_totals(susceptible_df=susceptible_df, infected_df=infected_df, immune_df=immune_df)
