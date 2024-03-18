import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


# Setting filename to be used for storing and reading data
mc_file_path = 'sim_with_hosts_mc'
totals_file_path = 'sim_with_hosts_totals'
event_path = 'host_events'

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting population data
N = 500
N_alive = int(1 * N)
I0 = 1

# Setting simulation data
n_iterations = 20
t_max, dt = 100, 1

# Setting epidemiological Parameters
gamma = 1/7
sigma = 0

# Setting vital parameters
mu_B = 0.07
mu_D = 0.04


##### SIMULATING CATASTROPHIC EVENTS

def get_event_times(sim_time, kappa_val):

    # Checking if an event occurs at each timestep
    return np.random.uniform(size=sim_time) < (kappa_val * dt)


def get_event_impact(sim_time, event_times, omega_val):

    # Initialising an array to store the event impact at each step and increasing rate at step
    event_impact = np.ones(shape=sim_time) * mu_D
    event_impact[event_times] += omega_val

    # Ensuring all rates are reasonable
    event_impact[event_impact > 1] = 1
    event_impact[event_impact < 0] = 0
    
    return event_impact


##### NETWORK INITIALISATION


def get_mosquito_transmission_rate():

    m = np.random.uniform(low=0, high=20)             # number of mosquitoes in the region per human
    a = np.random.uniform(low=0, high=5)             # rate at which a human is bitten by a mosquito
    b = np.random.uniform(low=0, high=1)              # proportion of infected bites that cause infection in the human host
    c = np.random.uniform(low=0, high=1)              # transmission efficiency from humans to mosquitoes
    life_exp = np.random.uniform(low=1/21, high=1/7)  # life expectancy of mosquitoes

    return (m * a**2 * b * c) * life_exp


def get_mosquito_transmission_probabilitity(rate):

    # Assuming exponentially-distributed
    return 1 - np.exp(-rate * dt)


def initialise_graph(n_nodes, n_alive):

    # Initialising the network structure so that each node is connected to all other nodes
    G = nx.complete_graph(n=n_nodes)

    # Randomly choosing n_alive nodes and letting the remaining nodes be initially dead
    alive_nodes = np.random.choice(np.arange(n_nodes), size=n_alive, replace=False)
    dead_nodes = np.setdiff1d(np.arange(n_nodes), alive_nodes)

    # Creating a dictionary to store the node vital signs
    node_vital_signs = {a_node: {'vitals': 'alive'} for a_node in alive_nodes}
    node_vital_signs.update({d_node: {'vitals': 'dead'} for d_node in dead_nodes})

    # Setting the node vital dynamics
    nx.set_node_attributes(G, node_vital_signs)

    # Looping through all edges
    for u, v in G.edges():

        # Determining the mosquito transmission rate and probability
        m_rate = get_mosquito_transmission_rate()
        m_prob = get_mosquito_transmission_probabilitity(m_rate)

        # Adding weights to the edges to represent mosquito transmission
        G[u][v]['weight'] = m_prob

    return G


def initialise_infections(G, n_nodes, n_infected):

    # Determining all living nodes
    living_nodes = np.array([node for node in G.nodes() if G.nodes()[node]['vitals'] == 'alive'])

    # Randomly infecting living nodes
    random_infected_nodes = np.random.choice(living_nodes, size=n_infected, replace=False)

    # Looping through all nodes
    for node in range(G.number_of_nodes()):

        # Checking if current node was chosen to be infected
        if node in random_infected_nodes:
            G.nodes()[node]['state'] = infected

        # Otherwise making node susceptible
        else:
            G.nodes()[node]['state'] = susceptible

    return G


##### RUNNING THE SIMULATION

def get_potential_node_birth(G, node_label):

    # Determining the current birth probability
    p_birth = 1 - np.exp(-mu_B * dt)

    # Determining whether a birth occurred
    has_birth_occurred = np.random.uniform() < p_birth

    # Updating node if birth occurred
    if has_birth_occurred:
        G.nodes()[node_label]['vitals'] = 'alive'
        G.nodes()[node_label]['state'] = susceptible
        
    return G, int(has_birth_occurred)


def get_potential_node_death(G, node_label, event_impact):

    # Determining the current death probability
    p_death = 1 - np.exp(-event_impact * dt)

    # Determining whether a death occurred
    has_death_occurred = np.random.uniform() < p_death

    # Updating node if death occurred
    if has_death_occurred:
        G.nodes()[node_label]['vitals'] = 'dead'
        
    return G, int(has_death_occurred)


def check_for_infection(G, source_label, target_label, reinfection):

    # Initialising variable
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

    # Converting rate to probability
    p_recovery = 1 - np.exp(-gamma * dt)

    # Checking if the infected individual has recovery with probability gamma
    return np.random.uniform() < p_recovery


def complete_step(G):

    # Finding all living nodes
    living_nodes = np.array([node for node in G.nodes() if G.nodes()[node]['vitals'] == 'alive'])
    
    # Choosing two neighbours at random within the population
    source_node, target_node = np.random.choice(living_nodes, size=2, replace=False)

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

    # Determining all living nodes
    living_nodes = np.array([node for node in G.nodes() if G.nodes()[node]['vitals'] == 'alive'])
    
    # Counting the number of individuals within each state
    S_total = np.sum([G.nodes()[node]['state'] == susceptible for node in living_nodes])
    I_total = np.sum([G.nodes()[node]['state'] == infected for node in living_nodes])
    M_total = np.sum([G.nodes()[node]['state'] == immune for node in living_nodes])

    return S_total, I_total, M_total


def run_simulation_iteration(G, n_nodes, I0, sim_time, event_impact, kappa_val, omega_val, iter_num):

    # Variable to be used to check for stochastic extinction
    peak_inf = 0

    # Creating a file to store the mc results to
    mc_outfile = open(mc_file_path + '_k-%s_om-%s_%s.txt' % (kappa_val, omega_val, (iter_num + 1)), 'w')
    mc_outfile.write('N=%s,I0=%s,t_max=%s,gamma=%s,sigma=%s,kappa=%s,omega=%s' % (n_nodes, I0, sim_time, gamma, sigma, kappa_val, omega_val))
    mc_outfile.write('\ntimestep,source_label,target_label,source_during,target_before,target_after,S_total,I_total,M_total')

    # Creating a file to store the total results to
    totals_outfile = open(totals_file_path + '_k-%s_om-%s_%s.txt' % (kappa_val, omega_val, (iter_num + 1)), 'w')
    totals_outfile.write('N=%s,I0=%s,t_max=%s,gamma=%s,sigma=%s,kappa=%s,omega=%s' % (n_nodes, I0, sim_time, gamma, sigma, kappa_val, omega_val))
    totals_outfile.write('\ntimestep,S_total,I_total,M_total,n_births,n_deaths')
    
    # Looping through timesteps
    for t in tqdm(range(sim_time)):

        # Initialising iteration births and deaths
        n_births = n_deaths = 0

        # Looping through number of nodes
        for n in range(G.number_of_nodes()):

            # Checking if dead node is reborn
            if G.nodes()[n]['vitals'] == 'dead':
                G, is_birth = get_potential_node_birth(G, n)
                n_births += is_birth

            # Checking if living node dies
            else:
                G, is_death = get_potential_node_death(G, n, event_impact[t])
                n_deaths += is_death
                
            # Completing an iteration step
            G, source_label, target_label, source_during, target_before, target_after = complete_step(G)
            
            # Updating network if required
            if target_before != target_after:
                G.nodes()[target_label]['state'] = target_after
    
            # Counting the number of individuals in each state
            S_total, I_total, M_total = get_state_totals(G)
        
            # Logging the mcs results
            mc_outfile.write('\n%s,%s,%s,%s,%s,%s,%s,%s,%s' % (
            t, source_label, target_label, source_during, target_before, target_after, S_total, I_total, M_total))

        # Determining if I total is greater than current peak inf
        if I_total > peak_inf:
            peak_inf = I_total
        
        # Logging total results
        totals_outfile.write('\n%s,%s,%s,%s,%s,%s' % (t, S_total, I_total, M_total, n_births, n_deaths))

    # Closing the data file
    mc_outfile.close()
    totals_outfile.close()

    # Returning whether an extinction event occurred
    return peak_inf < 2
    

def repeat_simulation(N, I0, t_max, kappa_val, omega_val, num_iterations=1):

    # Initialising a variable to store whether at least one event occurred
    event_occurred = False

    # Repeating until an event occurs
    while event_occurred == False:

        # Determining event impact for simulations
        event_times = get_event_times(t_max, kappa_val)
        event_impact = get_event_impact(t_max, event_times, omega_val)

        # Determining whether only one event occurred
        if len(np.nonzero(event_times)[0]) == 1:
            event_occurred = True

    # Creating file to write catastrophic event data to
    with open(event_path + '_k-%s_om-%s.txt' % (kappa_val, omega_val), 'w') as event_outfile:
        event_outfile.write('event_occurred,event_impact')

        # Writing catastrophic event data
        for event_data in list(zip(event_times, event_impact)):
            event_outfile.write('\n%s,%s' % (event_data))

    # Initialising a variable to store proportition of stochastic extinctions and number of attempts
    no_attempts, no_extinctions = 0, 0

    # Repeating entire simulation required number of times
    for n in range(num_iterations):

        # Updating attempt counter
        no_attempts += 1

        # Creating a temporary variable to force a simulation to repeat until non extinctions occur
        was_extinct = True

        # Repeating the simulation until an extinction does not occur
        while was_extinct == True:

            # Initialising the simulation
            G = initialise_graph(n_nodes=N, n_alive=N_alive)
            G = initialise_infections(G, n_nodes=N, n_infected=I0)

            # Running the current iteration
            was_extinct = run_simulation_iteration(G, N, I0, t_max, event_impact, kappa_val, omega_val, n)

            # Updating the number of extinctions if required
            if was_extinct:
                no_extinctions +=1

    # Returning the extinction proportion
    return no_extinctions / (no_attempts + no_extinctions)


##### CREATING USEFUL DATAFRAMES

def get_simulation_parameters(kappa_val, omega_val):

    # Creating a structure to store the simulation parameters
    parameters_dict = dict()

    # Locating all data files within the directory
    sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(totals_file_path + '_k-%s_om-%s' % (kappa_val, omega_val))]

    # Opening the first file (all simulation repeats have the same basic parameters)
    with open(sim_data_files[0], 'r') as in_file:

        # Reading first line
        data = in_file.readline().strip().split(',')

    # Looping through tuples of the form (parameter_name, value)
    for parameter_data in [parameter.split('=') for parameter in data]:
        
        # Storing the name and value
        parameter_name, parameter_value = parameter_data
        parameters_dict[parameter_name] = float(parameter_value)

    return parameters_dict
    

def get_events_dataframe(kappa_val, omega_val):

    # Reading in the data
    events_df = pd.read_csv(event_path + '_k-%s_om-%s.txt' % (kappa_val, omega_val), delimiter=',')
    return events_df


def get_results_dataframe(kappa_val, omega_val):

    # Creating a dataframe to store all results from all files
    results_df = pd.DataFrame()

    # Locating all data files within the directory
    sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(totals_file_path + '_k-%s_om-%s' % (kappa_val, omega_val))]

    # Looping through each data file
    for counter, file in enumerate(sim_data_files):

        # Reading in the data
        current_df = pd.read_csv(file, delimiter=',', skiprows=1)

        # Changing column names to match their origin file (i.e. 'column_name_number')
        new_col_names = [current_df.columns[0]] + [col_name + '_%s' % (counter+1) for col_name in current_df.columns[1:]]
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


##### PLOTTING THE RESULTS

def plot_state_totals(susceptible_df, infected_df, immune_df, events_df, parameters):

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

    # Overplotting event times
    for t, is_event_time in enumerate(events_df['event_occurred']):

        # Plotting event if required
        if is_event_time:
            plt.vlines(x=t, ymin=0, ymax=N, linestyles='dashed')
    
    # Adding plot titles and legend
    plt.title('Population sizes versus time for individual-based SIM model\nwith gamma=%s and sigma=%s' % (parameters['gamma'], parameters['sigma']))
    plt.xlabel('Time (days)')
    plt.ylabel('Population sizes')
    plt.show()


##### MAIN

# Setting the event parameters
kappa_val = 0.02
omega_val = 0.2

# Repeating the simulation
prop_extinct = repeat_simulation(N, I0, t_max, kappa_val, omega_val, n_iterations)

# Accessing the simulation parameters and creating useful dataframes
parameters_dict = get_simulation_parameters(kappa_val, omega_val)
events_df = get_events_dataframe(kappa_val, omega_val)
results_df = get_results_dataframe(kappa_val, omega_val)
susceptible_df, infected_df, immune_df = get_state_dataframes(results_df)

# Plotting the data
plot_state_totals(susceptible_df, infected_df, immune_df, events_df, parameters_dict)
