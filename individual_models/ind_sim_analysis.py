import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import string
import os

# Displaying annoying jupyter notebook warnings
import warnings
warnings.filterwarnings('ignore')

# Setting filename to be used for storing and reading data
mc_file_path = 'sim_with_hosts_mc'
event_path = 'host_events'

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'


##### USEFUL HELPER FUNCTIONS

def get_sorted_tree_edges(multi_tree):

    # Sort edges based on the timestep values
    sorted_edges = sorted(multi_tree.edges(data=True), key=lambda edge: edge[2]['timestep'])
    return sorted_edges


##### TRANSMISSION TREE ANALYSIS

def get_simulation_parameters(filename):

    # Initialising dictionary to hold results
    parameters_dict = dict()

    # Opening the first file (all simulation repeats have the same basic parameters)
    with open(filename, 'r') as in_file:

        # Reading first line
        data = in_file.readline().strip().split(';')

    # Looping through tuples of the form (parameter_name, value)
    for parameter_data in [parameter.split('=') for parameter in data]:
        
        # Storing the name and value
        parameter_name, parameter_value = parameter_data
        parameters_dict[parameter_name] = float(parameter_value)

    return parameters_dict


def get_infection_df(filename):

    # Importing transmission data and only retaining infection events
    df = pd.read_csv(filename, delimiter=';', skiprows=1)
    infection_df = df.loc[(df['source_during'] == infected) \
                          & (df['target_before'].isin([susceptible, immune])) \
                          & (df['target_after'] == infected)]

    return infection_df
    
    
def get_transmission_dict(infection_df):

    # Creating a dictionary to store the timestep of an event and the involved nodes
    transmission_dict = dict()

    # Looping through data file in order of increasing time
    for ind in infection_df.index:
        
        # Extracting the timestep
        t = infection_df['timestep'][ind]
        
        # Extracting the source and target node labels
        source_label = infection_df['source_label'][ind]
        target_label = infection_df['target_label'][ind]

        # Extracting the mutation pathogen
        current_pathogen = infection_df['target_mutation_after'][ind]

        # Determining whether the node was susceptible or immune prior to the infection
        was_reinfection = (infection_df['target_before'][ind] == immune)

        # Extracting whether an event occurred
        was_event = infection_df['event_occurred'][ind]

        # Storing the results
        transmission_dict[ind] = {'source_label': source_label, 'target_label': target_label, 'timestep': t,
                                  'transmitted_pathogen': current_pathogen, 'was_reinfection': was_reinfection,
                                  'was_event': was_event}

    return transmission_dict


def get_transmission_tree(transmission_dict):

    # Creating a directed multigraph
    transmission_tree = nx.MultiDiGraph()
    
    # Looping through data in order of increasing time
    for t_ind, data in transmission_dict.items():
    
        # Extracting the data from the current event
        source_label, target_label, timestep, current_pathogen, reinfection_occurred, \
        was_event = [value for key, value in data.items()]
    
        # Adding an edge between the current source and target nodes and attributing the timestep
        transmission_tree.add_edge(source_label, target_label, timestep=timestep, \
                                   transmitted_path=current_pathogen, was_reinfection=reinfection_occurred, was_event=was_event)

    return transmission_tree


##### ADDING NODE PATHOGEN HISTORIES

def get_substitution_tree(transmission_tree):

    # Creating a copy of the transmission tree
    sub_tree = transmission_tree.copy()
    
    # Sorting the transmission tree edges in order of increasing time
    sorted_sub_edges = get_sorted_tree_edges(transmission_tree)

    # # Determinining the initial transmission and initialising the data
    initial_source_label, initial_target_label, initial_data = sorted_sub_edges[0]
    sub_tree.nodes[initial_source_label]['pathogen_history'] = {initial_data['timestep']: initial_data['transmitted_path']}
    sub_tree.nodes[initial_target_label]['pathogen_history'] = {initial_data['timestep']: initial_data['transmitted_path']}

    # Looping through the edges in order of increasing time
    for source_label, target_label, data in sorted_sub_edges[1:]:

        # Extracting the data from the current event
        timestep, transmitted_pathogen, reinfection_occurred, was_event = [value for key, value in data.items()]

        # Checking if the node already has pathogen name data
        if 'pathogen_history' in sub_tree.nodes[target_label]:
            sub_tree.nodes[target_label]['pathogen_history'][timestep] = transmitted_pathogen

        # Initialising pathogen name data if not
        else:
            sub_tree.nodes[target_label]['pathogen_history'] = {timestep: transmitted_pathogen}  

    return sub_tree


##### PRODUCING THE PHYLOGENETIC TREE

def get_unique_pathogens(sub_tree):
    
    # Determining all possible pathogens
    possible_pathogens = []
    
    # Looping through all nodes in the tree
    for node, data in sub_tree.nodes(data=True):
    
        # Adding the variants encountered by the current node
        possible_pathogens.extend([value for key, value in data['pathogen_history'].items()])
    
    # Determining all unique pathogens
    possible_pathogens = np.array(possible_pathogens)
    unique_pathogens = np.unique(possible_pathogens)

    return unique_pathogens

    
def get_phylogenetic_tree(sub_tree):

    # Creating a graph to store the phylogenetic data
    phylo_tree = nx.Graph()

    # Determining all unique pathogens
    unique_pathogens = get_unique_pathogens(sub_tree)

    # Looping through the unique pathogens
    for pathogen in unique_pathogens:

        # Substitutions are denoted by the final instance of '_' (eg '1_2' is a mutation of '1', and '1_2_3' is a mutation of '1_2')
        substitutions = pathogen.split('_')

        # Looping through the data up to each instance of '_'
        for i in range(1, len(substitutions)):

            # Adding an edge between the the old variant and new mutation (character after the final instance '_')
            phylo_tree.add_edge('_'.join(substitutions[:i]), '_'.join(substitutions[:i + 1]))

    return phylo_tree


##### MODELLING CROSS IMMUNITY

def get_previous_pathogens(path_hist, new_path, t, phylo_tree):
    
    # Determining the current potential pathogens and pathogens up to the timestep
    previous_pathogens, phylogenic_distances = [], []

    # Looping through the pathogen history
    for t_infected, previous_pathogen in path_hist.items():

        # Breaking once the current timestep is reached
        if t_infected == t:
            break

        # Storing the shortest distance between the potential new pathogen and the current historic pathogen
        shortest_path = nx.shortest_path_length(phylo_tree, source=previous_pathogen, target=new_path)
        phylogenic_distances.append(shortest_path)

        # Storing the current pathogen
        previous_pathogens.append(previous_pathogen)

    return np.array(previous_pathogens), np.array(phylogenic_distances)
    

def get_immunity_probability(dist):

    # Drawing immunity probability from a sigmoid
    return 1 / (1 + np.exp(dist))


def get_immune_nodes(sub_tree, phylo_tree):
    
    # Creating a structure to store which nodes have had pathogens removed
    removed_node_data = dict()

    # Looping through the edges in order of increasing time
    for source_label, target_label, data in get_sorted_tree_edges(sub_tree):

        # Extracting data from the current event
        timestep, transmitted_path, was_reinfection, was_event = [value for key, value in data.items()]

        # Checking if the source node has already been removed, so removing the target if not already removed
        if source_label in removed_node_data and target_label not in removed_node_data:
            removed_node_data[target_label] = timestep

        # Checking if the event was a reinfection, in which case cross-immunity may occur
        elif was_reinfection:
            
            # Extracting the target pathogen history and the potential new pathogen
            target_path_hist = sub_tree.nodes(data=True)[target_label]['pathogen_history']
            new_pathogen = target_path_hist[timestep]

            # Determining the shortest of the distances to each previously encountered pathogen
            prev_pathogens, phylo_dists = get_previous_pathogens(target_path_hist, new_pathogen, timestep, phylo_tree)
            shortest_dist = np.min(phylo_dists)

            # If the node is immune, the edge is not added and its removal time is stored
            if np.random.uniform() < get_immunity_probability(dist=shortest_dist) and target_label not in removed_node_data:
                removed_node_data[target_label] = timestep

    return removed_node_data


def get_node_history_updates(sub_tree, removed_node_data):

    # Creating structures to store the cross immunity tree
    cross_tree = nx.MultiDiGraph()
    
    # Looping through all cross tree nodes
    for node in sub_tree.nodes():

        # Extracting the node's unedited pathogen history (from the sub tree) and making an updated copy
        node_path_history = sub_tree.nodes(data=True)[node]['pathogen_history']
        updated_node_path_history = node_path_history.copy()
    
        # Checking if removal data exists for the node
        if node in removed_node_data:

            # Extracting the time at which the node was removed
            removed_timestep = removed_node_data[node]

            # Looping through the history
            for path_t, path_name in node_path_history.items():
    
                # Removing pathogen data if the pathogen was trasmitted after the node was removed
                if path_t > removed_timestep:
                    del updated_node_path_history[path_t]

        # Adding the node to the tree
        cross_tree.add_node(node, pathogen_history=updated_node_path_history)

    return cross_tree


def get_edge_removals(cross_tree, sub_tree, removed_node_data):

    # Looping through the tree in order of increasing time
    for source_label, target_label, data in get_sorted_tree_edges(sub_tree):

        # Extracting data from the current event
        current_timestep, transmitted_pathogen, reinfection_occurred, was_event = [value for key, value in data.items()]

        # Checking if the source node is in the removed node data
        if source_label in removed_node_data:

            # Ensuring the tranmission occurred before the node was removed
            if current_timestep < removed_node_data[source_label]:

                # Adding the edge if required
                cross_tree.add_edge(source_label, target_label, timestep=current_timestep, transmitted_path=transmitted_pathogen, was_reinfection=reinfection_occurred)

    return cross_tree


def get_cross_immunity_tree(sub_tree, phylo_tree):

    # Finding the time at which nodes are removed
    removed_node_data = get_immune_nodes(sub_tree, phylo_tree)

    # Applying the pathogen history updates
    cross_tree = get_node_history_updates(sub_tree, removed_node_data)

    # Removing the edges for cross-immune events and determining which node pathogen histories require changes
    cross_tree = get_edge_removals(cross_tree, sub_tree, removed_node_data)

    return cross_tree


##### MODELLING VARIANT EMERGENCE

def get_snapshot_variants(tree, t1_step, t2_step):

    # Creating a structure to store the variants within the timestep
    snapshot_variants = dict()

    # Looping through nodes
    for node_label, node_data in tree.nodes(data=True):

        # Looping through the node's pathogen history
        for path_t, path_name in node_data['pathogen_history'].items():

            # Checking if the current pathogen is within the time frame
            if t1_step < path_t < t2_step:
                snapshot_variants[path_name] = path_t

    return snapshot_variants


def get_phylo_matrix(variants_at_t1, phylo_tree):
    
    # Determining the number of variants
    n_variants = len(variants_at_t1.keys())

    # Creating a matrix to store the phylogenetic distances
    phylo_matrix = np.zeros(shape=(n_variants, n_variants))
    
    # Looping through the keys
    for i, var_i in enumerate(variants_at_t1.keys()):
        for j, var_j in enumerate(variants_at_t1.keys()):
            
            # Determining the phylogenetic distance between the variants at i, j
            phylo_matrix[i, j] = nx.shortest_path_length(phylo_tree, source=var_i, target=var_j)

    return phylo_matrix


def get_threshold_matrix(dist_matrix, threshold):
    
    # Making a copy of the matrix
    th_matrix = np.zeros(shape=dist_matrix.shape)
    
    # Looping through the matrix
    for i in range(len(th_matrix)):
        for j in range(len(th_matrix)):
            
            # Setting all locations above the threshold to zero
            if dist_matrix[i, j] >= threshold:
                th_matrix[j, i] = 1

    # Converting the adjacency matrix into a graph
    th_network = nx.from_numpy_array(th_matrix)
    return th_network


def get_threshold_components(distance_matrix):
    
    # Determining the maximum distance and creating a counter to loop through distance thresholds
    max_dist = np.max(distance_matrix)
    
    # Creating arrays to store the results
    threshold_distances = np.arange(stop=(max_dist + 1)).astype(int)
    threshold_components = np.zeros(shape=threshold_distances.shape)
    
    # Looping through the threshold distances
    for dist_threshold in threshold_distances:
    
        # Determining the corresponding threshold network
        threshold_network = get_threshold_matrix(distance_matrix, threshold=dist_threshold)
        
        # Determining the number of components and storing the results
        n_components = nx.number_connected_components(threshold_network)
        threshold_components[dist_threshold] = n_components
        
    # Determining the maximum threshold distance and extracting the threshold network at that point
    best_threshold = np.argmax(threshold_distances)
    best_network = get_threshold_matrix(distance_matrix, threshold=threshold_distances[best_threshold])
    
    return threshold_components, threshold_distances, best_network
    
        
def get_shannon_entropies(n_components):
    
    # Creating an array to store the results
    shannon_entropies = np.zeros(shape=n_components.shape)
    
    # Looping through the component numbers
    for i in range(len(n_components)):
        
        # Extracting the components up to the current distance
        required_components = n_components[:i+1]
        
        # Determining the proportion and log prop of each number of components
        proportions = required_components / np.sum(required_components)
        log_proportions = np.log(proportions)
        
        # Determining the current Shannon entropy and storing the results
        current_entropy = - np.sum(proportions * log_proportions)
        shannon_entropies[i] = current_entropy
        
    return shannon_entropies
        
        
def get_main_variant(n_components, shannon_entropies):
    
    # Determining the location with the greatest shannon entropy and the corresponding component size
    greatest_entropy_loc = np.argmax(shannon_entropies)
    largest_variant = n_components[greatest_entropy_loc]
    
    return largest_variant


def get_snapshot_voc(n_components):

    # Determining the Shannon entropies
    shannon_entropies = get_shannon_entropies(n_components)

    # Determining the largest component
    largest_component = get_main_variant(n_components, shannon_entropies)
    
    return largest_component


def get_effective_number(prop_infections, q=2):

    # Initialising diversity variable
    diversity = 0

    # Checking if q=1 (this cannot be calculated using the Hill equation exactly, needs limit taken)
    if q == 1:

        # Determining exp of shannon entropy
        diversity = np.exp(-np.sum(prop_infections * np.log(prop_infections)))

    else:

        # Determining the effective number for the chosen q value
        diversity = (np.sum(prop_infections**q))**(1/(1-q))

    return diversity


def get_snapshot_diversities(n_components):
    
    # Determining the proportions of each component
    proportions = n_components / np.sum(n_components)
    
    # Determining the different diversity measures
    q0_div = get_effective_number(proportions, q=0)
    q1_div = get_effective_number(proportions, q=1)
    q2_div = get_effective_number(proportions, q=2)
    
    return q0_div, q1_div, q2_div

    
##### MAIN FUNCTIONS

def get_trees(filename, check_for_cross_immunity):

    # Creating the transmission dictionary
    inf_df = get_infection_df(filename)
    transmission_dict = get_transmission_dict(inf_df)

    # Creating the transmission tree and adding the substitution histories to the nodes
    transmission_tree = get_transmission_tree(transmission_dict)
    transmission_tree = get_substitution_tree(transmission_tree)

    # Creating the phylogenetic tree
    phylogenetic_tree = get_phylogenetic_tree(transmission_tree)

    # Temporarily setting cross immunity tree to sub tree
    cross_tree = transmission_tree.copy()
    
    # Checking if cross immunity occurred and updating if required
    if check_for_cross_immunity:
        cross_tree = get_cross_immunity_tree(transmission_tree, phylogenetic_tree)

    return inf_df, transmission_dict, transmission_tree, phylogenetic_tree, cross_tree


def get_variant_analysis(tree, t1_step, t2_step):
    
    # Determining the variants at the timestep and their antigenic distances
    snapshot_variants = get_snapshot_variants(tree, t1_step, t2_step)
    
    # Getting the phylogenetic distances between the variants
    distance_matrix = get_phylo_matrix(snapshot_variants, phylogenetic_tree)

    # Determining the threshold component results
    n_components, thresholds, best_network = get_threshold_components(distance_matrix)
    
    # Determining the voc
    snapshot_voc = get_snapshot_voc(n_components)
    
    # Determining the snapshot diversities
    q0_div, q1_div, q2_div = get_snapshot_diversities(n_components)
    
    return snapshot_voc, q0_div, q1_div, q2_div


def get_all_timesteps(tree):
    
    # Determining the timesteps
    all_timesteps = np.array([data['timestep'] for u, v, data in tree.edges(data=True)])
    unique_timesteps = np.unique(all_timesteps)
    
    return unique_timesteps
    
    
def get_variant_evolution(tree):
    
    # Determining all timesteps
    all_timesteps = get_all_timesteps(tree)
    
    # Creating structures to store the results
    vocs, q0s, q1s, q2s = [], [], [], []
    
    # Looping through the timesteps
    for i in range(1, len(all_timesteps) - 2):
        
        # Extracting the current windows
        t1_step = all_timesteps[i]
        t2_step = all_timesteps[i+2]
        
        # Determining the current variant analysis
        snapshot_voc, q0_div, q1_div, q2_div = get_variant_analysis(tree, t1_step, t2_step)
        
        # Storing the results
        vocs.append(snapshot_voc)
        q0s.append(q0_div)
        q1s.append(q1_div)
        q2s.append(q2_div)
        
    return vocs, q0s, q1s, q2s


def get_event_locs(filename):

    # Reading in the data and determining where events occurred
    mc_events_df = pd.read_csv(filename, delimiter=';', skiprows=1)
    event_locs = mc_events_df[mc_events_df['event_occurred'] == True]
    
    # Determining the timesteps at which the event occurred
    event_timesteps = event_locs['timestep']
    unique_timesteps = np.unique(event_timesteps)
    
    return unique_timesteps


################################################## MAIN

# Finding all files
sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(mc_file_path)]

# Creating lists to store the results
all_vocs = []

# Looping through files
for counter, filename in enumerate(tqdm(sim_data_files)):

    # Accessing the simulation parameters and storing the results
    parameters_dict = get_simulation_parameters(filename)
    k = parameters_dict['kappa']
    om = parameters_dict['omega']

    # Generating the relevant data and simulation trees
    check_for_cross_immunity = (parameters_dict['sigma'] != 0)
    inf_df, transmission_dict, transmission_tree, phylogenetic_tree, cross_tree = get_trees(filename, check_for_cross_immunity)
    
    # Analysing the variants
    vocs, q0s, q1s, q2s = get_variant_evolution(transmission_tree)
    
    # Storing the results
    all_vocs.append(vocs)


# Determining all simulation timesteps and creating a list to store total vocs
all_timesteps = get_all_timesteps(transmission_tree)
total_vocs = np.zeros(shape=all_timesteps.shape)

# Looping throught the variant results
for current_vocs in all_vocs:
    
    # Storing the current results and plotting
    total_vocs += current_vocs
    plt.plot(all_timesteps, current_vocs, alpha=0.2, color='firebrick')
    
# Overplotting the event locations
for event in get_event_locs(filename):
    plt.vlines(x=event, ymin=0, ymax=np.max(vocs), linestyle='dashed')
    
# Determining the average result and plotting
average_vocs = total_vocs / len(all_vocs)
plt.plot(all_timesteps, average_vocs, color='firebrick')
