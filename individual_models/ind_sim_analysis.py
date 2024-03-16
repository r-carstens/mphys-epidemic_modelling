import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import string
import os

# Setting filename to be used for storing and reading data
mc_file_path = 'sim_with_hosts_mc'
event_path = 'host_events'

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting a mutation rate
p_mutation = 0.3


##### USEFUL HELPER FUNCTIONS

def get_sorted_tree_edges(multi_tree):

    # Sort edges based on the timestep values
    sorted_edges = sorted(multi_tree.edges(data=True), key=lambda edge: edge[2]['timestep'])
    return sorted_edges


################################################## TREES

##### TRANSMISSION TREE ANALYSIS

def get_simulation_parameters(first_file):

    # Initialising dictionary to hold results
    parameters_dict = dict()

    # Opening the first file (all simulation repeats have the same basic parameters)
    with open(first_file, 'r') as in_file:

        # Reading first line
        data = in_file.readline().strip().split(',')

    # Looping through tuples of the form (parameter_name, value)
    for parameter_data in [parameter.split('=') for parameter in data]:
        
        # Storing the name and value
        parameter_name, parameter_value = parameter_data
        parameters_dict[parameter_name] = float(parameter_value)

    return parameters_dict


def get_infection_df(first_file):

    # Importing transmission data and only retaining infection events
    df = pd.read_csv(first_file, delimiter=',', skiprows=1)
    infection_df = df.loc[(df['source_during'] == infected) & (df['target_before'].isin([susceptible, immune])) & (df['target_after'] == infected)]

    return infection_df
    
    
def get_transmission_dict(infection_df):

    # Creating a dictionary to store the timestep of an event and the involved nodes
    transmission_dict = dict()

    # Looping through data file in order of increasing time
    for ind in infection_df.index:
        
        # Extracting the source and target node labels
        source_label = infection_df['source_label'][ind]
        target_label = infection_df['target_label'][ind]

        # Determining whether the node was susceptible or immune prior to the infection
        was_reinfection = (infection_df['target_before'][ind] == immune)

        # Storing the results
        transmission_dict[ind] = {'source_label': source_label, 'target_label': target_label, 'was_reinfection': was_reinfection}

    return transmission_dict


def get_transmission_tree(transmission_dict):

    # Creating a directed multigraph
    transmission_tree = nx.MultiDiGraph()
    
    # Looping through data in order of increasing time
    for t, data in transmission_dict.items():
    
        # Extracting the data from the current event
        source_label, target_label, reinfection_occurred = [value for key, value in data.items()]
    
        # Adding an edge between the current source and target nodes and attributing the timestep
        transmission_tree.add_edge(source_label, target_label, timestep=t, was_reinfection=reinfection_occurred)

    return transmission_tree


##### MODELLING SUBSTITUTIONS ACROSS TREE

def get_initialised_sub_tree(transmission_tree):

    # Copying the transmission tree
    sub_tree = transmission_tree.copy()

    # Sorting the edges in order of increasing time
    sorted_sub_edges = get_sorted_tree_edges(sub_tree)

    # Looping through the edges in order of increasing time
    for u, v, data in sorted_sub_edges:
        
        # Checking if a substitution occurs and adding the result to the edge
        substitution_occurred = np.random.uniform() < p_mutation
        data['substitution_occurred'] = substitution_occurred

    # Determinining the initial event nodes
    source_label, target_label, data = sorted_sub_edges[0]

    # Infecting corresponding source node with 'a' and initialising its pathogen history
    sub_tree.nodes()[source_label]['current_pathogen_name'] = 'a'
    sub_tree.nodes()[source_label]['pathogen_history'] = {data['timestep']: 'a'}

    # Repeating for the target label
    sub_tree.nodes()[target_label]['current_pathogen_name'] = 'a'
    sub_tree.nodes()[target_label]['pathogen_history'] = {data['timestep']: 'a'}
            
    return sub_tree


def get_substitution_tree(transmission_tree):

    # Simulating substitutions across the transmission tree and sorting the edges in order of increasing time
    sub_tree = get_initialised_sub_tree(transmission_tree)
    sorted_sub_edges = get_sorted_tree_edges(sub_tree)

    # Creating an array of possible subsitutions
    sub_names = list(string.ascii_lowercase)[1:] + list(string.ascii_uppercase) + list(np.arange(10000).astype(str))
    sub_counter = 0

    # Looping through the edges in order of increasing time
    for source_label, target_label, data in sorted_sub_edges:

        # Extracting the data from the current event
        timestep, substitution_occurred, reinfection_occurred = [value for key, value in data.items()]

        # Extracting the source pathogen name at the timestep
        source_pathogen = sub_tree.nodes(data=True)[source_label]['current_pathogen_name']
        target_pathogen = source_pathogen

        # Checking if a substitution occurred during the current event
        if substitution_occurred:

            # Updating the pathogen name to include the substitution
            target_pathogen += '_%s' % sub_names[sub_counter]
            sub_counter += 1

        # Checking if the node already has pathogen name data
        if 'current_pathogen_name' in sub_tree.nodes()[target_label]:
            
            # Updating the node data
            sub_tree.nodes()[target_label]['current_pathogen_name'] = target_pathogen
            sub_tree.nodes()[target_label]['pathogen_history'][timestep] = target_pathogen

        # Initialsing pathogen name data if not
        else:

            # Adding pathogen data to node
            sub_tree.nodes()[target_label]['current_pathogen_name'] = target_pathogen
            sub_tree.nodes()[target_label]['pathogen_history'] = {timestep: target_pathogen}

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


def get_edge_removals(sub_tree, phylo_tree):

    # Creating structures to store the cross immunity tree and data on which nodes have had pathogens removed
    cross_tree = nx.MultiDiGraph()
    removed_node_data = dict()

    # Looping through the edges in order of increasing time
    for source_label, target_label, data in get_sorted_tree_edges(sub_tree):

        # Extracting data from the current event
        timestep, reinfection_occurred, substitution_occurred = [value for key, value in data.items()]

        # Checking if the source node has already been removed, in which case the target can't have been infected and so is removed
        if source_label in removed_node_data:

            # Checking if there is already data on the current node (only want to keep the initial removal timestep)
            if target_label not in removed_node_data:
                removed_node_data[target_label] = timestep
            
        # Checking if the event was a reinfection, in which case cross immunity may occur
        elif reinfection_occurred:

            # Extracting the target pathogen history and the potential new pathogen
            target_path_hist = sub_tree.nodes(data=True)[target_label]['pathogen_history']
            new_pathogen = target_path_hist[timestep]

            # Determining the shortest of the distances to each previously encountered pathogen
            prev_pathogens, phylo_dists = get_previous_pathogens(path_hist=target_path_hist, new_path=new_pathogen, t=timestep, phylo_tree=phylo_tree)
            shortest_dist = np.min(phylo_dists)

            # If the node is immune, the edge is not added and its removal time is stored
            if np.random.uniform() < get_immunity_probability(dist=shortest_dist):
                
                # Checking if there is already data on the current node (only want to keep the initial removal timestep)
                if target_label not in removed_node_data:
                    removed_node_data[target_label] = timestep

            # If the node is not immune, it stays within the tree
            else:
                cross_tree.add_edge(source_label, target_label, timestep=timestep, was_reinfection=reinfection_occurred,
                                substitution_occurred=substitution_occurred)
                
        # Checking if the event was an initial infection, in which case the data is unchanged
        else:
            cross_tree.add_edge(source_label, target_label, timestep=timestep, was_reinfection=reinfection_occurred,
                                substitution_occurred=substitution_occurred)

    return cross_tree, removed_node_data


def get_node_pathogen_removals(cross_tree, sub_tree, removed_node_data):

    # Looping through all removed nodes
    for node_label, removed_timestep in removed_node_data.items():

        # Checking if the node is in the cross tree (nodes with one edge may have been completely removed if the source was removed)
        if cross_tree.has_node(node_label):

            # Extracting the node's unedited pathogen history (from the sub tree) and making an updated copy
            node_path_history = sub_tree.nodes(data=True)[node_label]['pathogen_history']
            updated_node_path_history = node_path_history.copy()

            # Looping through the history
            for path_t, path_name in node_path_history.items():
    
                # Removing pathogen data if the pathogen was trasmitted after the node was removed
                if path_t >= removed_timestep:
                    del updated_node_path_history[path_t]
    
            # Replacing cross tree node with new data (current pathogen isn't used, but I noticed it too late to debug and remove it)
            cross_tree.nodes[node_label]['current_pathogen_name'] = sub_tree.nodes(data=True)[node_label]['current_pathogen_name']
            cross_tree.nodes[node_label]['pathogen_history'] = updated_node_path_history

    return cross_tree

    
def get_cross_immunity_tree(sub_tree, phylo_tree):

    # Removing the edges for cross-immune events and determining which node pathogen histories require changes
    cross_tree, removed_node_data = get_edge_removals(sub_tree, phylo_tree)
    
    # Applying the pathogen history updates
    cross_tree = get_node_pathogen_removals(cross_tree, sub_tree, removed_node_data)

    return cross_tree


##### MODELLING VARIANT EMERGENCE

def get_time_intervals(sorted_tree_edges, t1_percent=0.2, t2_percent=0.8, delta_t_percent=0.2):

    # Determining all event timesteps and the number of events
    all_timesteps = [data['timestep'] for source_label, target_label, data in sorted_tree_edges]
    n_timesteps = len(all_timesteps)

    # Determining the timewindow reach
    delta_t = int(delta_t_percent * n_timesteps)

    # Determining when the time intervals will be set
    t1 = int(t1_percent * n_timesteps)
    t2 = int(t2_percent * n_timesteps)

    # Determining the time at the intervals
    t1_step = all_timesteps[t1]
    t2_step = all_timesteps[t2]

    return t1_step, t2_step, delta_t
    
    
def get_snapshot_pathogen_history(target_path_hist, t2_step, delta_t):

    # Creating a variable to store the restricted data
    restricted_pathogen_data = dict()

    # Looping through the path data
    for time, pathogen in target_path_hist.items():

        # Checking if the current transmission falls within the time snapshot
        if (t2_step - delta_t) <= time <= (t2_step + delta_t):

            # Storing the result
            restricted_pathogen_data[time] = pathogen

    return restricted_pathogen_data
    

def get_final_t_data(path_history, t2_step):

    # Creating a struture to store which pathogen the host was infected with at t2
    final_t, path_at_t2 = -1, ''

    # Looping through the target node's pathogen history
    for path_t, path_name in path_history.items():

        # Checking if the current transmission occurred before t2
        if path_t < t2_step:

            # Saving the current details
            final_t = path_t
            path_at_t2 = path_name

    return final_t, path_at_t2


def get_predecessor_node(substitution_tree, final_t, target_label):

    # Initialising variable to store the found predecessor
    found_sorce_label = -1
    
    # Looping through all edges
    for temp_source_label, temp_target_label, temp_data in substitution_tree.edges(data=True):
    
        # Checking if the target label and timestep match
        if temp_data['timestep'] == final_t and temp_target_label == target_label:
            found_source_label = temp_source_label

    return found_source_label


def get_variant_predecessors(tree):

    # Creating a structure to store which nodes infected others
    origin_nodes = dict()
    
    # Sorting the tree in order of increasing time
    sorted_tree_edges = get_sorted_tree_edges(tree)
    
    # Determining the timesteps to cut the tree at
    t1_step, t2_step, delta_t = get_time_intervals(sorted_tree_edges)
    
    # Reversing the tree order to go back in time 
    reversed_sorted_tree_edges = list(reversed(sorted_tree_edges))
    
    # Looping through sorted tree edges
    for source_label, target_label, data in reversed_sorted_tree_edges:
        
        # Extracting the target node data
        target_path_hist = substitution_tree.nodes(data=True)[target_label]['pathogen_history']
    
        # Determining which pathogen the node was infectious with at time t2
        final_t, path_at_t2 = get_final_t_data(target_path_hist, t2_step)
    
        # Finding the source node 
        found_source = get_predecessor_node(substitution_tree, final_t, target_label)
        
        # Checking if the source already exists in the dictionary
        if found_source in origin_nodes:

            # Checking if the target label is already stored (nb: tested this, it is a bug due to using the multi graph, not a mistake)
            if target_label not in origin_nodes[found_source]:
                origin_nodes[found_source].append(target_label)
        
        # Initialising the data
        else:
            origin_nodes[found_source] = [target_label]

    return origin_nodes
    

def get_variant_proportions(origin_nodes):

    # Creating structures to store the source label name and number of onward infections caused
    source_labels = np.array(list(origin_nodes.keys()))
    no_infections_caused = np.zeros(shape=source_labels.shape)

    # Looping through the dictionary
    for counter, (key, data) in enumerate(origin_nodes.items()):

        # Storing the number of infections caused
        no_infections_caused[counter] = len(data)

    # Normalising the result to get proportions
    proportion_infections = no_infections_caused / len(no_infections_caused)

    return proportion_infections


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


def get_all_variant_diversity_numbers(prop_infections):

    # Determining the different diversity measures
    div_q0 = get_effective_number(prop_infections, q=0)
    div_q1 = get_effective_number(prop_infections, q=1)
    div_q2 = get_effective_number(prop_infections, q=2)

    return div_q0, div_q1, div_q2


################################################## MAIN FUNCTIONS

def get_trees(first_file):

    # Reading all data that involved an infection from the results file and producing a transmission dictionary
    inf_df = get_infection_df(first_file)
    transmission_dict = get_transmission_dict(inf_df)
    
    # Producing the transmission tree
    transmission_tree = get_transmission_tree(transmission_dict)
    
    # Producing the substitution tree
    substitution_tree = get_substitution_tree(transmission_tree)
    
    # Producing the phylogenetic tree
    phylogenetic_tree = get_phylogenetic_tree(substitution_tree)
    
    # Producing the cross-immunity tree
    cross_tree = get_cross_immunity_tree(substitution_tree, phylogenetic_tree)

    return inf_df, transmission_dict, transmission_tree, substitution_tree, phylogenetic_tree, cross_tree


def get_variant_analysis(substitution_tree, cross_tree):

    # Determining origin nodes for the substitution and cross-immunity trees
    sub_origin_nodes = get_variant_predecessors(substitution_tree)
    cross_origin_nodes = get_variant_predecessors(cross_tree)

    # Determining the proportion that each variant occurrs
    sub_variant_props = get_variant_proportions(sub_origin_nodes)
    cross_variant_props = get_variant_proportions(cross_origin_nodes)

    # Accessing the different diversity measures
    sub_div_q0, sub_div_q1, sub_div_q3 = get_all_variant_diversity_numbers(sub_variant_props)
    cross_div_q0, cross_div_q1, cross_div_q3 = get_all_variant_diversity_numbers(cross_variant_props)

    return np.array([sub_div_q0, sub_div_q1, sub_div_q3]), np.array([cross_div_q0, cross_div_q1, cross_div_q3])


################################################## MAIN

# Finding all files
sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(mc_file_path)]
first_file = sim_data_files[0]

# Accessing the simulation parameters
parameters_dict = get_simulation_parameters(first_file)

# Determining the relevant simulation trees and data
inf_df, transmission_dict, transmission_tree, substitution_tree, phylogenetic_tree, cross_immunity_tree = get_trees(first_file)

# Analysising the variants
sub_diversities, cross_diversities = get_variant_analysis(substitution_tree, cross_immunity_tree)