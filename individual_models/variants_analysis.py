import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import string
import os

# Setting input filename
complete_path = 'complete_without_events'

# Setting number of files to analyse
n_iterations = 1

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting a mutation rate
p_mutation = 0.3


##### TRANSMISSION TREE ANALYSIS

def get_infection_df(iter_num):

    # Determining input file name
    in_file = complete_path + '_%s.txt' % (iter_num + 1)
    
    # Importing transmission data and only retaining infection events
    df = pd.read_csv(in_file, delimiter=',', skiprows=1)
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


def get_sorted_tree_edges(multi_tree):

    # Sort edges based on the timestep values
    sorted_edges = sorted(multi_tree.edges(data=True), key=lambda edge: edge[2]['timestep'])
    return sorted_edges
    

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


def get_cross_immunity_tree(sub_tree, phylo_tree):

    # Copying the substitution tree and sorting edges in order of increasing time
    cross_tree = sub_tree.copy()
    sorted_sub_edges = get_sorted_tree_edges(sub_tree)

    # Creating a structure to store when nodes are removed
    removed_nodes = []
    
    # Looping through the edges in order of increasing time
    for source_label, target_label, data in sorted_sub_edges:

        # Extracting data from the current event
        timestep, reinfection_occurred, substitution_occurred = [value for key, value in data.items()]

        # Removing current event if source label has already been removed
        if source_label in removed_nodes:
            cross_tree.remove_edge(source_label, target_label)

        # Checking if event was a reinfection
        if reinfection_occurred:

            # Finding the corresponding node in the cross-immunity tree
            target_node_data = sub_tree.nodes(data=True)[target_label]
            
            # Extracting all pathogens encountered by the node
            pathogen_history = target_node_data['pathogen_history']
            new_pathogen = pathogen_history[timestep]

            # Determining the distance to each previously encountered pathogen
            prev_pathogens, phylo_dists = get_previous_pathogens(path_hist=pathogen_history, new_path=new_pathogen, t=timestep, phylo_tree=phylo_tree)
            
            # Finding the shortest distance and immunity probability
            shortest_dist = np.min(phylo_dists)
            immunity_prob = get_immunity_probability(dist=shortest_dist)

            # Testing if the node is immune
            if np.random.uniform() < immunity_prob:

                # Labelling all successors as removed as of the current timestep
                for successor in list(sub_tree.successors(target_label)):
                    removed_nodes.append(successor)

    return cross_tree


##### MAIN

# Reading all data that involved an infection from the results file and producing a transmission dictionary
inf_df = get_infection_df(iter_num=0)
transmission_dict = get_transmission_dict(inf_df)

# Producing the transmission tree
transmission_tree = get_transmission_tree(transmission_dict)

# Producing the substitution tree
substitution_tree = get_substitution_tree(transmission_tree)

# Producing the phylogenetic tree
phylogenetic_tree = get_phylogenetic_tree(substitution_tree)

# Producing the cross-immunity tree
cross_tree = get_cross_immunity_tree(substitution_tree, phylogenetic_tree)