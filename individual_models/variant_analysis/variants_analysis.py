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

# Setting whether cross-immunity should be incorporated
include_cross_immunity = True


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

        # Checking if a mutation occurred
        substitution_occurred = np.random.uniform() < p_mutation

        # Storing the results
        transmission_dict[ind] = {'source_label': source_label, 'target_label': target_label, 'was_reinfection': was_reinfection,
                                  'substitution_occurred': substitution_occurred}

    return transmission_dict


##### MODELLING SUBSTITUTIONS ACROSS TREE

def get_substitution_tree(transmission_dict):

    # Creating an array of possible subsitutions
    sub_names = list(string.ascii_lowercase)[1:] + list(string.ascii_uppercase) + list(np.arange(10000).astype(str))
    sub_counter = 0
    
    # Creating a directed graph to store the substitution tree
    sub_tree = nx.DiGraph()
    
    # Determining the time of the first infection and first node
    t_zero = list(transmission_dict.keys())[0]
    first_node = transmission_dict[t_zero]['source_label']
    
    # Initialising a variable to hold the infection history
    pathogen_name_dict = dict()
    pathogen_name_dict[t_zero] = 'a'
    
    # Adding the first node to the tree
    sub_tree.add_node(first_node, time_infected=[t_zero], reinfection=[False], pathogen_name_hist=pathogen_name_dict, current_pathogen_name='a')
    
    # Looping through the transmission data in order of increasing time
    for t, data in transmission_dict.items():
    
        # Extracting the data from the current event
        source_label, target_label, reinfection_occurred, substitution_occurred = [value for key, value in data.items()]
    
        # Extracting the source pathogen name at the timestep
        source_pathogen = sub_tree.nodes(data=True)[source_label]['current_pathogen_name']
        target_pathogen = source_pathogen
        
        # Checking if a substitution occurred during the current event
        if substitution_occurred:

            # Updating the pathogen name to include the substitution
            target_pathogen += '_%s' % sub_names[sub_counter]
            sub_counter += 1
    
        # Adding the target label if it does not yet exist
        if not sub_tree.has_node(target_label):
    
            # Initialising a variable to hold the infection history
            pathogen_name_dict = dict()
            pathogen_name_dict[t] = target_pathogen

            # Adding the node and attributes
            sub_tree.add_node(target_label, time_infected=[t], reinfection=[reinfection_occurred],
                              pathogen_name_hist=pathogen_name_dict, current_pathogen_name=target_pathogen)
    
        else:

            # Updating the node data
            sub_tree.nodes()[target_label]['time_infected'].append(t)
            sub_tree.nodes()[target_label]['reinfection'].append(reinfection_occurred)
            sub_tree.nodes()[target_label]['pathogen_name_hist'][t] = target_pathogen
            sub_tree.nodes()[target_label]['current_pathogen_name'] = target_pathogen

        # Adding edge between source and target nodes
        sub_tree.add_edge(source_label, target_label)

    return sub_tree


##### PRODUCING THE PHYLOGENETIC TREE

def get_unique_pathogens(sub_tree):
    
    # Determining all possible pathogens
    possible_pathogens = []
    
    # Looping through all nodes in the tree
    for node, data in sub_tree.nodes(data=True):
    
        # Adding the variants encountered by the current node
        possible_pathogens.extend([value for key, value in data['pathogen_name_hist'].items()])
    
    # Determining all unique pathogens
    possible_pathogens = np.array(possible_pathogens)
    unique_pathogens = np.unique(possible_pathogens)

    return unique_pathogens

    
def get_phylogenetic_tree(sub_tree):

    # Creating a graph to store the phylogenetic data
    phylo_tree = nx.Graph()

    # Determining all unique pathogens
    unique_pathogens = get_unique_pathogens(sub_tree=sub_tree)

    # Looping through the unique pathogens
    for pathogen in unique_pathogens:

        # Substitutions are denoted by the final instance of '_' (eg '1' becomes '1_2', but '1_2_3' is a mutation of '1_2')
        substitutions = pathogen.split('_')

        # Looping through the data up to each instance of '_'
        for i in range(1, len(substitutions)):

            # Adding an edge between the the old variant and new mutation (character after the final instance '_')
            phylo_tree.add_edge('_'.join(substitutions[:i]), '_'.join(substitutions[:i + 1]))

    return phylo_tree


##### INCORPORATING CROSS IMMUNITY

def get_previous_pathogen_data(path_hist, new_path, t, phylo_tree):
    
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

            
def get_cross_immunity_tree(transmission_dict, sub_tree, phylo_tree):

    # Creating a copy of the sub tree
    cross_immunity_tree = sub_tree.copy()
    
    # Looping through transmission tree in order of increasing time
    for t, data in transmission_dict.items():
        
        # Extracting the data from the current event
        source_label, target_label, reinfection_occurred, substitution_occurred = [value for key, value in data.items()]
    
        # Checking if event was a reinfection
        if reinfection_occurred:

            # Skipping node if it no longer exists
            if not cross_immunity_tree.has_node(target_label):
                continue
            
            # Finding the corresponding node in the cross-immunity tree
            current_node = cross_immunity_tree.nodes(data=True)[target_label]
    
            # Extracting all pathogens encountered by the node
            pathogen_history = current_node['pathogen_name_hist']
            potential_new_pathogen = pathogen_history[t]

            # Determining the distance to each previously encountered pathogen
            previous_pathogens, phylogenic_distances = get_previous_pathogen_data(path_hist=pathogen_history, new_path=potential_new_pathogen,
                                                                                  t=t, phylo_tree=phylo_tree)
            # Finding the shortest distance and immunity probability
            shortest_dist = np.min(phylogenic_distances)
            immunity_prob = get_immunity_probability(dist=shortest_dist)

            # Testing against sigmoid
            if np.random.uniform() < immunity_prob:
                
                # Removing all onward nodes
                cross_immunity_tree.remove_node(target_label)

    return cross_immunity_tree


##### MAIN

# Creating a transmission dictionary from the infection data
inf_df = get_infection_df(iter_num=0)
transmission_dict = get_transmission_dict(infection_df=inf_df)

# Simulating substitutions accross the tree
substitution_tree = get_substitution_tree(transmission_dict=transmission_dict)

# Creating the phylogenetic tree
phylogenetic_tree = get_phylogenetic_tree(sub_tree=substitution_tree)

# Implementing cross immunity if required
cross_immunity_tree = substitution_tree 

if include_cross_immunity:
    cross_immunity_tree = get_cross_immunity_tree(transmission_dict=transmission_dict, sub_tree=substitution_tree, phylo_tree=phylogenetic_tree)