import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import matplotlib.pyplot as plt
import string
import os

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting a mutation rate
p_mutation = 0.1


##### INITIALISING THE TRANSMISSION TREE

def initialise_transmission_tree(in_file):

    # Importing transmission data and only retaining infection events
    df = pd.read_csv(in_file, delimiter=',', skiprows=1)
    infection_df = df.loc[(df['source_during'] == infected) & (df['target_before'] == susceptible) & (df['target_after'] == infected)]

    # Creating directed graph
    transmission_tree = nx.DiGraph()

    # Adding connections in order at which infections occurred (with increasing simulation time)
    for ind in infection_df.index:
        transmission_tree.add_edge(infection_df['source_label'][ind], infection_df['target_label'][ind])

    # Adding attributes to each node
    attributes = {'has_mutated': False, 'variant': ''}
    nx.set_node_attributes(transmission_tree, {node: attributes for node in transmission_tree.nodes})

    # Accessing the node that caused the first infection and giving it variant 'a'
    topological_list = list(nx.topological_sort(transmission_tree))
    transmission_tree.nodes()[topological_list[0]]['variant'] = 'a'

    return transmission_tree


##### SIMULATING AND SAVING VARIANT TRANSMISSION THROUGHOUT THE POPULATION

def simulate_variant_mutations(transmission_tree):

    # Looping in order of infections but skipping the first (mutations are assumed to occur from the second infection)
    for node in list(nx.topological_sort(transmission_tree))[1:]:

        # Checking if a mutation occurs at the point of the node
        if np.random.uniform() < p_mutation:
            transmission_tree.nodes()[node]['has_mutated'] = True

    return transmission_tree


def get_variant_transmission_tree(transmission_tree):

    # Simulating variant mutations
    transmission_tree = simulate_variant_mutations(transmission_tree)

    # Initialising variant names and counter to allow for a new character to be chosen
    variant_characters = list(string.ascii_lowercase)[1:] + list(string.ascii_uppercase) + list(np.arange(10000).astype(str))
    mutation_counter = 0

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):

        # Checking if a mutation has occurred within the current node
        if transmission_tree.nodes()[node]['has_mutated']:

            # Updating current node's variant name if required
            transmission_tree.nodes()[node]['variant'] += '_' + variant_characters[mutation_counter]
            mutation_counter += 1

        # Looping through the nodes directly infected by the current node
        for successor in list(transmission_tree.successors(node)):

            # Infecting immediate successors with the current node's variant
            transmission_tree.nodes()[successor]['variant'] = transmission_tree.nodes()[node]['variant']

    return transmission_tree


def save_variant_transmission_data(transmission_tree, out_file):

    # Creating a dataframe to store the results
    variant_df = pd.DataFrame()

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):

        # Accessing useful information
        successors = list(transmission_tree.successors(node))
        has_mutated, variant_name = [attribute for attribute in transmission_tree.nodes()[node].values()]

        # Creating df entry
        entry = {'node': node, 'mutation_occurred': has_mutated, 'infected_by': variant_name, 'successors': successors}

        # Adding data to the df
        variant_df = variant_df._append(entry, ignore_index=True)

    # Writing the dataframe to the outfile
    variant_df.to_csv(out_file, sep='\t', index=False)


##### SIMULATING AND SAVING VARIANT EVOLUTION

def get_variant_evolution_tree(transmission_tree):

    # Creating a directed graph and determining the possible variants
    evolution_graph = nx.DiGraph()
    all_variants = [transmission_tree.nodes()[node]['variant'] for node in transmission_tree.nodes()]

    # Determining the unique variant names and indexing into original order (np.unique does not conserve order)
    unique_unsorted_variants, original_locations = np.unique(all_variants, return_index=True)
    unique_variants = unique_unsorted_variants[np.argsort(original_locations)]

    # Looping through each unique variant in the dataset
    for variant in unique_variants:

        # Mutations are denoted by the final instance of '_' (eg '1' becomes '1_2', but '1_2_3' is a mutation of '1_2')
        mutations = variant.split('_')

        # Looping through the data up to each instance of '_'
        for i in range(1, len(mutations)):

            # Adding an edge between the the old variant and new mutation (character after the final instance '_')
            evolution_graph.add_edge('_'.join(mutations[:i]), '_'.join(mutations[:i + 1]))

    # Creating a dictionary to store the labels and relabelling the nodes
    variant_label_dict = dict(zip(unique_variants, np.arange(len(unique_variants))))
    evolution_graph = nx.relabel_nodes(evolution_graph, variant_label_dict)

    return evolution_graph, variant_label_dict


def save_variant_evolution_figure(evolution_tree, variant_dict, out_file):

    # Drawing the graph as an evolutionary tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title('Variant evolution for mutation probability=%s' % p_mutation)
    nx.draw(evolution_tree, pos=graphviz_layout(evolution_tree, prog='dot'), with_labels=True, ax=ax)
    plt.savefig(out_file)


def save_variant_evolution_data(evolution_tree, variant_dict, out_file):

    # Creating a dataframe to store the results
    evolution_df = pd.DataFrame()

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(evolution_tree)):

        # Determining variant name
        variant_name = [key for key, value in variant_dict.items() if value == node]

        # Accessing connected node information
        ancestors = list(nx.ancestors(evolution_tree, node))
        successors = list(evolution_tree.successors(node))

        # Creating df entry
        entry = {'node': node, 'variant_name': variant_name, 'num_mutations': len(ancestors), 'ancestors': ancestors, 'successors': successors}

        # Adding data to the df
        evolution_df = evolution_df._append(entry, ignore_index=True)

    # Writing the dataframe to the outfile
    evolution_df.to_csv(out_file, sep='\t', index=False)


##### RUNNING REPEATED SIMULATIONS

def simulate_variant_transmission(n, input_path, output_path):

    # Initialising the transmission tree without variants
    infection_df_in_file_name = input_path + '_%s.txt' % n
    transmission_tree = initialise_transmission_tree(in_file=infection_df_in_file_name)

    # Checking if the produced transmission tree is directed and acyclic
    if not nx.is_directed_acyclic_graph(transmission_tree):
        print('Error: Transmission tree from file %s is not DAG' % n)

    # Simulating variant mutations across the tree
    variant_transmission_tree = get_variant_transmission_tree(transmission_tree=transmission_tree)

    # Saving the results
    transmission_out_file_name = output_path + '\\variants_outfile_%s.txt' % n
    save_variant_transmission_data(transmission_tree=variant_transmission_tree, out_file=transmission_out_file_name)

    return transmission_tree


def simulate_variant_evolution(variant_transmission_tree, n, output_path):

    # Creating a variant evolutionary tree
    variant_evolution_tree, variant_dict = get_variant_evolution_tree(transmission_tree=variant_transmission_tree)

    # Creating file names to save the results to
    image_name = output_path + '\\evolution_image_%s.jpg' % n
    file_name = output_path + '\\evolution_outfile_%s.txt' % n

    # Saving the results
    save_variant_evolution_figure(evolution_tree=variant_evolution_tree, variant_dict=variant_dict, out_file=image_name)
    save_variant_evolution_data(evolution_tree=variant_evolution_tree, variant_dict=variant_dict, out_file=file_name)


def repeat_measurements(in_path, out_path, num_iterations=1):

    # Looping through required number of repeated measurements
    for n in range(num_iterations):

        # Creating subdirectory path to store current output data in
        current_output_directory = out_path + '\\variant_simulation_%s' % n

        # Creating the output subdirectory if it does not already exist
        if not os.path.isdir(current_output_directory):
            os.mkdir(current_output_directory)

        # Completing variant transmission simulation
        variant_transmission_tree = simulate_variant_transmission(n=n, input_path=in_path, output_path=current_output_directory)

        # Completing variant evolution simulation
        simulate_variant_evolution(variant_transmission_tree=variant_transmission_tree, n=n, output_path=current_output_directory)

        # Displaying progress
        print(n)


##### MAIN

in_file_path = 'simulation_data\\individual_sim_outfile'
out_file_path = 'variant_data'

# Checking if directory containing data exists
if not os.path.isdir('simulation_data'):
    print('No data found, requires individual simulation data files to be placed in a directory called simulation_data')

# Creating a directory to store variant data in
if not os.path.isdir(out_file_path):
    os.mkdir(out_file_path)

# Running repeated measurements
repeat_measurements(in_path=in_file_path, out_path=out_file_path, num_iterations=2)
