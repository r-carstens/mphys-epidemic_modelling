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

# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting a mutation rate
p_mutation = 0.3

# Setting required number of mutations for new variant
req_n_mutations = 3


##### ANALYSING MUTATIONS

def initialise_transmission_tree(iter_num):

    # Determining input file name
    in_file = complete_path + '_%s.txt' % (iter_num + 1)
    
    # Importing transmission data and only retaining infection events
    df = pd.read_csv(in_file, delimiter=',', skiprows=1)
    infection_df = df.loc[(df['source_during'] == infected) & (df['target_before'] == susceptible) & (df['target_after'] == infected)]

    # Creating directed graph
    transmission_tree = nx.DiGraph()

    # Adding connections in order at which infections occurred (with increasing simulation time)
    for ind in infection_df.index:
        transmission_tree.add_edge(infection_df['source_label'][ind], infection_df['target_label'][ind])

    # Adding attributes to each node
    attributes = {'has_mutated': False, 'strain': '', 'variant': ''}
    nx.set_node_attributes(transmission_tree, {node: attributes for node in transmission_tree.nodes})

    # Accessing the node that caused the first infection and giving it variant 'a'
    topological_list = list(nx.topological_sort(transmission_tree))
    transmission_tree.nodes()[topological_list[0]]['strain'] = 'a'
    transmission_tree.nodes()[topological_list[0]]['variant'] = 'a'

    return transmission_tree
    

def simulate_mutations(transmission_tree):

    # Looping in order of infections but skipping the first (mutations are assumed to occur from the second infection as first is wild type)
    for node in list(nx.topological_sort(transmission_tree))[1:]:

        # Checking if a mutation occurs at the point of the node
        if np.random.uniform() < p_mutation:
            transmission_tree.nodes()[node]['has_mutated'] = True

    return transmission_tree


def simulate_strains(transmission_tree):

    # Initialising strain names and counter to allow for a new character to be chosen
    strain_characters = list(string.ascii_lowercase)[1:] + list(string.ascii_uppercase) + list(np.arange(10000).astype(str))
    mutation_counter = 0

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):

        # Checking if a mutation has occurred within the current node
        if transmission_tree.nodes()[node]['has_mutated']:

            # Updating current node's strain name if required
            transmission_tree.nodes()[node]['strain'] += '_' + strain_characters[mutation_counter]
            mutation_counter += 1

        # Looping through the nodes directly infected by the current node
        for successor in list(transmission_tree.successors(node)):

            # Infecting immediate successors with the current node's strain
            transmission_tree.nodes()[successor]['strain'] = transmission_tree.nodes()[node]['strain']

    return transmission_tree


def simulate_variants(transmission_tree):

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):
        
        # Splitting the current strain mutations
        mutations = transmission_tree.nodes()[node]['strain'].split('_')

        # Checking if a variant emerges
        if len(mutations) >= req_n_mutations:

            # Creating a variable to store the newly emerged variant
            new_variant = ''
            
            # Determining the number of mutation clusters
            num_mutation_clusters = len(mutations) // req_n_mutations

            # Looping through each cluster
            for i in range(num_mutation_clusters):

                # Creating an array containing mutation clusters and adding each cluster to variant name
                current_cluster = mutations[i * req_n_mutations : (i + 1) * req_n_mutations]
                new_variant += ''.join(current_cluster)

            # Checking for remaining mutations
            if len(mutations) % req_n_mutations != 0:

                # Adding remaining mutations to variant name
                new_variant += '_'
                new_variant += '_'.join(mutations[num_mutation_clusters * req_n_mutations:])

            # Updating host node's variant
            transmission_tree.nodes()[node]['variant'] = new_variant

        # Looping through the nodes directly infected by the current node
        for successor in list(transmission_tree.successors(node)):

            # Infecting immediate successors with the current node's strain
            transmission_tree.nodes()[successor]['variant'] = transmission_tree.nodes()[node]['variant']

    return transmission_tree
            
        
def save_variant_transmission_data(transmission_tree, iter_num):

    # Creating a dataframe to store the results
    variant_df = pd.DataFrame()

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):

        # Accessing useful information
        successors = list(transmission_tree.successors(node))
        has_mutated, strain_name, variant_name = [attribute for attribute in transmission_tree.nodes()[node].values()]

        # Creating df entry
        entry = {'node': node, 'mutation_occurred': has_mutated, 'strain_infected_by': strain_name, 'variant_infected_by': variant_name, 'successors': successors}

        # Adding data to the df
        variant_df = variant_df._append(entry, ignore_index=True)

    # Writing the dataframe to the outfile
    out_file = 'variants_outfile_%s.txt' % (iter_num + 1)
    variant_df.to_csv(out_file, sep='\t', index=False)


def analyse_variant_transmission(iter_num):

    # Initialising the transmission tree without variants
    transmission_tree = initialise_transmission_tree(iter_num=iter_num)

    # Simulating variant mutations
    transmission_tree = simulate_mutations(transmission_tree)

    # Simulating strain transmission
    transmission_tree = simulate_strains(transmission_tree)

    # Simulating variant transmission
    transmission_tree = simulate_variants(transmission_tree)

    # Checking if the produced transmission tree is directed and acyclic
    if not nx.is_directed_acyclic_graph(transmission_tree):
        print('Error: Transmission tree from file %s is not DAG' % iter_num)

    # Saving the results
    save_variant_transmission_data(transmission_tree, iter_num=iter_num)

    return transmission_tree


##### ANALYSING VARIANT EVOLUTION

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


def save_variant_evolution_figure(evolution_tree, variant_dict, iter_num):

    # Drawing the graph as an evolutionary tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title('Variant evolution for mutation probability=%s' % p_mutation)
    nx.draw(evolution_tree, pos=graphviz_layout(evolution_tree, prog='dot'), with_labels=True, ax=ax)
    plt.savefig('evolution_image_%s.jpg' % (iter_num + 1))


def save_variant_evolution_data(evolution_tree, variant_dict, iter_num):

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
    out_file = 'evolution_outfile_%s.txt' % (iter_num + 1)
    evolution_df.to_csv(out_file, sep='\t', index=False)


def simulate_variant_evolution(variant_transmission_tree, iter_num):

    # Creating a variant evolutionary tree
    variant_evolution_tree, variant_dict = get_variant_evolution_tree(transmission_tree=variant_transmission_tree)

    # Saving the results
    save_variant_evolution_figure(evolution_tree=variant_evolution_tree, variant_dict=variant_dict, iter_num=iter_num)
    save_variant_evolution_data(evolution_tree=variant_evolution_tree, variant_dict=variant_dict, iter_num=iter_num)


def repeat_measurements(num_iterations=1):

    # Looping through required number of repeated measurements
    for n in tqdm(range(num_iterations)):

        # Completing variant transmission simulation
        variant_transmission_tree = analyse_variant_transmission(iter_num=n)

        # Completing variant evolution simulation
        simulate_variant_evolution(variant_transmission_tree=variant_transmission_tree, iter_num=n)


##### MAIN

# Setting number of measurement iterations
n_iterations = 1

# Running repeated measurements
repeat_measurements(num_iterations=n_iterations)
