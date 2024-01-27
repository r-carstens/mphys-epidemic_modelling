import numpy as np
import networkx as nx
import pandas as pd
import string


# Initialising possible infection states
susceptible = 'S'
infected = 'I'
immune = 'M'

# Setting a mutation rate
p_mutation = 0.1


##### INITIALISING THE TRANSMISSION TREE

def initialise_tree(infection_df):

    # Creating directed graph
    transmission_tree = nx.DiGraph()

    # Adding connections in order at which infections occurred (with increasing simulation time)
    for ind in infection_df.index:
        transmission_tree.add_edge(infection_df['source_label'][ind], infection_df['target_label'][ind])

    # Adding attributes to each node
    attributes = {'has_mutated': False, 'variant': 'a'}
    nx.set_node_attributes(transmission_tree, {node: attributes for node in transmission_tree.nodes})

    return transmission_tree


##### SIMULATING MUTATIONS

def check_for_mutation():

    # Checking if a mutation occurs for a given probability
    return np.random.uniform() < p_mutation


def simulate_mutations(transmission_tree):

    # Looping through each node in the graph
    for node in transmission_tree.nodes():

        # Checking if a mutation occurs at the point of the node
        if check_for_mutation():
            transmission_tree.nodes()[node]['has_mutated'] = True

    return transmission_tree


def simulate_variant_transmission(transmission_tree):

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


##### SAVING THE RESULTS

def save_variant_graph(transmission_tree, out_file_name):

    # Creating a dataframe to store the results
    variant_df = pd.DataFrame()

    # Looping through the graph in order of infections
    for node in list(nx.topological_sort(transmission_tree)):

        # Accessing useful information
        successors = list(transmission_tree.successors(node))
        has_mutated, variant_name = [attribute for attribute in transmission_tree.nodes()[node].values()]

        # Creating df entry
        entry = {'node': node, 'mutation_occurred': has_mutated, 'variant_name': variant_name, 'successors': successors}

        # Adding data to the df
        variant_df = variant_df._append(entry, ignore_index=True)

    # Writing the dataframe to the outfile
    variant_df.to_csv(out_file_name, sep='\t', index=False)


##### MAIN

# Importing transmission data and only retaining infection events
df = pd.read_csv('individual_sim_outfile.txt', delimiter=',', skiprows=1)
infection_df = df.loc[(df['source_during'] == infected) & (df['target_before'] == susceptible) & (df['target_after'] == infected)]

# Creating the graph and checking it is directed and acyclic
transmission_tree = initialise_tree(infection_df)
print('Graph is DAG: %s' % nx.is_directed_acyclic_graph(transmission_tree))

# Simulating infectious disease mutations
transmission_tree = simulate_mutations(transmission_tree)
transmission_tree = simulate_variant_transmission(transmission_tree)

# Saving the data
out_file = 'variants_outfile.txt'
save_variant_graph(transmission_tree, out_file_name=out_file)
