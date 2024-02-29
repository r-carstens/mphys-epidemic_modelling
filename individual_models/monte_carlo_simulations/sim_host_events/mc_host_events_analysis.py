import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Creating out file names to be used for storing and reading data
event_path = 'host_event_outfile'
mcs_path = 'mcs_host_event_data'
complete_path = 'complete_host_event_data'


##### CREATING USEFUL DATAFRAMES

def get_simulation_parameters(path):

    # Creating a structure to store the simulation parameters
    parameters_dict = dict()

    # Locating all data files within the directory
    sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(path)]

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
    
    
def get_events_dataframe(path):

    # Reading in the data
    current_df = pd.read_csv(path + '.txt', delimiter=',')

    return current_df
    

def get_results_dataframe(path):

    # Creating a dataframe to store all results from all files
    results_df = pd.DataFrame()

    # Locating all data files within the directory
    sim_data_files = [file for file in os.listdir(os.getcwd()) if file.startswith(path)]

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

# Accessing the simulation parameters
parameters_dict = get_simulation_parameters(path=complete_path)

# Creating useful dataframes
events_df = get_events_dataframe(path=event_path)
results_df = get_results_dataframe(path=complete_path)
susceptible_df, infected_df, immune_df = get_state_dataframes(results_df=results_df)

# Plotting the data
plot_state_totals(susceptible_df=susceptible_df, infected_df=infected_df, immune_df=immune_df, events_df=events_df, parameters=parameters_dict)