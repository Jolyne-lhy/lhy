# -*- coding: utf-8 -*-
"""
Dataset Generator
Copyright: (C) 2019, KIOS Research Center of Excellence
"""
import pandas as pd
import numpy as np
import wntr
import pickle
import os
import sys
import yaml
import shutil
import time
from math import sqrt
# import gc # Modified by LHY

# Read input arguments from yalm file
try:
    with open(f'{os.getcwd()}/config.yaml', 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.Loader)
except:
    print('"dataset_configuration" file not found.')
    sys.exit()

start_time = cfg['times']['StartTime']
end_time = cfg['times']['EndTime']
leakages = cfg['leakages']
leakages = leakages[1:]
number_of_leaks = len(leakages)
inp_file = cfg['Network']['filename']
print(f'Run input file: "{inp_file}"')
results_folder = f'{os.getcwd()}/SimulateResults/'
pressure_sensors = cfg['pressure_sensors']

# dma_c_pipe = cfg['DMA_C_PIPE']
zone_list = ['zone1', 'zone2', 'zone3','zone4','zone5','zone6','zone7','zone8','zone9','zone10']

def get_sensors(leak_pipes, field):
    sensors = []
    [sensors.append(str(sens)) for sens in leak_pipes[field]]
    return sensors

flow_sensors = get_sensors(cfg, 'flow_sensors')
pressure_sensors = get_sensors(cfg, 'pressure_sensors')
amrs = get_sensors(cfg, 'amrs')
flow_sensors = get_sensors(cfg, 'flow_sensors')
level_sensors = get_sensors(cfg, 'level_sensors')

# demand-driven (DD) or pressure dependent demand (PDD)
Mode_Simulation = 'PDD'  # 'PDD'#'PDD'


def create_folder(_path_):
    try:
        if os.path.exists(_path_):
            shutil.rmtree(_path_)
        os.makedirs(_path_)
    except Exception as error:
        pass


class LeakDatasetCreator:
    def __init__(self):

        # Load EPANET network file
        self.wn = wntr.network.WaterNetworkModel(inp_file)

        for name, node in self.wn.junctions():
            node.required_pressure = 25
            #print(node.nominal_pressure)
            #print(node.minimum_pressure)

        self.inp = os.path.basename(self.wn.name)[0:-4]

        # Get the name of input file
        self.net_name = f'{results_folder}{self.inp}'

        # Get time step
        self.time_step = round(self.wn.options.time.hydraulic_timestep)
        # Create time_stamp
        self.time_stamp = pd.date_range(start_time, end_time, freq=str(self.time_step / 60) + "min")

        # Simulation duration in steps
        self.wn.options.time.duration = (len(self.time_stamp) - 1) * 300  # 5min step
        self.TIMESTEPS = int(self.wn.options.time.duration / self.wn.options.time.hydraulic_timestep)

    def create_csv_file(self, values, time_stamp, columnname, pathname):

        file = pd.DataFrame(values)
        file['Timestamp'] = time_stamp
        file = file.set_index(['Timestamp'])
        file.columns.values[0] = columnname
        file.to_csv(pathname)
        del file, time_stamp, values


    def dataset_generator(self, leak_i, msg):
        # Path of EPANET Input File
        print(f"{msg} Dataset Generator run {leak_i}...")

        # Initialize parameters for the leak

        # leakages: startTime, endTime, leakDiameter
        leakage_line = leakages[0].split(',')

        # Start time of leak
        ST = self.time_stamp.get_loc(leakage_line[0])

        # End Time of leak
        ET = self.time_stamp.get_loc(leakage_line[1])

        leak_diameter = float(leakage_line[2])
        leak_area = 3.14159 * (leak_diameter / 2) ** 2

        leak_starts = self.time_stamp[ST]
        leak_starts = leak_starts._date_repr + ' ' + leak_starts._time_repr
        leak_ends = self.time_stamp[ET]
        leak_ends = leak_ends._date_repr + ' ' + leak_ends._time_repr

        #self.wn.write_inpfile(f'{leakages_folder}\\{self.inp}_with_leaknodes.inp')


        # Save the water network model to a file before using it in a simulation
        with open('self.wn.pickle', 'wb') as f:
            pickle.dump(self.wn, f)
        


        # Split pipe and add a leak node
        for j in range(1, number_of_leaks):
            leak_j_info = leakages[j].split(',')
            leak_j_pipe = leak_j_info[0]
            leak_j_diameter = float(leak_j_info[1])
            leak_j_area = 3.14159 * (leak_j_diameter / 2) ** 2

            pipe_id = self.wn.get_link(leak_j_pipe)
            node_leak = f'{pipe_id}'
            self.wn = wntr.morph.split_pipe(self.wn, pipe_id, f'{pipe_id}_Bleak', node_leak)
            leak_node = self.wn.get_node(self.wn.node_name_list[self.wn.node_name_list.index(node_leak)])

            leak_node.add_leak(self.wn, discharge_coeff=0.75,
                                    area=leak_j_area,
                                    start_time=self.time_stamp.get_loc(start_time) * self.time_step,
                                    end_time=(self.time_stamp.get_loc(end_time)+1) * self.time_step)

        # Add a leak on exist node (without spliting pipe)
        leak_node = self.wn.get_node(leak_i)

        leak_node._leak_end_control_name = str(leak_i) + 'end'
        leak_node._leak_start_control_name = str(leak_i) + 'start'

        leak_node.add_leak(self.wn, discharge_coeff=0.75,
                                    area=leak_area,
                                    start_time=ST * self.time_step,
                                    end_time=(ET+1) * self.time_step)
        

        # Run wntr simulator
        self.wn.options.hydraulic.demand_model = Mode_Simulation
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1
        

        if results:
            decimal_size = 2

            leaks = results.node['leak_demand'][str(leak_node)].values
            # Convert m^3/s (wntr default units) to m^3/h
            # https://wntr.readthedocs.io/en/latest/units.html#epanet-unit-conventions
            leaks = [elem * 3600 for elem in leaks]
            leaks = [round(elem, decimal_size) for elem in leaks]
            leaks = leaks[:len(self.time_stamp)]

            self.create_csv_file(leaks, self.time_stamp, str(leak_node), f'{demand_folder}/Leak_{str(leak_node)}_demand.csv')

            # Create csv file with Measurements
            total_pressures = {'Timestamp': self.time_stamp}

            for j in range(0, self.wn.num_nodes):
                node_id = self.wn.node_name_list[j]

                if node_id in pressure_sensors:
                    pres = results.node['pressure'][node_id]
                    pres = pres[:len(self.time_stamp)]
                    pres = [round(elem, decimal_size) for elem in pres]
                    total_pressures[node_id] = pres

            # Create a Pandas dataframe from the data.
            df = pd.DataFrame(total_pressures)
            df.to_csv(f'{pressure_folder}/Leak_{str(leak_node)}_pressure.csv', index=False)
        else:
            print('Results empty.')
            return -1
        
        leak_node.remove_leak(self.wn)
        self.wn.reset_initial_values()

        os.remove('self.wn.pickle')

    
    def normal_generator(self):
        # Path of EPANET Input File
        print(f"Normal Generator run...")

        # Save the water network model to a file before using it in a simulation
        with open('self.wn.pickle', 'wb') as f:
            pickle.dump(self.wn, f)

        # Run wntr simulator
        self.wn.options.hydraulic.demand_model = Mode_Simulation
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1
        

        if results:
            decimal_size = 2

            # Create csv file with Measurements
            total_pressures = {'Timestamp': self.time_stamp}

            for j in range(0, self.wn.num_nodes):
                node_id = self.wn.node_name_list[j]

                if node_id in pressure_sensors:
                    pres = results.node['pressure'][node_id]
                    pres = pres[:len(self.time_stamp)]
                    pres = [round(elem, decimal_size) for elem in pres]
                    total_pressures[node_id] = pres

            # Create a Pandas dataframe from the data.
            df = pd.DataFrame(total_pressures)
            df.to_csv(f'{normal_folder}/Normal_pressure.csv', index=False)

        os.remove('self.wn.pickle')



if __name__ == '__main__':
    # Create Results folder
    create_folder(results_folder)

    for zone in zone_list:
        # Create tic / toc
        t = time.time()

        zone_folder = f'{results_folder}{zone}/'
        create_folder(zone_folder)

        leakages_folder = f'{zone_folder}Leakages/'
        normal_folder = f'{zone_folder}Normal/'

        create_folder(leakages_folder)
        create_folder(normal_folder)

        demand_folder = f'{leakages_folder}Demand'
        pressure_folder = f'{leakages_folder}Pressure'

        create_folder(demand_folder)
        create_folder(pressure_folder)


        for i, node in enumerate(cfg[zone]):
            # Call leak dataset creator
            L = LeakDatasetCreator()
            # Create scenario one-by-one
            L.dataset_generator(node, f'[{i+1}/{len(cfg[zone])}]')

        # Call leak dataset creator
        L = LeakDatasetCreator()
        # Create scenario one-by-one
        L.normal_generator()

        print(f'Total Elapsed time is {str(time.time() - t)} seconds.')


