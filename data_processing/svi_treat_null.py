# import libraries

import os
import pandas as pd
import geopandas as gpd
import libpysal as ps
import numpy as np
from tqdm import tqdm


class fill_null_standerdize_svi_usa():
    def __init__(self, svi_data_path, variables, year, save_dir):
        self.svi_data_path = svi_data_path
        self.variables = variables
        self.save_dir = save_dir
        self.year = year
        self.svi_null_treated = None

    def filter_trear_null(self):

        file_name = f'/USA_SVI_{self.year}_cleaned_null_treated.gdb'
        full_path = self.save_dir + file_name

        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return

        print("Processing SVI null treatment...")

        # read the svi geojson file
        svi = gpd.read_file(self.svi_data_path)

        variables_with_censusinfo = ['FIPS', 'STCNTY'] + self.variables + ['geometry'] +['ST_ABBR','COUNTY']
        
        # filter the data
        svi = svi[variables_with_censusinfo]

        svi = gpd.GeoDataFrame(svi, geometry="geometry")

        # Compute Queen adjacency
        w = ps.weights.Queen.from_dataframe(svi, ids=svi["FIPS"])

        # if we keep the islands it will cause error in the adjacency computation(empty adjacencies)
        # Get the islands fips to a list
        islands = w.islands

        # drop the islands
        svi = svi[~svi["FIPS"].isin(islands)]

        # Store adjacent county FIPS in a new column
        svi["adjacent_fips"] = svi["FIPS"].apply(lambda fips: w.neighbors.get(fips, []))

        # replace negative values with the average of the adjacent counties without negative adjacent values
        for county in tqdm(svi['FIPS'].unique(), desc="Processing Counties"):
            for variable in self.variables:

                if svi.loc[svi['FIPS'] == county, variable].values[0] < 0:

                    # get those adjacent counties variable values to a list
                    adjacent_fips = svi.loc[svi['FIPS'] == county, 'adjacent_fips'].values[0]

                    if len(adjacent_fips) == 0:
                        svi.loc[svi['FIPS'] == county, variable] = 0
                    else:
                        # get the adjacent values for the variable
                        adjacent_values = []
                        for adjacent_county in adjacent_fips:
                            adjacent_values.append(svi.loc[svi['FIPS'] == adjacent_county, variable].values[0])

                        # get the average of the adjacent values without the negative values
                        adjacent_values = [value for value in adjacent_values if value >= 0]

                        if len(adjacent_values) == 0:
                            svi.loc[svi['FIPS'] == county, variable] = 0
                        else:
                            average_adjacent_value = np.mean(adjacent_values)
                            svi.loc[svi['FIPS'] == county, variable] = average_adjacent_value
                else:
                    continue

        # drop the adjacent_fips column
        svi = svi.drop(columns=['adjacent_fips'])

        # save the data
        svi.to_file(full_path)
        self.svi_null_treated = svi
        print(f"File saved as {file_name}")


    # standerdize for each state and variable
    def minmax_scale_svi_state(self, state_abbr):
        
        file_name = f'/USA_SVI_{self.year}_{state_abbr}_cleaned_scaled.gdb'
        full_path = self.save_dir + file_name

        if os.path.exists(full_path):
            print(f"File already exists: {full_path}")
            return

        if self.svi_null_treated is None:
            # Check if the null-treated data is available (i.e., if filter_trear_null() has been run separately)
            svi_cleaned_path = self.save_dir + f'/USA_SVI_{self.year}_cleaned_null_treated.gdb'

            if os.path.exists(svi_cleaned_path):
                print("Loading existing SVI null-treated data...")
                self.svi_null_treated = gpd.read_file(svi_cleaned_path)
            else:
                print("Error: SVI null-treated data not available. Run `filter_trear_null()` first.")
                return

        print(f"Scaling SVI data for {state_abbr}...")

        svi_state = self.svi_null_treated[self.svi_null_treated['ST_ABBR'] == state_abbr]

        for variable in self.variables:
            min_val = svi_state[variable].min()
            max_val = svi_state[variable].max()
            svi_state[variable] = (svi_state[variable] - min_val) / (max_val - min_val)

        svi_state.to_file(full_path)
        print(f"File saved as {full_path}")





