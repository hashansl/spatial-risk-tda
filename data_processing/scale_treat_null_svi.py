# import libraries
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

    def filter_clean_gdf(self):
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

        # min max scaling for each variable
        for variable in self.variables:
            svi[variable] = (svi[variable] - svi[variable].min()) / (svi[variable].max() - svi[variable].min())

        # save the file as a gdb in save_dir folder
        file_name =  f'/USA_SVI_{self.year}_cleaned.gdb'
        svi.to_file(self.save_dir + file_name)

        print(f"File saved as {file_name}")

# run the main function
if __name__ == "__main__":
    svi_data_path = "/Users/h6x/ORNL/git/WORKSTAION GIT/nvss-experiments/data/SVI2018_US_tract.gdb"
    save_dir = "/Users/h6x/ORNL/git/WORKSTAION GIT/nvss-experiments/usa_models/data/processed"

    variables = [
    'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
    'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
    year = "2018"

    svi = fill_null_standerdize_svi_usa(svi_data_path, variables, year, save_dir)
    svi.filter_clean_gdf()


