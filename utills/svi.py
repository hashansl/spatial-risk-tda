import pandas as pd
import geopandas as gpd
import libpysal as ps
import numpy as np
from tqdm import tqdm

import utills.invr as invr
from utills.adjacency_simplex import AdjacencySimplex  # Import the class
from utills.calculate_tda_summaries import compute_persistence

# import invr as invr
# from adjacency_simplex import AdjacencySimplex  # Import the class
# from calculate_tda_summaries import compute_persistence


class svi_tda_summaries:

    def __init__(self, scaled_svi_path, variables, state_abbr=None, threshold=None, filter_method='down'):
        
        self.scaled_svi_path = scaled_svi_path
        self.threshold = threshold
        self.filter_method = filter_method
        self.variables = variables
        self.state_abbr = state_abbr


        self.svi = gpd.read_file(self.scaled_svi_path)

    def compute_tda_summaries_svi(self):

        # Let's compute the adjacency information for the SVI data

        # get the unique county stcnty
        county_stcntys = self.svi['STCNTY'].unique()

        def process_county_variable(county_id, county_df, result_df, variable,filter_method):
            """
            Process a single county and variable combination, updating the result DataFrame.

            Parameters:
                county_id (str): The county identifier.
                county_df (GeoDataFrame): GeoDataFrame filtered to the current county.
                result_df (DataFrame): The DataFrame to update with results.
                variable (str): The variable to process.

            Returns:
                DataFrame: The updated result DataFrame.
            """
            # Select only the relevant columns
            temp_df = county_df[[variable, 'geometry']]

            average = temp_df[variable].mean()

            # Initialize the AdjacencySimplex object
            adj_simplex = AdjacencySimplex(
                gdf=temp_df,
                variable=variable,
                threshold=None,
                filter_method=filter_method
            )

            # Filter and sort the GeoDataFrame; ignore the second return value if not needed
            filtered_df, _ = adj_simplex.filter_sort_gdf()

            # Calculate adjacent countries and form the simplicial complex
            adj_simplex.calculate_adjacent_countries()
            simplex = adj_simplex.form_simplicial_complex()

            # Compute persistence values
            total_h0_points, tl, al, tml, aml, intervals_dim0 = compute_persistence(
                simplices=simplex,
                filtered_df=filtered_df,
                variable_name=variable
            )

            # Store the computed persistence values in the DataFrame
            result_df.loc[county_id, f'{variable}_TL'] = tl
            result_df.loc[county_id, f'{variable}_AL'] = al
            result_df.loc[county_id, f'{variable}_TML'] = tml
            result_df.loc[county_id, f'{variable}_AML'] = aml
            result_df.loc[county_id, f'{variable}_AV_ORI'] = average
            result_df.loc[county_id, 'filter_method'] = filter_method

            return result_df
        
        # Create a result DataFrame with county identifiers as its index
        result_df = pd.DataFrame(index=county_stcntys)
        result_df.index.name = 'STCNTY'

        # Loop through each county and process each variable of interest
        for county_id in tqdm(county_stcntys, desc='Processing Counties'):
            # Filter the main GeoDataFrame for the current county
            county_df = self.svi[self.svi['STCNTY'] == county_id]
            
            # Process each variable of interest for this county
            for variable in self.variables:
                result_df = process_county_variable(county_id, county_df, result_df, variable,filter_method=self.filter_method)

        result_df = result_df.reset_index()
        self.result_df = result_df

        print('TDA summaries computed successfully.')

        return result_df

        
        # print(result_df.head())


# if __name__ == '__main__':
#     scaled_svi_path = '/Users/h6x/ORNL/git/WORKSTAION GIT/nvss-experiments/data/SVI2018_US_tract.gdb'
#     variables = [
#     'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
#     'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
#     ]
#     state_abbr = 'TX'

#     svi_tda = svi_tda_summaries(scaled_svi_path, variables, state_abbr)
#     svi_tda.filter_clean_gdf()
#     svi_tda.compute_tda_summaries_svi()


