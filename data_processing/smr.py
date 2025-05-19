import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from typing import Optional


class StandardizedMortalityRatio:
    """
    A class to calculate and plot the Standardized Mortality Ratio (SMR)
    for the entire USA for a given year.
    """

    def __init__(self, year: int,state_abbr: str) -> None:
        self.year = year
        self.state_abbr = state_abbr

        self.state_codes = {
                    'AK': '02', 'AL': '01', 'AR': '05', 'AZ': '04', 'CA': '06',
                    'CO': '08', 'CT': '09', 'DC': '11', 'DE': '10', 'FL': '12',
                    'GA': '13', 'HI': '15', 'IA': '19', 'ID': '16', 'IL': '17',
                    'IN': '18', 'KS': '20', 'KY': '21', 'LA': '22', 'MA': '25',
                    'MD': '24', 'ME': '23', 'MI': '26', 'MN': '27', 'MO': '29',
                    'MS': '28', 'MT': '30', 'NC': '37', 'ND': '38', 'NE': '31',
                    'NH': '33', 'NJ': '34', 'NM': '35', 'NV': '32', 'NY': '36',
                    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
                    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
                    'VA': '51', 'VT': '50', 'WA': '53', 'WI': '55', 'WV': '54',
                    'WY': '56'
                }

        # These will be populated by the respective methods.
        self.population_data: Optional[pd.DataFrame] = None
        self.standard_population_mortality_by_age: Optional[pd.DataFrame] = None
        self.standard_population_mortality_by_sex: Optional[pd.DataFrame] = None
        self.mortality_by_county: Optional[pd.DataFrame] = None

        
        
        

    def get_population_data(self, return_pop_data: bool = False) -> Optional[pd.DataFrame]:
        """
        Retrieve county-level population data for all states from the Census API.
        # https://api.census.gov/data/2019/acs/acs5/subject/variables.html
        """
        base_url = f"https://api.census.gov/data/{self.year}/acs/acs5/subject"
        params = {
            "get": (
                "NAME,S0101_C01_001E,"  # Total population
                "S0101_C01_002E,S0101_C01_003E,S0101_C01_004E,S0101_C01_005E,"
                "S0101_C01_006E,S0101_C01_007E,S0101_C01_008E,S0101_C01_009E,"
                "S0101_C01_010E,S0101_C01_011E,S0101_C01_012E,S0101_C01_013E,"
                "S0101_C01_014E,S0101_C01_015E,S0101_C01_016E,S0101_C01_017E,"
                "S0101_C01_018E,S0101_C01_019E,S0101_C03_001E,S0101_C05_001E"
            ),
            "for": "county:*",
            "in": f"state:{self.state_codes.get(self.state_abbr)}",  # Get data for selected state
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Convert the returned list of lists into a DataFrame.
            df = pd.DataFrame(data[1:], columns=data[0])

            # Rename columns to more readable names.
            df = df.rename(columns={
                "NAME": "County_Name",
                "S0101_C01_001E": "Total Population",
                "S0101_C01_002E": "Under 5 years",
                "S0101_C01_003E": "5 to 9 years",
                "S0101_C01_004E": "10 to 14 years",
                "S0101_C01_005E": "15 to 19 years",
                "S0101_C01_006E": "20 to 24 years",
                "S0101_C01_007E": "25 to 29 years",
                "S0101_C01_008E": "30 to 34 years",
                "S0101_C01_009E": "35 to 39 years",
                "S0101_C01_010E": "40 to 44 years",
                "S0101_C01_011E": "45 to 49 years",
                "S0101_C01_012E": "50 to 54 years",
                "S0101_C01_013E": "55 to 59 years",
                "S0101_C01_014E": "60 to 64 years",
                "S0101_C01_015E": "65 to 69 years",
                "S0101_C01_016E": "70 to 74 years",
                "S0101_C01_017E": "75 to 79 years",
                "S0101_C01_018E": "80 to 84 years",
                "S0101_C01_019E": "85 years and over",
                "S0101_C03_001E": "Male Population",
                "S0101_C05_001E": "Female Population"
            })

            # Convert numeric columns.
            num_cols = [col for col in df.columns if col not in ['County_Name', 'state', 'county']]
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            self.population_data = df

            if return_pop_data:
                return self.population_data # Return the data if requested.
        else:
            raise ConnectionError(f"Error {response.status_code}: {response.text}")

    def clean_population_data(self, retun_clean_data: bool = False) -> Optional[pd.DataFrame]:
        """
        Clean and restructure the population data for nationwide SMR calculations.
        """
        if self.population_data is None:
            raise ValueError("Population data is not loaded. Run get_population_data() first.")
        
        if self.population_data is None:
            raise ValueError("Population data is not loaded. Run get_population_data() first.")

        # Aggregate age groups into broader bins.
        self.population_data["under_9"] = self.population_data["Under 5 years"] + self.population_data["5 to 9 years"]
        self.population_data["10_19"] = self.population_data["10 to 14 years"] + self.population_data["15 to 19 years"]
        self.population_data["20_29"] = self.population_data["20 to 24 years"] + self.population_data["25 to 29 years"]
        self.population_data["30_39"] = self.population_data["30 to 34 years"] + self.population_data["35 to 39 years"]
        self.population_data["40_49"] = self.population_data["40 to 44 years"] + self.population_data["45 to 49 years"]
        self.population_data["50_59"] = self.population_data["50 to 54 years"] + self.population_data["55 to 59 years"]
        self.population_data["60_69"] = self.population_data["60 to 64 years"] + self.population_data["65 to 69 years"]
        self.population_data["above_70"] = self.population_data["70 to 74 years"] + self.population_data["75 to 79 years"]+ self.population_data["80 to 84 years"] + self.population_data["85 years and over"]

        # Process FIPS codes by combining the state and county codes.
        self.population_data['state'] = self.population_data['state'].astype(str).str.zfill(2)
        self.population_data['county'] = self.population_data['county'].astype(str).str.zfill(3)
        self.population_data['FIPS'] = self.population_data['state'] + self.population_data['county']

        # Drop columns that won't be needed.
        cols_to_drop = [
            "Total Population", "County_Name",
            "Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years",
            "20 to 24 years", "25 to 29 years", "30 to 34 years", "35 to 39 years",
            "40 to 44 years", "45 to 49 years", "50 to 54 years", "55 to 59 years",
            "60 to 64 years", "65 to 69 years", "70 to 74 years", "75 to 79 years",
            "80 to 84 years", "85 years and over"
        ]
        self.population_data = self.population_data.drop(columns=cols_to_drop, errors='ignore')

        if retun_clean_data:
            return self.population_data

    
    @staticmethod
    def convert_to_years(detail_age: str) -> Optional[float]:
        """
        Convert NVSS detailed age codes into years.
        """
        try:
            age_str = str(detail_age)
            if len(age_str) < 2:
                return np.nan
            age_unit = int(age_str[0])
            age_value = int(age_str[1:])
            if age_unit == 1:  # Years
                return age_value
            elif age_unit == 2:  # Months
                return age_value / 12.0
            elif age_unit == 4:  # Days
                return age_value / 365.0
            elif age_unit == 5:  # Hours
                return age_value / (365.0 * 24)
            elif age_unit == 6:  # Minutes
                return age_value / (365.0 * 24 * 60)
            elif age_unit == 9:  # Age not stated
                return np.nan
            else:
                return np.nan
        except Exception:
            return np.nan


    def process_nvss_data(self, path: str, return_processed_data: bool = False) -> Optional[pd.DataFrame]:
        """
        Process NVSS mortality data from a CSV file to compute mortality counts for selected state.
        """

        mortality_df = pd.read_csv(path)

        # filter by state
        mortality_df = mortality_df[mortality_df['occurrence_state']== self.state_abbr]

        icd_codes = [
            'X40', 'X41', 'X42', 'X43', 'X44',
            'X60', 'X61', 'X62', 'X63', 'X64',
            'X85', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14'
        ]
        mortality_df = mortality_df[mortality_df['cause of death_icd10'].str.contains(
            '|'.join(icd_codes), na=False)]
        
        state_codes = {
                    'AK': '02', 'AL': '01', 'AR': '05', 'AZ': '04', 'CA': '06',
                    'CO': '08', 'CT': '09', 'DC': '11', 'DE': '10', 'FL': '12',
                    'GA': '13', 'HI': '15', 'IA': '19', 'ID': '16', 'IL': '17',
                    'IN': '18', 'KS': '20', 'KY': '21', 'LA': '22', 'MA': '25',
                    'MD': '24', 'ME': '23', 'MI': '26', 'MN': '27', 'MO': '29',
                    'MS': '28', 'MT': '30', 'NC': '37', 'ND': '38', 'NE': '31',
                    'NH': '33', 'NJ': '34', 'NM': '35', 'NV': '32', 'NY': '36',
                    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
                    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
                    'VA': '51', 'VT': '50', 'WA': '53', 'WI': '55', 'WV': '54',
                    'WY': '56'
                }
        # Map state codes to FIPS codes.
        mortality_df['state'] = mortality_df['occurrence_state'].map(state_codes)

        # Process FIPS codes by combining the state and county codes.
        mortality_df['county'] = mortality_df['occurrence_county'].astype(str).str.zfill(3)
        mortality_df['FIPS'] = mortality_df['state'] + mortality_df['county']

        # Convert detailed age into years.
        mortality_df['age_years'] = mortality_df['detail_age'].apply(self.convert_to_years)

        # Categorize ages into bins.
        bins = [0, 9, 19, 29, 39, 49, 59, 69, float('inf')]
        labels = ['under_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', 'above_70']
        mortality_df['age_category'] = pd.cut(mortality_df['age_years'], bins=bins, labels=labels, right=True)

        # Aggregate mortality counts by age.
        standard_population_mortality_by_age = (
            mortality_df.groupby('age_category').size().reset_index(name='deaths')
        )

        # Aggregate mortality counts by sex.
        standard_population_mortality_by_sex = (
            mortality_df.groupby('sex').size().reset_index(name='deaths')
        )

        # Aggregate mortality counts by county.
        mortality_by_county = (
            mortality_df.groupby('FIPS').size().reset_index(name='deaths')
        )

        self.standard_population_mortality_by_age = standard_population_mortality_by_age
        self.standard_population_mortality_by_sex = standard_population_mortality_by_sex
        self.mortality_by_county = mortality_by_county = mortality_by_county

        if return_processed_data:
            return standard_population_mortality_by_age, standard_population_mortality_by_sex,mortality_by_county

    def calculate_smr(self) -> Optional[pd.DataFrame]:
        """
        Calculate the SMR by county for the selected state.
        """
        if self.population_data is None:
            raise ValueError("Population data is not available. Run get_population_data() and clean_population_data() first.")
        if self.mortality_by_county is None:
            raise ValueError("Mortality county data is missing. Run process_nvss_data() first.")
        if self.standard_population_mortality_by_sex is None or self.standard_population_mortality_by_age is None:
            raise ValueError("Mortality data by age/sex is missing. Run process_nvss_data() first.")
        
        # Filter the columns needed for SMR calculation.
        self.population_data = self.population_data[[
            'FIPS', 'Male Population', 'Female Population',
            'under_9', '10_19', '20_29', '30_39', '40_49',
            '50_59', '60_69', 'above_70'
        ]]

        # Merge county-level observed mortality counts.
        self.population_data = self.population_data.merge(self.mortality_by_county,
                            on ='FIPS', how='left')
        self.population_data['deaths'] = self.population_data['deaths'].fillna(0)
        self.population_data = self.population_data.rename(columns={'deaths': 'observed_deaths'})

        # Compute the overall (standard) populations for each sex.
        total_male_pop = self.population_data['Male Population'].sum()
        total_female_pop = self.population_data['Female Population'].sum()

        # Calculate sex-specific mortality rates.
        try:
            r_j_Male = self.standard_population_mortality_by_sex.loc[
                self.standard_population_mortality_by_sex["sex"] == "M", 'deaths'
            ].values[0] / total_male_pop
        except IndexError:
            r_j_Male = 0.0

        try:
            r_j_Female = self.standard_population_mortality_by_sex.loc[
                self.standard_population_mortality_by_sex["sex"] == "F", 'deaths'
            ].values[0] / total_female_pop
        except IndexError:
            r_j_Female = 0.0

        self.population_data['r_j_Male'] = r_j_Male
        self.population_data['r_j_Female'] = r_j_Female

        # List of age group columns (common to both DataFrames)
        age_columns = ['under_9', '10_19', '20_29', '30_39', '40_49', '50_59', '60_69', 'above_70']

        for age in age_columns:
            total_pop = self.population_data[age].sum()
            death_count = self.standard_population_mortality_by_age.loc[
                self.standard_population_mortality_by_age["age_category"] == age, 'deaths'
            ]
            if not death_count.empty and total_pop > 0:
                rate = death_count.values[0] / total_pop
            else:
                rate = 0.0
            self.population_data[f'r_i_{age}'] = rate

        # Compute expected deaths.
        self.population_data['expected_deaths'] = (
            self.population_data['r_j_Male'] * self.population_data['Male Population'] +
            self.population_data['r_j_Female'] * self.population_data['Female Population'] +
            self.population_data['r_i_under_9'] * self.population_data['under_9'] +
            self.population_data['r_i_10_19'] * self.population_data['10_19'] +
            self.population_data['r_i_20_29'] * self.population_data['20_29'] +
            self.population_data['r_i_30_39'] * self.population_data['30_39'] +
            self.population_data['r_i_40_49'] * self.population_data['40_49'] +
            self.population_data['r_i_50_59'] * self.population_data['50_59'] +
            self.population_data['r_i_60_69'] * self.population_data['60_69'] +
            self.population_data['r_i_above_70'] * self.population_data['above_70']
        )

        # Compute the SMR.
        self.population_data['SMR'] = self.population_data.apply(
            lambda row: row['observed_deaths'] / row['expected_deaths']
            if row['expected_deaths'] != 0 else np.nan, axis=1
        )

        self.population_data = self.population_data[['FIPS', 'observed_deaths', 'expected_deaths', 'SMR']]

        # save the SMR data to a CSV file.
        # self.population_data.to_csv(f"smr_data_{self.year}_{self.state_abbr}.csv", index=False)

        return self.population_data


    def plot_map(self, geo_data_path: str, save_dir: Optional[str] = None) -> None:
        """
        Plot the nationwide SMR on a geographic map.
        """
        if self.population_data is None:
            raise ValueError("SMR data not available. Run calculate_smr() first.")

        # Load geographic data (assuming it covers the entire USA).
        geo_df = gpd.read_file(geo_data_path, dtype={'FIPS': str})
        # Ensure FIPS codes are 5 digits.
        geo_df['FIPS'] = geo_df['FIPS'].astype(str).str.zfill(5)

        # Merge SMR data with geographic boundaries.
        merged_df = self.population_data.merge(geo_df[['FIPS', 'geometry']], on='FIPS', how='left')
        merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')

        # drop rows with FIPS starts with 72, 78, 02, 15, 60
        merged_df = merged_df[~merged_df['FIPS'].str.startswith(('72', '78', '02', '15', '60'))]

        # Define plotting parameters.
        zero_color = "white"  # For counties with SMR == 0.
        cmap = "inferno"      # Colormap for SMR values.

        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot counties with nonzero SMR.
        merged_df[merged_df["SMR"] != 0].plot(
            column="SMR",
            cmap=cmap,
            ax=ax,
            legend=True,
            legend_kwds={
                'orientation': "horizontal",
                'shrink': 0.5,
                'pad': 0.02,
                'aspect': 30
            }
        )
        # Overlay counties with SMR == 0.
        merged_df[merged_df["SMR"] == 0].plot(
            color=zero_color, ax=ax, edgecolor="black", alpha=0.5
        )

        plt.title("Standardized Mortality Ratio (SMR) for the USA")
        plt.axis("off")

        if save_dir:
            plt.savefig(f"{save_dir}/smr_map_USA.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# Example usage:
# smr_calc = NationalStandardizedMortalityRatio(year=2018, return_data=True)
# smr_calc.get_population_data()
# smr_calc.clean_population_data()
# smr_calc.process_nvss_data("/path/to/mort_2018.csv")
# smr_calc.calculate_smr()
# smr_calc.plot_map("/path/to/SVI2018_US_county.gdb", ".")
