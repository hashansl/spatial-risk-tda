"""
smr.py

A library for calculating and plotting the Standardized Mortality Ratio (SMR)
for a specified state and year. The library retrieves population data from the
Census API, processes NVSS mortality data, calculates the SMR by county, and
creates a geographic plot of the results.

Usage example:
    from smr import StandardizedMortalityRatio

    smr_calc = StandardizedMortalityRatio(year=2018, state_code=47, state_abbr='TN', return_data=True)
    smr_calc.get_population_data()
    smr_calc.clean_population_data()
    smr_calc.process_nvss_data("/path/to/mort_2018.csv")
    smr_calc.calculate_smr()
    smr_calc.plot_map("/path/to/SVI2018_US_county.gdb", ".")
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from typing import Optional


class StandardizedMortalityRatio:
    """
    A class to calculate and plot the Standardized Mortality Ratio (SMR)
    for a given state and year.

    Attributes:
        year (int): Year for which data is being analyzed.
        state_code (int): FIPS code of the state.
        state_abbr (str): State abbreviation (e.g., 'TN').
        return_data (bool): Whether to return processed data from methods.
        population_data (pd.DataFrame): Cleaned population data.
        standard_population_mortality_by_age (pd.DataFrame): Mortality counts aggregated by age.
        standard_population_mortality_by_sex (pd.DataFrame): Mortality counts aggregated by sex.
        mortality_by_county (pd.DataFrame): Mortality counts aggregated by county.
        rr_df (pd.DataFrame): DataFrame containing the SMR (observed and expected deaths) per county.
    """

    def __init__(self, year: int, state_code: int, state_abbr: str, return_data: bool = False) -> None:
        self.year = year
        self.state_code = state_code
        self.state_abbr = state_abbr
        self.return_data = return_data

        # These will be populated by the respective methods.
        self.population_data: Optional[pd.DataFrame] = None
        self.standard_population_mortality_by_age: Optional[pd.DataFrame] = None
        self.standard_population_mortality_by_sex: Optional[pd.DataFrame] = None
        self.mortality_by_county: Optional[pd.DataFrame] = None
        self.rr_df: Optional[pd.DataFrame] = None

    def get_population_data(self) -> None:
        """
        Retrieve population data from the Census API for the specified year and state.

        The API returns county-level data for the state that includes total population,
        detailed age groups, and sex counts. The method converts the JSON response into a
        pandas DataFrame and ensures that numeric fields are properly typed.
        """
        base_url = f"https://api.census.gov/data/{self.year}/acs/acs5/subject"
        params = {
            "get": (
                "NAME,S0101_C01_001E,"  # Total population
                "S0101_C01_002E,S0101_C01_003E,S0101_C01_004E,S0101_C01_005E,"
                "S0101_C01_006E,S0101_C01_007E,S0101_C01_008E,S0101_C01_009E,"
                "S0101_C01_010E,S0101_C01_011E,S0101_C01_012E,S0101_C01_013E,"
                "S0101_C01_014E,S0101_C01_015E,S0101_C01_016E,S0101_C01_017E,"
                "S0101_C02_001E,S0101_C03_001E"
            ),
            "for": "county:*",
            "in": f"state:{self.state_code}",
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            # Convert the returned list of lists into a DataFrame. The first row contains headers.
            df = pd.DataFrame(data[1:], columns=data[0])

            # Rename columns to more readable names.
            df = df.rename(columns={
                "NAME": "County",
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
                "S0101_C01_017E": "75 years and over",
                "S0101_C02_001E": "Male Population",
                "S0101_C03_001E": "Female Population"
            })

            # Convert all columns (except identifiers) to numeric types.
            num_cols = [col for col in df.columns if col not in ['County', 'state', 'county']]
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            self.population_data = df
        else:
            raise ConnectionError(f"Error {response.status_code}: {response.text}")

    def clean_population_data(self, retun_cleaned_df = False) -> None:
        """
        Clean and restructure the population data for subsequent SMR calculations.

        Steps:
            1. Recalculate Male Population (if needed) as Total Population minus Female Population.
            2. Aggregate detailed age groups into broader 10-year groups.
            3. Format county codes by zero-filling and combining them with the state code.
            4. Remove columns that are not needed in later steps.
        """
        if self.population_data is None:
            raise ValueError("Population data is not loaded. Run get_population_data() first.")

        # Male poplulation data is corrupted so we need to drop it and recalculate it using the total and female population.
        self.population_data.drop(columns=["Male Population"], inplace=True)

        # Recalculate Male Population as Total Population minus
        self.population_data["Male Population"] = self.population_data["Total Population"] - self.population_data["Female Population"]

        # Aggregate age groups into 10-year bins.
        self.population_data["under_9"] = self.population_data["Under 5 years"] + self.population_data["5 to 9 years"]
        self.population_data["10_19"] = self.population_data["10 to 14 years"] + self.population_data["15 to 19 years"]
        self.population_data["20_29"] = self.population_data["20 to 24 years"] + self.population_data["25 to 29 years"]
        self.population_data["30_39"] = self.population_data["30 to 34 years"] + self.population_data["35 to 39 years"]
        self.population_data["40_49"] = self.population_data["40 to 44 years"] + self.population_data["45 to 49 years"]
        self.population_data["50_59"] = self.population_data["50 to 54 years"] + self.population_data["55 to 59 years"]
        self.population_data["60_69"] = self.population_data["60 to 64 years"] + self.population_data["65 to 69 years"]
        self.population_data["above_70"] = self.population_data["70 to 74 years"] + self.population_data["75 years and over"]

        # Process county FIPS codes:
        #   - Ensure county codes are three digits.
        #   - Prepend the state code.
        self.population_data['county'] = self.population_data['county'].astype(str).str.zfill(3)
        self.population_data['county'] = str(self.state_code) + self.population_data['county']
        self.population_data = self.population_data.rename(columns={"county": "FIPS"})

        # Drop columns that are not used in further calculations.
        cols_to_drop = [
            "Total Population", "state", "County",
            "Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 19 years",
            "20 to 24 years", "25 to 29 years", "30 to 34 years", "35 to 39 years",
            "40 to 44 years", "45 to 49 years", "50 to 54 years", "55 to 59 years",
            "60 to 64 years", "65 to 69 years", "70 to 74 years", "75 years and over"
        ]
        self.population_data = self.population_data.drop(columns=cols_to_drop, errors='ignore')

        if retun_cleaned_df:
            return self.population_data

    @staticmethod
    def _convert_to_years(detail_age: str) -> Optional[float]:
        """
        Helper function to convert NVSS detailed age codes into years.

        The first digit of the detail_age string indicates the unit:
            - 1: Years
            - 2: Months
            - 4: Days
            - 5: Hours
            - 6: Minutes
            - 9: Age not stated

        The remaining digits represent the age value.
        Returns:
            Age in years as a float, or np.nan if conversion fails.
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

    def process_nvss_data(self, path: str) -> None:
        """
        Process NVSS mortality data from a CSV file to compute mortality counts.

        The method performs the following steps:
            1. Reads the CSV file.
            2. Filters the data for the specified state (by abbreviation).
            3. Filters records based on a set of ICD-10 codes for diseases of interest.
            4. Formats county codes to match the population data (zero-filling and prepending state code).
            5. Converts the detailed age field to years.
            6. Bins the ages into predefined groups.
            7. Aggregates mortality counts by age category, sex, and county.

        Args:
            path (str): The file path to the NVSS mortality CSV.
        """
        mortality_df = pd.read_csv(path)

        # 1. Filter by state abbreviation.
        mortality_df = mortality_df[mortality_df["occurrence_state"] == self.state_abbr]

        # 2. Filter by ICD-10 codes of interest.
        icd_codes = [
            'X40', 'X41', 'X42', 'X43', 'X44',
            'X60', 'X61', 'X62', 'X63', 'X64',
            'X85', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14'
        ]
        mortality_df = mortality_df[mortality_df['cause of death_icd10'].str.contains(
            '|'.join(icd_codes), na=False)]

        # 3. Process county codes.
        mortality_df['occurrence_county'] = mortality_df['occurrence_county'].astype(str).str.zfill(3)
        mortality_df['occurrence_county'] = str(self.state_code) + mortality_df['occurrence_county']

        # 4. Convert detailed age into years.
        mortality_df['age_years'] = mortality_df['detail_age'].apply(self._convert_to_years)

        # 5. Categorize ages into bins.
        bins = [0, 9, 19, 29, 39, 49, 59, 69, float('inf')]
        labels = ['<9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        mortality_df['age_category'] = pd.cut(mortality_df['age_years'], bins=bins, labels=labels, right=True)

        # 6. Aggregate mortality counts by age.
        self.standard_population_mortality_by_age = (
            mortality_df.groupby('age_category').size().reset_index(name='deaths')
        )

        # 7. Aggregate mortality counts by sex.
        self.standard_population_mortality_by_sex = (
            mortality_df.groupby('sex').size().reset_index(name='deaths')
        )

        # 8. Aggregate mortality counts by county.
        self.mortality_by_county = (
            mortality_df.groupby('occurrence_county').size().reset_index(name='deaths')
        )

    def calculate_smr(self) -> Optional[pd.DataFrame]:
        """
        Calculate the Standardized Mortality Ratio (SMR) by county.

        The SMR is calculated as the ratio of observed deaths to expected deaths.
        Expected deaths are computed by applying the standard (population-wide)
        mortality rates—derived separately for sex and age groups—to the local
        population counts.

        Steps:
            1. Merge county-level mortality counts with the population data.
            2. Compute sex-specific mortality rates (r_j) using the standard population.
            3. Compute age-specific mortality rates (r_i) for each age group.
            4. Calculate expected deaths per county.
            5. Compute the SMR as observed deaths divided by expected deaths.

        Returns:
            pd.DataFrame: A DataFrame with FIPS, observed deaths, expected deaths, and SMR.
        """
        if self.population_data is None:
            raise ValueError("Population data is not available. Run get_population_data() and clean_population_data() first.")
        if self.mortality_by_county is None:
            raise ValueError("Mortality county data is missing. Run process_nvss_data() first.")
        if self.standard_population_mortality_by_sex is None or self.standard_population_mortality_by_age is None:
            raise ValueError("Mortality data by age/sex is missing. Run process_nvss_data() first.")

        # Copy population data needed for the calculation.
        rr_df = self.population_data[[
            'FIPS', 'Male Population', 'Female Population',
            'under_9', '10_19', '20_29', '30_39', '40_49',
            '50_59', '60_69', 'above_70'
        ]].copy()

        # Merge county-level observed mortality counts.
        rr_df = rr_df.merge(self.mortality_by_county,
                            left_on='FIPS', right_on='occurrence_county', how='left')
        rr_df['deaths'] = rr_df['deaths'].fillna(0)
        rr_df = rr_df.rename(columns={'deaths': 'observed_deaths'}).drop(columns=['occurrence_county'])

        # Compute the overall (standard) populations for each sex.
        total_male_pop = rr_df['Male Population'].sum()
        total_female_pop = rr_df['Female Population'].sum()

        # Calculate sex-specific mortality rates (r_j).
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

        rr_df['r_j_Male'] = r_j_Male
        rr_df['r_j_Female'] = r_j_Female

        # Define the mapping of local population column names to age group labels.
        age_groups = {
            'under_9': '<9',
            '10_19': '10-19',
            '20_29': '20-29',
            '30_39': '30-39',
            '40_49': '40-49',
            '50_59': '50-59',
            '60_69': '60-69',
            'above_70': '70+'
        }

        # Calculate age-specific mortality rates (r_i) for each age group.
        for pop_col, age_label in age_groups.items():
            total_pop = rr_df[pop_col].sum()
            death_count = self.standard_population_mortality_by_age.loc[
                self.standard_population_mortality_by_age["age_category"] == age_label, 'deaths'
            ]
            if not death_count.empty and total_pop > 0:
                rate = death_count.values[0] / total_pop
            else:
                rate = 0.0
            rr_df[f'r_i_{pop_col}'] = rate

        # Compute expected deaths by applying sex- and age-specific rates to local populations.
        rr_df['expected_deaths'] = (
            rr_df['r_j_Male'] * rr_df['Male Population'] +
            rr_df['r_j_Female'] * rr_df['Female Population'] +
            rr_df['r_i_under_9'] * rr_df['under_9'] +
            rr_df['r_i_10_19'] * rr_df['10_19'] +
            rr_df['r_i_20_29'] * rr_df['20_29'] +
            rr_df['r_i_30_39'] * rr_df['30_39'] +
            rr_df['r_i_40_49'] * rr_df['40_49'] +
            rr_df['r_i_50_59'] * rr_df['50_59'] +
            rr_df['r_i_60_69'] * rr_df['60_69'] +
            rr_df['r_i_above_70'] * rr_df['above_70']
        )

        # Compute the SMR (observed / expected). Use np.nan for counties with zero expected deaths.
        rr_df['SMR'] = rr_df.apply(
            lambda row: row['observed_deaths'] / row['expected_deaths']
            if row['expected_deaths'] != 0 else np.nan, axis=1
        )

        # Retain only the essential columns.
        self.rr_df = rr_df[['FIPS', 'observed_deaths', 'expected_deaths', 'SMR']]

        if self.return_data:
            return self.rr_df

    def plot_map(self, geo_data_path: str, save_dir: Optional[str] = None) -> None:
        """
        Plot the Standardized Mortality Ratio (SMR) on a geographic map.

        Steps:
            1. Load geographic data from a GeoDatabase or shapefile.
            2. Filter the geographic data for the desired state.
            3. Merge the SMR data (by county FIPS) with the geographic geometries.
            4. Plot counties using a colormap for nonzero SMR values and highlight counties with zero SMR.
            5. Display the plot if save_dir is None; otherwise, save the plot.

        Args:
            geo_data_path (str): Path to the geographic data file.
            save_dir (Optional[str]): Directory where the plot image will be saved. If None, the plot is displayed.
        """
        if self.rr_df is None:
            raise ValueError("SMR data not available. Run calculate_smr() first.")

        # Load and filter the geographic data.
        geo_df = gpd.read_file(geo_data_path, dtype={'FIPS': str})
        geo_df = geo_df[geo_df['ST_ABBR'] == self.state_abbr][['FIPS', 'geometry']]

        # Merge the SMR results with geographic boundaries.
        merged_df = self.rr_df.merge(geo_df, on='FIPS', how='left')
        merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')

        # Define plotting parameters.
        zero_color = "white"  # For counties with SMR == 0.
        cmap = "inferno"      # Colormap for SMR values.

        fig, ax = plt.subplots(figsize=(10, 10))
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

        plt.title(f"Standardized Mortality Ratio (SMR) for {self.state_abbr}")
        plt.axis("off")

        if save_dir:
            plt.savefig(f"{save_dir}/smr_map_{self.state_abbr}.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# # When running as a standalone script, an example usage is provided below.
# if __name__ == '__main__':
#     # Example usage:
#     # Ensure that the paths below point to valid data files on your system.
#     smr_calc = StandardizedMortalityRatio(year=2018, state_code=47, state_abbr='TN', return_data=True)
#     smr_calc.get_population_data()
#     smr_calc.clean_population_data()
#     smr_calc.process_nvss_data("/Users/h6x/ORNL/git/WORKSTAION GIT/nvss-experiments/experiment_1/data/mort_2018.csv")
#     smr_calc.calculate_smr()
#     smr_calc.plot_map("/Users/h6x/ORNL/git/WORKSTAION GIT/nvss-experiments/experiment_1/data/SVI2018_US_county.gdb", ".")
