from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
from datetime import timedelta
import pandas as pd
import numpy as np
import logging
import os

# Setting up the logger for logging information and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataProcessor:
    def __init__(self, data_path, key_path, project_id):
        """
        Initializes the F1DataProcessor class with the specified data path, key path, and project ID.

        :param data_path: Path where the data files are stored.
        :param key_path: Path to the service account key file.
        :param project_id: ID of the Google Cloud Project.
        """
        self.data_path = data_path
        self.credentials = service_account.Credentials.from_service_account_file(
            key_path, 
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.client = bigquery.Client(credentials=self.credentials, project=project_id)

        # Initializing DataFrames as None
        self.constructors_df = self.results_df = self.drivers_df = None
        self.races_df = self.teams_df = self.race_to_events_df = None
        self.race_events_to_driver_df = self.season_to_race_df = None
        self.season_only_data_df = self.race_to_venue_df = None
        self.Added_constructors_df = self.Added_results_df = None
        self.Added_drivers_df = self.Added_races_df = None

    def table_exists(self, dataset_id, table_id):
        """
        Checks if a table exists in the specified dataset.

        :param dataset_id: The ID of the dataset.
        :param table_id: The ID of the table to check.
        :return: True if the table exists, False otherwise.
        """
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        try:
            self.client.get_table(table_ref)
            return True
        except NotFound:
            return False

    def create_table(self, dataset_id, table_id, schema):
        """
        Creates a new table in the specified dataset with the given schema.

        :param dataset_id: The ID of the dataset in which to create the table.
        :param table_id: The ID of the new table.
        :param schema: The schema of the new table.
        """
        dataset_ref = self.client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        self.client.create_table(table)
        logger.info(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    def generate_schema_from_df(self, dataframe):
        """
        Generates a BigQuery schema from a pandas DataFrame.

        :param dataframe: The pandas DataFrame from which to generate the schema.
        :return: A list of BigQuery SchemaField objects.
        """
        schema = []
        for column_name, dtype in dataframe.dtypes.iteritems():
            field_type = 'STRING'
            if pd.api.types.is_integer_dtype(dtype):
                field_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                field_type = 'FLOAT'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                field_type = 'TIMESTAMP'
            schema.append(bigquery.SchemaField(column_name, field_type))
        return schema

    def insert_data(self, dataset_id, table_id, dataframe):
        """
        Inserts data from a pandas DataFrame into a BigQuery table. If the table doesn't exist, it's created.

        :param dataset_id: The ID of the dataset.
        :param table_id: The ID of the table to insert data into.
        :param dataframe: The pandas DataFrame containing the data to insert.
        """
        # Generate schema from the DataFrame
        schema = self.generate_schema_from_df(dataframe)

        # Check if the table exists, create if not
        if not self.table_exists(dataset_id, table_id):
            self.create_table(dataset_id, table_id, schema)

        # Prepare the table reference
        table_ref = self.client.dataset(dataset_id).table(table_id)

        # Create a LoadJobConfig to control the load job
        job_config = bigquery.LoadJobConfig(
            schema=schema, 
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        )

        # Insert the data
        job = self.client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
        job.result()  # Wait for the load job to complete

        if job.errors:
            logger.error(f"Encountered errors while inserting rows: {job.errors}")
        else:
            logger.info(f"Data inserted successfully into {table_id}.")

    def load_data(self, file_name, file_type='excel'):
        """
        Loads data from a file into a DataFrame.

        :param file_name: The name of the file to be loaded.
        :param file_type: The type of the file ('csv' or 'excel').
        :return: A pandas DataFrame containing the loaded data.
        """
        file_path = os.path.join(self.data_path, file_name)
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type == 'excel':
                return pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Error loading data from {file_name}: {e}")
            raise

    def add_new_teams(self, race_to_team_positions_df):
        """
        Adds new teams to the teams DataFrame based on provided race to team positions DataFrame.

        :param race_to_team_positions_df: A DataFrame containing race to team positions data.
        :return: The updated teams DataFrame with new teams added.
        """
        unique_teams_df = race_to_team_positions_df[
            ['teamId', 'team_name', 'team_gender', 'team_nationality', 'team_country_code']
        ].drop_duplicates(subset=['teamId']).reset_index(drop=True)

        max_id_num = unique_teams_df['teamId'].str.extract('(\d+)').astype(int).max()[0]

        # Defining the new teams
        new_teams = [
            {'teamId': f"sr:competitor:{max_id_num + i + 1}", 'team_name': team_name, 'team_gender': 'male', 'team_nationality': nationality, 'team_country_code': country_code}
            for i, (team_name, nationality, country_code) in enumerate([
                ('Racing Point', 'Great Britain', 'GBR'),
                ('Renault', 'France', 'FRA'),
                ('Toro Rosso', 'Italy', 'ITA'),
                ('Sauber', 'Switzerland', 'CHE'),
                ('BMW Sauber', 'Switzerland', 'CHE'),
                ('Lotus', 'Great Britain', 'GBR'),
                ('Lotus F1', 'Great Britain', 'GBR'),  
                ('Marussia', 'Great Britain', 'GBR'),
                ('Spyker', 'Great Britain', 'GBR'),
                ('Virgin', 'Great Britain', 'GBR')
            ])
        ]

        # Adding the new teams to the DataFrame
        return unique_teams_df.append(new_teams, ignore_index=True)
    
    def parse_time(self, time_str):
        """
        Parses a time string into a timedelta object. Handles different time formats.

        :param time_str: A string representing time, in various possible formats.
        :return: A timedelta object representing the parsed time, or None if parsing fails.
        """
        if pd.isna(time_str):
            return None
        try:
            if '+' in time_str:  # Time format adjustment if it contains '+'
                time_str = time_str.replace('+', '')
                parts = time_str.split(':')
                if len(parts) == 2:  # Format: MM:SS.FFF
                    return timedelta(minutes=int(parts[0]), seconds=float(parts[1]))
                else:  # Format: SS.FFF
                    return timedelta(seconds=float(time_str))
            else:
                # Format: HH:MM:SS.FFF
                h, m, s = time_str.split(':')
                return timedelta(hours=int(h), minutes=int(m), seconds=float(s))
        except Exception as e:
            return None

    def calculate_actual_time_safe(self, df, first_position_times):
        """
        Calculates the actual race time for each driver, considering the first position time.

        :param df: A DataFrame containing race data.
        :param first_position_times: A dictionary with race IDs and the total race time of the first position.
        :return: A list of actual race times.
        """
        actual_times = []
        for _, row in df.iterrows():
            if pd.isna(row['TotalRaceTimeDelta']) or row['RaceId'] not in first_position_times:
                actual_times.append(None)
            elif row['FinalPosition'] == 1:
                actual_times.append(row['TotalRaceTimeDelta'])
            else:
                actual_times.append(first_position_times[row['RaceId']] + row['TotalRaceTimeDelta'])
        return actual_times

    def format_time(self, delta):
        """
        Formats a timedelta object into a string representation of time.

        :param delta: A timedelta object representing time duration.
        :return: A string formatted as 'HH:MM:SS.mmm' or 'MM:SS.mmm' depending on the duration.
        """
        if pd.isna(delta):
            return None
        # Extracting hours, minutes, seconds, and milliseconds
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        # Formatting the time string
        if hours > 0:
            return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"
        else:
            return f"{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"

    def build_fact_table(self, Added_constructors_df, Added_results_df, Added_drivers_df, Added_races_df, 
                         teams_df, race_to_events_df, race_events_to_driver_df, season_to_race_df, 
                         race_to_venue_df, data_types):
        """
        Builds the fact table by merging and transforming various data sources.
        
        :param Added_constructors_df: DataFrame containing constructor data.
        :param Added_results_df: DataFrame containing race results.
        :param Added_drivers_df: DataFrame containing driver data.
        :param Added_races_df: DataFrame containing race data.
        :param teams_df: DataFrame containing team data.
        :param race_to_events_df: DataFrame mapping races to events.
        :param race_events_to_driver_df: DataFrame mapping race events to drivers.
        :param season_to_race_df: DataFrame mapping seasons to races.
        :param race_to_venue_df: DataFrame mapping races to venues.
        :param data_types: Dictionary specifying data types for columns.
        :return: A DataFrame representing the constructed fact table.
        """
        # Merging results with constructors and drivers data
        Added_merged_df = Added_results_df.merge(Added_constructors_df, on='constructorId', how='left')
        Added_merged_df = Added_merged_df.merge(Added_drivers_df, on='driverId', how='left')
        Added_merged_df = pd.merge(Added_merged_df, teams_df, how='left', left_on='name', right_on='team_name')
        Added_merged_df['driver_name'] = Added_merged_df['surname'] + ', ' + Added_merged_df['forename']

        # Merging race events to driver data with season to race data
        joined_df = race_events_to_driver_df.merge(race_to_events_df[['eventId', 'raceId']], on='eventId', how='left')
        joined_df = joined_df.merge(season_to_race_df, on='raceId', how='left')

        # Data preprocessing and datetime conversions
        Added_races_df['date'] = pd.to_datetime(Added_races_df['date'])
        joined_df['scheduled'] = pd.to_datetime(joined_df['scheduled']).dt.tz_localize(None)
        joined_df['scheduled_end'] = pd.to_datetime(joined_df['scheduled_end']).dt.tz_localize(None)
        joined_df['scheduled_expanded_start'] = joined_df['scheduled'] - timedelta(days=1)
        joined_df['scheduled_expanded_end'] = joined_df['scheduled_end'] + timedelta(days=1)

        joined_df['foreign_raceId'] = None

        # Matching race IDs in joined data with races data
        for joined_index, joined_row in joined_df.iterrows():
            scheduled_start = joined_row['scheduled_expanded_start']
            scheduled_end = joined_row['scheduled_expanded_end']
            matching_races = Added_races_df[(Added_races_df['date'] >= scheduled_start) & (Added_races_df['date'] <= scheduled_end)]
            if not matching_races.empty:
                joined_df.at[joined_index, 'foreign_raceId'] = matching_races.iloc[0]['raceId']

        # Final merging to create the fact table
        final_df = pd.merge(joined_df, Added_merged_df, left_on=['foreign_raceId', 'driver_name'], right_on=['raceId', 'driver_name'], how='left')
        final_df = pd.merge(final_df, race_to_venue_df, how='left', left_on='raceId_x', right_on='raceId')

        # Renaming and reorganizing columns
        fact_new_column_names = {
            'raceId_x': 'RaceId', 'driverId_x': 'DriverId', 'teamId_y': 'TeamId', 
            'venueId': 'VenueId', 'seasonId': 'SeasonId', 'fastestLap': 'FastestLap',
            'fastest_lap_time': 'FastestLapTime', 'fastestLapSpeed': 'FastestLapSpeed', 
            'laps_x': 'TotalLaps', 'position_x': 'FinalPosition', 'grid_x': 'GridPosition', 
            'points': 'IndividualPoints', 'pitstop_count': 'PitstopCount', 'time_y': 'TotalRaceTime', 
            'status': 'Status'
        }  # Mapping of old to new column names
        final_df.rename(columns=fact_new_column_names, inplace=True)

        # Generating a unique identifier for race performance
        final_df.insert(0, 'RacePerformanceId', final_df['RaceId'].astype(str) + final_df['DriverId'].astype(str) + final_df['TeamId'].astype(str))

        # Calculating race time deltas and formatting
        final_df['TotalRaceTimeDelta'] = final_df['TotalRaceTime'].apply(self.parse_time)
        first_position_times = final_df[final_df['FinalPosition'] == 1].groupby('RaceId')['TotalRaceTimeDelta'].first()
        final_df['ActualTotalRaceTime'] = self.calculate_actual_time_safe(final_df, first_position_times)
        final_df['ActualTotalRaceTime'] = final_df['ActualTotalRaceTime'].apply(self.format_time)
        
        fact_table_selected_columns = [
        'RacePerformanceId', 'RaceId', 'DriverId', 'TeamId', 'VenueId', 'SeasonId', 
        'FastestLap', 'FastestLapTime', 'FastestLapSpeed', 'TotalLaps', 
        'FinalPosition', 'GridPosition', 'IndividualPoints', 'PitstopCount', 'ActualTotalRaceTime', 'Status'
        ]

        fact_table = final_df[fact_table_selected_columns]
        fact_table = fact_table.replace('\\N', np.nan)
        fact_table = fact_table.fillna(value=np.nan)
        
        if data_types:
            for column, dtype in data_types.items():
                if column in fact_table.columns:
                    try:
                        fact_table[column] = fact_table[column].astype(dtype)
                    except Exception as e:
                        logger.error(f"Error converting column {column} to {dtype}: {e}")

        
        return fact_table
    

    def create_dimension_table(self, dataframe, columns_to_keep, new_column_names, column_types):
        """
        Creates a dimension table from the given dataframe with specified data types.

        Args:
        dataframe (pd.DataFrame): The source dataframe.
        columns_to_keep (list): List of columns to keep in the dimension table.
        new_column_names (dict): Dictionary mapping old column names to new column names.
        column_types (dict): Dictionary specifying the data type for each column.

        Returns:
        pd.DataFrame: The processed dimension table.
        """
        # Filter and rename columns
        dim_table = dataframe[columns_to_keep].drop_duplicates(subset=[columns_to_keep[0]]).reset_index(drop=True)
        dim_table.rename(columns=new_column_names, inplace=True)
        
        # Apply column data types
        for column, dtype in column_types.items():
            if column in dim_table:
                try:
                    dim_table[column] = dim_table[column].astype(dtype)
                except Exception as e:
                    logger.error(f"Error converting {column} to {dtype}: {e}")
                    # Optionally, handle or log the error

        dim_table.dropna(how='all', inplace=True)
        dim_table = dim_table.fillna(value=np.nan)
        return dim_table

    def load_and_process_all_data(self):
        """
        Loads and processes all required data.
        """ 
        # Load data
        self.Added_constructors_df = self.load_data('constructors.csv', file_type='csv')
        self.Added_results_df = self.load_data('results.csv', file_type='csv')
        self.Added_drivers_df = self.load_data('drivers.csv', file_type='csv')
        self.Added_races_df = self.load_data('races.csv', file_type='csv')

        not_processed_teams_df = self.load_data('raceToTeamPositions.xlsx')
        self.teams_df = self.add_new_teams(not_processed_teams_df)

        self.race_to_events_df = self.load_data('raceToEvents.xlsx')
        self.race_to_events_df = self.race_to_events_df[self.race_to_events_df['description'] == 'Race']
        self.race_events_to_driver_df = self.load_data('raceEventsToDriver.xlsx')
        self.season_to_race_df = self.load_data('seasonToRace.xlsx')
        self.race_to_venue_df = self.load_data('raceToVenue.xlsx')
        self.season_only_data_df = self.load_data('seasonData.xlsx')

    def prepare_data_tables(self):
        """
        Prepares data tables by defining column specifications and building dimension tables.
        It builds fact table and dimension tables for drivers, venues, teams, seasons, and race events.
        """
        fact_table_column_types = {
            'RacePerformanceId': 'string',
            'RaceId': 'string',
            'DriverId': 'string',
            'TeamId': 'string',
            'VenueId': 'string',
            'SeasonId': 'string',
            'FastestLap': 'string',
            'FastestLapTime': 'string',
            'FastestLapSpeed': 'float64',
            'TotalLaps': 'Int64',
            'FinalPosition': 'Int64',
            'GridPosition': 'Int64',
            'IndividualPoInt64s': 'Int64',
            'PitstopCount': 'Int64',
            'ActualTotalRaceTime': 'string',
            'Status': 'string'
        }

        driver_dim_columns = ['driverId', 'driver_name', 'driver_gender', 'driver_nationality', 'driver_country_code']
        driver_dim_column_names = {
            'driverId': 'DriverId', 'driver_name': 'DriverName', 'driver_gender': 'Gender', 
            'driver_nationality': 'Nationality', 'driver_country_code': 'CountryCode'
        }
        driver_dim_column_types = {
            'DriverId': 'string',
            'DriverName': 'string',
            'Gender': 'string',
            'Nationality': 'string',
            'CountryCode': 'string'
        }

        venue_dim_columns = ['venueId', 'name', 'city', 'country','country_code','venue_length','curves_left','curves_right','coordinates','laps']
        venue_dim_column_names = {
            'venueId': 'VenueId', 'name': 'Name', 'city': 'City', 
            'country': 'Country','country_code': 'CountryCode','venue_length': 'Length',
            'curves_left': 'CurvesLeft','curves_right': 'CurvesRight','coordinates': 'Coordinates', 
            'laps': 'Laps', 
        }
        venue_dim_column_types = {
            'VenueId': 'string',
            'Name': 'string',
            'City': 'string',
            'Country': 'string',
            'CountryCode': 'string',
            'Length': 'float64',
            'CurvesLeft': 'Int64',
            'CurvesRight': 'Int64',
            'Coordinates': 'string',
            'Laps': 'Int64'
        }

        team_dim_columns = ['teamId', 'team_name', 'team_nationality', 'team_country_code']
        team_dim_column_names = {
            'teamId': 'TeamId', 'team_name': 'TeamName', 'team_nationality': 'Nationality', 
            'team_country_code': 'CountryCode'
        }
        team_dim_column_types = {
            'TeamId': 'string',
            'TeamName': 'string',
            'Nationality': 'string',
            'CountryCode': 'string'
        }

        season_dim_columns = ['seasonId', 'description', 'scheduled', 'scheduled_end']
        season_dim_column_names = {
            'seasonId': 'SeasonId', 'description': 'Description', 'scheduled': 'StartDate', 
            'scheduled_end': 'EndDate'
        }
        season_dim_column_types = {
            'SeasonId': 'string',
            'Description': 'string',
            'StartDate': 'datetime64',  
            'EndDate': 'datetime64'
        }

        race_event_dim_columns = ['raceId', 'description', 'scheduled', 'scheduled_end','status','air_temperature','track_temperature','humidity','weather']
        race_event_column_names = {
            'raceId': 'RaceId', 'description': 'Description', 'scheduled': 'ScheduledStart', 
            'scheduled_end': 'ScheduledEnd','status': 'Status','air_temperature': 'AirTemp',
            'track_temperature': 'TrackTemp','humidity': 'Humidity','weather': 'Weather'
        }
        race_event_dim_column_types = {
            'RaceId': 'string',
            'Description': 'string',
            'ScheduledStart': 'datetime64',
            'ScheduledEnd': 'datetime64',
            'Status': 'string',
            'AirTemp': 'Int64',
            'TrackTemp': 'Int64',
            'Humidity': 'Int64',
            'Weather': 'string'
        }   
  
        self.tables_dict = {
            "fact_table": self.build_fact_table(self.Added_constructors_df, self.Added_results_df, self.Added_drivers_df, self.Added_races_df, self.teams_df, self.race_to_events_df, self.race_events_to_driver_df, self.season_to_race_df, self.race_to_venue_df,fact_table_column_types),
            "driver_dim": self.create_dimension_table(self.race_events_to_driver_df, driver_dim_columns, driver_dim_column_names, driver_dim_column_types),
            "venue_dim": self.create_dimension_table(self.race_to_venue_df, venue_dim_columns, venue_dim_column_names, venue_dim_column_types),
            "team_dim": self.create_dimension_table(self.teams_df, team_dim_columns, team_dim_column_names, team_dim_column_types),
            "season_dim": self.create_dimension_table(self.season_only_data_df, season_dim_columns, season_dim_column_names, season_dim_column_types),
            "race_event_dim": self.create_dimension_table(self.race_to_events_df, race_event_dim_columns, race_event_column_names, race_event_dim_column_types)
        }

def main():
    key_path = "/Users/ronnieasatourian/Desktop/F1 Case/red-bull-case-1-ab62fe2d9df9.json"
    RedBull_DATA_PATH_plus_added = '/Users/ronnieasatourian/Desktop/F1 Case/Requirements/f1_case_data'

    project_id = "red-bull-case-1"
    dataset_id = "redbull"
    
    f1_processor = F1DataProcessor(RedBull_DATA_PATH_plus_added, key_path, project_id)

    f1_processor.load_and_process_all_data()
    f1_processor.prepare_data_tables()

    for table_name, df in f1_processor.tables_dict.items():
        f1_processor.insert_data(dataset_id, table_name, df)

if __name__ == '__main__':
    main()
    
    