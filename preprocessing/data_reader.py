import xlrd
import pandas as pd


class ExcelReader:
    def __init__(self, file_path, sheets_names=None):
        self.file_path = file_path
        self.workbook = xlrd.open_workbook(file_path)
        self.sheets = [self.workbook.sheet_by_name(sheet_idx) for sheet_idx in sheets_names]

    def read_data(self):
        sheets_data = []
        for sheet in self.sheets:
            data = []
            for row in range(sheet.nrows):
                row_data = []
                for col in range(sheet.ncols):
                    cell_value = sheet.cell_value(row, col)
                    row_data.append(cell_value)
                data.append(row_data)
            sheets_data.append(data)
        return sheets_data


class DataProcessor:
    def __init__(self, data):
        self.data = data

    def find_common_columns(self, list_of_dfs):
        """
        Takes a list of pandas DataFrames and returns a list of column names
        common to all the DataFrames in the list.

        Parameters:
        list_of_dfs (list): A list of pandas DataFrame objects.

        Returns:
        list: A list of column names common to all DataFrames.
        """

        # Check if the list is not empty and contains DataFrames
        if not list_of_dfs or not all(isinstance(df, pd.DataFrame) for df in list_of_dfs):
            raise ValueError("Please provide a non-empty list of pandas DataFrames.")

        # Convert the columns of the first DataFrame to a set
        common_columns_set = set(list_of_dfs[0].columns)

        # Find the intersection with the sets of columns from the other DataFrames
        for dataframe in list_of_dfs[1:]:
            common_columns_set &= set(dataframe.columns)
        # Convert the resulting set to a list
        common_columns_list = list(common_columns_set)

        return common_columns_list

    def format_dataframe(self, chunk):
        """
        Formats a chunk of data into a DataFrame by setting the appropriate headers and indices.

        Parameters:
        chunk (iterable): A chunk of data.

        Returns:
        DataFrame: A formatted DataFrame with the appropriate headers and reset indices.
        """
        df = pd.DataFrame(chunk).drop([0])

        # Create new headers, ensuring they are unique by adding a sequence number if necessary.
        new_header = df.iloc[0] + ' (' + df.iloc[1] + ')'
        if new_header.duplicated().any():
            # If there are duplicates, we create a unique sequence to append to the duplicates.
            new_header += new_header.groupby(new_header).cumcount().astype(str).replace('0', '')

        # Apply the new unique headers and format the DataFrame.
        df = df[2:]
        df.columns = new_header
        return df.reset_index(drop=True)

    def save_to_dataframe(self):
        formatted_dfs = [self.format_dataframe(chunk) for chunk in self.data]

        common_columns = self.find_common_columns(formatted_dfs)
        if not common_columns:
            raise ValueError("No common columns exist across the DataFrames.")

        combined_df = pd.DataFrame()

        # Iterate over the list of DataFrames
        for df in formatted_dfs:
            # Select only the common columns for the current DataFrame
            selected_columns_df = df[common_columns]

            # Append the selected columns to the combined DataFrame
            combined_df = pd.concat([combined_df, selected_columns_df], ignore_index=True)

        return combined_df



