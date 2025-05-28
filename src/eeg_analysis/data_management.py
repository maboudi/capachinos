import pandas as pd
import warnings

def load_patient_group_labels(file_path):
    """
    Load an Excel file and group patients based on their labels and preprocessing status.

    Parameters:
        file_path (str): Path to the Excel file.
    Returns:
        dict: Dictionary with grouped patient labels if successful, None otherwise.
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        print("Excel file loaded successfully.")
    except FileNotFoundError:
        warnings.warn(f"Warning: The file at {file_path} was not found.", UserWarning)
        return None
    except Exception as e:
        warnings.warn(f"Warning: An error occurred while loading the Excel file: {e}", UserWarning)
        return None

    # Initialize the dictionary for groups
    groups = {'A': [], 'B': []}

    try:
        # Print column names to verify
        print(f"Column names: {df.columns}")

        # Populate the dictionary based on the group label
        for index, row in df.iterrows():
            try:
                label = row['Patient_Label']
                group = row['Group_Number']
                data_available = row['Preprocessed_Data_Available']
                valid_MAC = row['Valid MAC_and_normal_closure']

                if group == 1 and data_available == 1 and valid_MAC == 1:
                    groups['A'].append(label)
                elif group == 2 and data_available == 1 and valid_MAC == 1:
                    groups['B'].append(label)
            except KeyError as e:
                warnings.warn(f"Warning: The column {e} is missing from the Excel file.", UserWarning)
            except Exception as e:
                warnings.warn(f"Warning: An error occurred while processing the row {index}: {e}", UserWarning)

        print(groups)
        print(f"Group A: {len(groups['A'])} subjects, Group B: {len(groups['B'])} subjects")

    except Exception as e:
        warnings.warn(f"Warning: An error occurred while processing the data: {e}", UserWarning)
        return None

    return groups