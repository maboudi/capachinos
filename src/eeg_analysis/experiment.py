class Experiment:
    def __init__(self, participant_id):
        """
        Initialize the Experiment with a participant ID, an empty list for anesthesia status,
        and an empty dictionary for analysis periods.

        Parameters:
        participant_id: The ID of the participant.
        """
        self.participant_id = participant_id
        self.anesthesia_status = []
        self.analysis_periods = {
            'pre_op_resting': None,
            'pre_oxygenation': None,
            'loc': None,  # loss of consciousness
            'maintenance': None,
            'peri_extubation': None,
            'emergence': None,  # anesthetic emergence
            'pacu_resting': None  # post-anesthesia care unit
        }

    def add_anesthesia_status(self, status):
        """
        Add a status to the anesthesia_status list.

        Parameters:
        status (dict): A dictionary containing anesthesia status data, e.g., {'time': timestamp, 'mac': value}.
        """
        # Ensure the status contains required fields
        required_fields = {'time', 'mac'}
        if not required_fields.issubset(status.keys()):
            raise ValueError(f"Status must contain the following fields: {required_fields}")

        self.anesthesia_status.append(status)

    def define_analysis_periods(self, events):
        """
        Define and store analysis periods based on clinically relevant events.

        Parameters:
        events (dict): A dictionary with event names as keys and their corresponding timestamps as values.
        """
        required_events = [
            'pre_op_resting', 'pre_oxygenation', 'loc',
            'maintenance', 'peri_extubation', 'emergence', 'pacu_resting'
        ]

        # Ensure all required events are present
        if not all(event in events for event in required_events):
            raise ValueError(f"Events must include the following keys: {required_events}")

        # Assign events to analysis periods
        for period in self.analysis_periods:
            self.analysis_periods[period] = events.get(period)

"""
# Example usage:
# Initialize an Experiment instance with a participant ID
experiment = Experiment(participant_id=1)

# Add anesthesia status entries
experiment.add_anesthesia_status({'time': '2023-05-01T08:00:00', 'mac': 0.8})
experiment.add_anesthesia_status({'time': '2023-05-01T08:30:00', 'mac': 1.2})

# Define analysis periods based on clinically relevant events
events = {
    'pre_op_resting': '2023-05-01T07:50:00',
    'pre_oxygenation': '2023-05-01T08:00:00',
    'loc': '2023-05-01T08:15:00',  # loss of consciousness
    'maintenance': '2023-05-01T08:20:00',
    'peri_extubation': '2023-05-01T09:00:00',
    'emergence': '2023-05-01T09:10:00',  # anesthetic emergence
    'pacu_resting': '2023-05-01T09:30:00'  # post-anesthesia care unit
}

experiment.define_analysis_periods(events)

print("Anesthesia Status:", experiment.anesthesia_status)
print("Analysis Periods:", experiment.analysis_periods)

"""