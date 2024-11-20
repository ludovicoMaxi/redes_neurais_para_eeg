class Patient:
    def __init__(self, id_patient, eeg_data, eeg_record, fs, clinical_features = []):
        self.id_patient = id_patient
        self.clinical_features = clinical_features 
        self.eeg_data = eeg_data 
        self.eeg_record = eeg_record
        self.fs = fs