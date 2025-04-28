import os
import h5py
import pickle
import numpy as np
from typing import Tuple, List
from tqdm.auto import tqdm


def shuffled_batch_indices(percent_data: float, all_y: np.ndarray, attack_dirs: List[str] = None) -> np.ndarray:
    '''
    Select a random subset of indices from the dataset, ensuring an equal number of anomalies and non-anomalies.
    Split among attack types as well as normal data. Target proportions:
    - 50% normal, 50% attacks
    - Among attacks: 19% ddos, 74% dos, 1% ransomware, 6% scanning
    
    :param percent_data: Percentage of data to select (0.0-1.0)
    :param all_y: Labels (0 for normal, 1 for attack)
    :param attack_dirs: Optional list of attack directories for each sample, used for more precise distribution
    :return: Array of selected indices
    '''
    # Get indices for normal and attack samples
    normal_indices = np.where(all_y == 0)[0]
    attack_indices = np.where(all_y == 1)[0]
    
    # Calculate total number of samples to select
    total_indices = len(all_y)
    num_indices_to_select = int(total_indices * percent_data)
    
    # Make sure num_indices_to_select is even for equal split between normal and attack
    if num_indices_to_select % 2 != 0:
        num_indices_to_select -= 1
    
    # Split between normal and attack (50% each)
    num_normal = min(num_indices_to_select // 2, len(normal_indices))
    num_attack = min(num_indices_to_select // 2, len(attack_indices))
    
    # Adjust if there are not enough samples in either category
    if num_normal < num_indices_to_select // 2:
        num_attack = min(num_indices_to_select - num_normal, len(attack_indices))
    elif num_attack < num_indices_to_select // 2:
        num_normal = min(num_indices_to_select - num_attack, len(normal_indices))
    
    # Randomly select normal indices
    selected_normal_indices = np.random.choice(normal_indices, size=num_normal, replace=False)
    
    # Check if attack_dirs is valid before using it
    valid_attack_dirs = (attack_dirs is not None and 
                         len(attack_dirs) == len(all_y) and
                         isinstance(attack_dirs[0], str))
    
    if valid_attack_dirs:
        # Target proportions for attack types
        attack_props = {
            'ddos': 0.24,
            'dos': 0.70,
            'ransomware': 0.01,
            'scanning': 0.05
        }
        
        # Group attack indices by attack type
        attack_by_type = {}
        for i in attack_indices:
            attack_type = attack_dirs[i]
            if attack_type not in attack_by_type:
                attack_by_type[attack_type] = []
            attack_by_type[attack_type].append(i)
        
        # Only proceed if we found attack types
        if attack_by_type:
            # Calculate target number of samples for each attack type
            attack_targets = {}
            for attack_type, prop in attack_props.items():
                if attack_type in attack_by_type:
                    target = int(num_attack * prop)
                    # Ensure we don't try to select more than available
                    available = len(attack_by_type[attack_type])
                    attack_targets[attack_type] = min(target, available)
            
            # Adjust targets to match exact total
            total_targeted = sum(attack_targets.values())
            
            # First, ensure we don't exceed what's available
            total_available = sum(len(indices) for indices in attack_by_type.values())
            num_attack_adjusted = min(num_attack, total_available)
            
            if total_targeted < num_attack_adjusted:
                # Distribute remaining samples proportionally among types with available samples
                remaining = num_attack_adjusted - total_targeted
                
                # Sort attack types by those with most room left
                types_with_room = [(t, len(attack_by_type[t]) - attack_targets.get(t, 0)) 
                                  for t in attack_by_type if t in attack_targets and 
                                  len(attack_by_type[t]) > attack_targets.get(t, 0)]
                types_with_room.sort(key=lambda x: x[1], reverse=True)
                
                # Add remaining samples one by one to types with most room
                for _ in range(remaining):
                    for attack_type, room in types_with_room:
                        if room > 0:
                            attack_targets[attack_type] += 1
                            # Update room count
                            idx = next(i for i, (t, _) in enumerate(types_with_room) if t == attack_type)
                            types_with_room[idx] = (attack_type, types_with_room[idx][1] - 1)
                            # Re-sort the list
                            types_with_room.sort(key=lambda x: x[1], reverse=True)
                            break
            
            # Select samples from each attack type
            selected_attack_indices = []
            for attack_type, target in attack_targets.items():
                if target > 0 and attack_type in attack_by_type:
                    type_indices = np.random.choice(attack_by_type[attack_type], size=target, replace=False)
                    selected_attack_indices.extend(type_indices)
            
            selected_attack_indices = np.array(selected_attack_indices)
            
            # Print the distribution of attack types selected
            '''print(f"Attack type distribution in the selected batch:")
            print(f"  Normal: {len(selected_normal_indices)} samples")
            for attack_type in attack_targets:
                if attack_type in attack_by_type:
                    type_count = sum(1 for i in selected_attack_indices if attack_dirs[i] == attack_type)
                    type_percent = type_count / len(selected_attack_indices) * 100 if len(selected_attack_indices) > 0 else 0
                    print(f"  {attack_type}: {type_count} samples ({type_percent:.1f}%)")'''
            
            # Verify we got exactly the right number
            if len(selected_attack_indices) == num_attack_adjusted:
                valid_attack_dirs = True
                # If we had to adjust down due to availability, print a message
                if num_attack_adjusted < num_attack:
                    print(f"Warning: Only {num_attack_adjusted}/{num_attack} attack samples available.")
            else:
                # Fall back to random sampling if something went wrong
                valid_attack_dirs = False
                print(f"Warning: Expected {num_attack_adjusted} attack samples but got {len(selected_attack_indices)}.")
    
    if not valid_attack_dirs:
        # Without attack_dirs, just randomly sample from all attack indices
        selected_attack_indices = np.random.choice(attack_indices, size=num_attack, replace=False)
        print("Warning: Using random sampling for attack indices due to invalid attack_dirs.")
    
    # Combine and shuffle
    all_selected_indices = np.concatenate([selected_normal_indices, selected_attack_indices])
    np.random.shuffle(all_selected_indices)
    
    return all_selected_indices


class IDSDataset:
    def __init__(self,
                 h5_path: str = None,
                 train_x_path: str = None,
                 train_y_path: str = None,
                 val_x_path: str = None,
                 val_y_path: str = None,
                 test_x_path: str = None,
                 test_y_path: str = None,
                 debug: bool = False):
        self._debug = debug

        # List of attack directories to process
        self.attack_dirs = ['ddos', 'dos', 'ransomware', 'scanning', 'normal']

        if h5_path is not None:
            if debug:
                print("Processing HDF5 Dataset...")
            self._process_h5(h5_path)
            if debug:
                print("\tDone.")
        else:
            if debug:
                print("Loading Pickle Dataset...")
            with open(train_x_path, "rb") as f:
                self.train_x = pickle.load(f)
            with open(train_y_path, "rb") as f:
                self.train_y = pickle.load(f)
            with open(val_x_path, "rb") as f:
                self.val_x = pickle.load(f)
            with open(val_y_path, "rb") as f:
                self.val_y = pickle.load(f)
            with open(test_x_path, "rb") as f:
                self.test_x = pickle.load(f)
            with open(test_y_path, "rb") as f:
                self.test_y = pickle.load(f)

                
    def get_dminmax(self) -> Tuple[List[float], List[float]]:
        """
        Calculate min and max values for each feature in the training data.
        Returns a tuple of (mins, maxs) where each is a list of values.
        """
        # Extract feature statistics across all runs
        all_feature_values = []
        
        for run in self.train_x:
            # Get input vector (shape: 31, run_length)
            input_vector = run[1]
            
            # Process each feature (across each row of the input vector)
            for feature_idx in range(input_vector.shape[0]):
                feature_values = input_vector[feature_idx]
                all_feature_values.append(feature_values)
        
        # Stack all values for each feature
        all_feature_values = np.hstack(all_feature_values)
        
        # Reshape to (31, num_total_values)
        all_feature_values = all_feature_values.reshape(31, -1)
        
        # Calculate min and max for each feature
        mins = np.min(all_feature_values, axis=1).tolist()
        maxs = np.max(all_feature_values, axis=1).tolist()
        
        return mins, maxs

    def save_pickle(self, dir_path: str):
        """
        Save the dataset as pickle files.
        """
        if self._debug:
            print('Saving Pickle Data...')
        os.makedirs(dir_path, exist_ok=True)

        #  Save training data
        with open(os.path.join(dir_path, "train_x.pkl"), "wb") as f:
            pickle.dump(self.train_x, f)
        with open(os.path.join(dir_path, "train_y.pkl"), "wb") as f:
            pickle.dump(self.train_y, f)

        # Save validation data
        with open(os.path.join(dir_path, "val_x.pkl"), "wb") as f:
            pickle.dump(self.val_x, f)
        with open(os.path.join(dir_path, "val_y.pkl"), "wb") as f:
            pickle.dump(self.val_y, f)

        # Save testing data
        with open(os.path.join(dir_path, "test_x.pkl"), "wb") as f:
            pickle.dump(self.test_x, f)
        with open(os.path.join(dir_path, "test_y.pkl"), "wb") as f:
            pickle.dump(self.test_y, f)

        if self._debug:
            print('\tDone.')


    @staticmethod
    def extract_data(attack_dirs: List[str], h5_file: h5py.File) -> Tuple[List, List[np.ndarray], List]:
        """
        Extract data from H5 file structure.

        Parameters:
        -----------
        attack_dirs : List[str]
            List of attack directories to process
        h5_file : h5py.File
            Open H5 file

        Returns:
        --------
        Tuple[List, List[np.ndarray], List]
            x_list: List of runs, where each run is [attack_dir, input_vector]
                    attack_dir is a string (e.g., "normal", "ddos", etc.)
                    input_vector is a numpy array of shape (31, run_length)
            y_list: List of arrays of labels for each run (packet-level labels)
            origin_dirs: List of the original attack directory for each run
        """
        x_list = []
        y_list = []
        origin_dirs = []
        label_count = 0
        skipped_runs_count = 0
        max_run_length = 50000
        min_run_length = 2000

        for attack_dir in tqdm(attack_dirs, desc="Processing attack directories"):
            if attack_dir not in h5_file.keys():
                print(f"Attack directory {attack_dir} not found in dataset. Skipping.")
                continue

            attack_dir_group = h5_file[attack_dir]

            for run in attack_dir_group.keys():
                run_group = attack_dir_group[run]

                if 'input_vectors' not in run_group or 'labels' not in run_group:
                    print(f"Skipping {run} in {attack_dir}: Missing required datasets.")
                    continue

                # Get data
                input_vectors = run_group['input_vectors'][:]
                labels = run_group['labels'][:]

                # Skip if no data
                if len(input_vectors) == 0:
                    print(f"Skipping {run} in {attack_dir}: No data.")
                    continue

                # Ensure input_vectors is 2D
                if len(input_vectors.shape) == 1:
                    input_vectors = input_vectors.reshape(1, -1)
                
                # Transpose the input vectors to have shape (31, run_length)
                # Assuming input_vectors originally has shape (run_length, 31)
                if input_vectors.shape[1] == 31:  # Check if features are in the second dimension
                    input_vectors = input_vectors.T  # Transpose to (31, run_length)
                
                # Check if run is too long (over max_run_length)
                run_length = input_vectors.shape[1]
                if run_length > max_run_length or run_length < min_run_length:
                    skipped_runs_count += 1
                    continue
                
                # Create a single entry for this run with [attack_dir, input_vectors]
                run_data = [attack_dir, input_vectors]
                
                # Store the full array of packet-level labels
                # Ensure labels has the same length as the run
                if len(labels) != run_length:
                    print(f"Warning: Labels length ({len(labels)}) does not match run length ({run_length}) in {run} in {attack_dir}. Adjusting.")
                    
                
                # Add to lists
                x_list.append(run_data)
                y_list.append(labels)  # Store the full array of packet labels
                origin_dirs.append(attack_dir)  # Store the origin attack directory

                # Count runs with at least one attack packet for statistics
                if np.any(labels == 1):
                    label_count += 1

        if len(x_list) == 0:
            raise ValueError("No valid data found in the H5 file.")

        print(f"Extracted {len(x_list)} runs, {label_count} containing attacks. {len(x_list) - label_count} entirely normal.")
        if skipped_runs_count > 0:
            print(f"Skipped {skipped_runs_count} runs because they were outside the length range of {min_run_length}-{max_run_length} time steps.")
        
        # Note: y_list is now a list of numpy arrays (one per run) rather than a single numpy array
        return x_list, y_list, origin_dirs
    

    def _process_h5(self, path: str):
        """
        Process H5 file and split data into train, validation, and test sets.
        Split is proportional to each attack directory's number of runs with a 60/10/30 split ratio.

        Parameters:
        -----------
        path : str
            Path to the H5 file
        """
        with h5py.File(path, 'r') as h:
            # Get all data and origin directories
            x_list, y_list, origin_dirs = IDSDataset.extract_data(self.attack_dirs, h)
            # Convert origin_dirs to numpy array for easier indexing
            origin_dirs = np.array(origin_dirs)

            # Group data by attack directory
            attack_dir_indices = {attack_dir: [] for attack_dir in self.attack_dirs}

            # Identify which attack directory each run belongs to
            for i, attack_dir in enumerate(origin_dirs):
                attack_dir_indices[attack_dir].append(i)

            # Define split ratios
            train_ratio = 0.6  # 60% for training
            val_ratio = 0.1  # 10% for validation
            test_ratio = 0.3  # 30% for testing

            # Initialize lists for each split
            train_indices = []
            val_indices = []
            test_indices = []

            # Split each attack directory's data according to the ratios
            for attack_dir, indices in attack_dir_indices.items():
                # Skip if no runs for this attack directory
                if not indices:
                    continue

                # Get data for this attack directory
                dir_indices = np.array(indices)
                
                # Check for attack packets in each run
                dir_has_attack = []
                for idx in dir_indices:
                    has_attack = np.any(y_list[idx] == 1)
                    dir_has_attack.append(has_attack)
                dir_has_attack = np.array(dir_has_attack)
                
                # Group by whether runs contain attacks or are all normal
                dir_anomaly_indices = dir_indices[dir_has_attack]
                dir_non_anomaly_indices = dir_indices[~dir_has_attack]

                # Split anomaly data
                n_anomaly = len(dir_anomaly_indices)
                if n_anomaly > 0:
                    # Shuffle indices to randomize selection
                    np.random.shuffle(dir_anomaly_indices)

                    # Calculate split points
                    train_end = int(train_ratio * n_anomaly)
                    val_end = train_end + int(val_ratio * n_anomaly)

                    # Add to respective splits
                    train_indices.extend(dir_anomaly_indices[:train_end].tolist())
                    val_indices.extend(dir_anomaly_indices[train_end:val_end].tolist())
                    test_indices.extend(dir_anomaly_indices[val_end:].tolist())

                # Split non-anomaly data
                n_non_anomaly = len(dir_non_anomaly_indices)
                if n_non_anomaly > 0:
                    # Shuffle indices to randomize selection
                    np.random.shuffle(dir_non_anomaly_indices)

                    # Calculate split points
                    train_end = int(train_ratio * n_non_anomaly)
                    val_end = train_end + int(val_ratio * n_non_anomaly)

                    # Add to respective splits
                    train_indices.extend(dir_non_anomaly_indices[:train_end].tolist())
                    val_indices.extend(dir_non_anomaly_indices[train_end:val_end].tolist())
                    test_indices.extend(dir_non_anomaly_indices[val_end:].tolist())

            # Randomize the order of indices within each split
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)

            # Split data - each X sample corresponds to a Y array of packet labels
            self.train_x = [x_list[i] for i in train_indices]
            self.train_y = [y_list[i] for i in train_indices]  # Each element is a label array for a run

            self.val_x = [x_list[i] for i in val_indices]
            self.val_y = [y_list[i] for i in val_indices]      # Each element is a label array for a run

            self.test_x = [x_list[i] for i in test_indices]
            self.test_y = [y_list[i] for i in test_indices]    # Each element is a label array for a run

            if self._debug:
                print(f"Dataset Statistics:")
                print(f"\tTraining Set:   {len(self.train_y)} runs")
                print(f"\tValidation Set: {len(self.val_y)} runs")
                print(f"\tTesting Set:    {len(self.test_y)} runs")
                
                # Calculate and print statistics about packet labels
                train_packet_counts = [len(y) for y in self.train_y]
                val_packet_counts = [len(y) for y in self.val_y]  
                test_packet_counts = [len(y) for y in self.test_y]
                
                train_attack_packets = sum(np.sum(y) for y in self.train_y)
                val_attack_packets = sum(np.sum(y) for y in self.val_y)
                test_attack_packets = sum(np.sum(y) for y in self.test_y)
                
                total_train_packets = sum(train_packet_counts)
                total_val_packets = sum(val_packet_counts)
                total_test_packets = sum(test_packet_counts)
                
                print(f"Packet Statistics:")
                print(f"\tTraining:   {total_train_packets} packets, {train_attack_packets} ({train_attack_packets/total_train_packets:.2%}) labeled as attacks")
                print(f"\tValidation: {total_val_packets} packets, {val_attack_packets} ({val_attack_packets/total_val_packets:.2%}) labeled as attacks")
                print(f"\tTesting:    {total_test_packets} packets, {test_attack_packets} ({test_attack_packets/total_test_packets:.2%}) labeled as attacks")

                # Print distribution of attack directories in each split
                for attack_dir in self.attack_dirs:
                    # Use numpy array operations for efficiency
                    train_count = np.sum(origin_dirs[train_indices] == attack_dir)
                    val_count = np.sum(origin_dirs[val_indices] == attack_dir)
                    test_count = np.sum(origin_dirs[test_indices] == attack_dir)
                    total_count = train_count + val_count + test_count

                    if total_count > 0:
                        print(f"\t{attack_dir}: Train {train_count} ({train_count / total_count:.1%}), "
                              f"Val {val_count} ({val_count / total_count:.1%}), "
                              f"Test {test_count} ({test_count / total_count:.1%})")