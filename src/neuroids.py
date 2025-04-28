import time
import numpy as np
from py_apps.utils.classify_app import ClassifyApp
import eons
import neuro
import traceback
from neuro import IO_Stream
from tqdm.auto import tqdm
from multiprocessing import Pool, Manager
from idsdb import IDSDataset, shuffled_batch_indices
from idsdb import IDSDataset
from typing import Tuple, List

def mp_fitness(bundle):
    app, net, proc, proc_params, shared_x, shared_y, start_i, num_elems = bundle

    try:
        processor = proc(proc_params)
        all_decisions = []
        all_labels = []

        # Process each run to collect packet-level decisions
        for i in range(num_elems):
            data_index = start_i + i
            run_x = shared_x[data_index]
            run_y = shared_y[data_index]

            # Process data with the network to get packet-level decisions
            decisions = app.process_run(processor, net, run_x)
            
            # Add decisions and true labels for each packet
            all_decisions.extend(decisions)
            all_labels.extend(run_y)
        
       
        # Find optimal threshold based on J score
        rolling_sums = app.calculate_rolling_sums(all_decisions)
        
        # Define fixed thresholds directly here
        fixed_thresholds = [
            0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30,
        ]
        
        best_threshold, best_j = app.find_optimal_threshold(
            all_decisions, rolling_sums, fixed_thresholds, all_labels)
        
        # Apply the best threshold to get final decisions
        windowed_decisions = app.apply_threshold(rolling_sums, best_threshold)
        
        # Calculate metrics using the windowed decisions
        fitness_result = app.calculate_metrics(windowed_decisions, all_labels)
        fitness_result["threshold"] = best_threshold
        
        return fitness_result
    except Exception as e:
        print(f"Error in mp_fitness: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def mp_evaluate(bundle):
    """
    Process a batch of runs with a given network at packet level.
    This function must be defined at the module level for multiprocessing.
    
    :param bundle: A tuple containing (app, network, proc_cls, proc_params, 
                  runs_x, runs_y, start_index, threshold, window_size)
    :return: Dictionary with results for this batch
    """
    app, network, proc_cls, proc_params, runs_x, runs_y, start_index, threshold, window_size = bundle
    
    try:
        processor = proc_cls(proc_params)
        all_decisions = []
        all_true_labels = []
        
        # Process each run in this batch
        for i, (run_x, run_y) in enumerate(zip(runs_x, runs_y)):
            # Process data with the network to get packet-level decisions
            decisions = app.process_run(processor, network, run_x)
            
            all_decisions.extend(decisions)
            all_true_labels.extend(run_y)
        
        # Apply rolling window approach if window_size is enabled
        if window_size > 0:
            # Calculate rolling sums for this batch
            rolling_sums = app.calculate_rolling_sums(all_decisions)
            
            # Apply the threshold to get windowed decisions
            windowed_decisions = app.apply_threshold(rolling_sums, threshold)
            
            # Calculate metrics for this batch
            metrics = app.calculate_metrics(windowed_decisions, all_true_labels)
            
            # Return only the metrics specified
            return {
                "start_index": start_index,
                "windowed_decisions": windowed_decisions,
                "true_labels": all_true_labels,
                "threshold": threshold,
                "mcc": metrics["mcc"],
                "j": metrics["j"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
                "p": metrics["p"],
                "n": metrics["n"]
            }
        else:
            # Original approach - use raw decisions directly
            metrics = app.calculate_metrics(all_decisions, all_true_labels)
            
            # Return only the metrics specified
            return {
                "start_index": start_index,
                "decisions": all_decisions,
                "true_labels": all_true_labels,
                "mcc": metrics["mcc"],
                "j": metrics["j"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "ppv": metrics["ppv"],
                "npv": metrics["npv"],
                "p": metrics["p"],
                "n": metrics["n"]
            }
    except Exception as e:
        print(f"Error in mp_evaluate: {e}")
        traceback.print_exc()
        return {"error": str(e), "start_index": start_index}

class NeuroIDS(ClassifyApp):
    def __init__(self,
                 config,
                 dminmax: Tuple[List[float], List[float]],
                 percent_runs: float = 1.0,
                 shuffle: bool = True,
                 temporal: bool = True,
                 quiet: bool = False):
        
        if percent_runs <= 0:
            raise ValueError("percent_runs must be greater than 0")
        
        self.percent_runs = percent_runs
        self.shuffle = shuffle
        self.temporal = temporal
        self.epoch = 0
        self.quiet = quiet

        # how long the network runs for before decoding the output
        self.runtime = config["runtime"]
        # filename where the network is/will be stored
        self.network_filename = config["network_filename"]

        self.log_file = self.network_filename.replace('.txt', '.log')
        # binary classifier, so labels or simply 0 and 1
        self.labels = sorted([0, 1])
        self.config = config
        self.config["app_params"] = {}
        # Spike encoders
        if "encoder_array" in config.keys():
            self.encoder = neuro.EncoderArray(config["encoder_array"])
        else:
            ea_json = {}
            if "encoders" in config.keys():
                ea_json["encoders"] = config["encoders"]
            elif "named_encoders" in config.keys():
                ea_json["named_encoders"] = config["named_encoders"]
                ea_json["use_encoders"] = config["use_encoders"]
            else:
                ea_json["encoders"] = [{"spike": {}}]

            ea_json["dmin"], ea_json["dmax"] = dminmax
            ea_json["interval"] = config["encoder_interval"]

            self.encoder = neuro.EncoderArray(ea_json)
        # Spike decoders
        if "decoder_array" in config.keys():
            self.decoder = neuro.DecoderArray(config["decoder_array"])
        else:
            da_json = {}
            if "decoders" in config.keys():
                da_json["decoders"] = config["decoders"]
            elif "named_decoders" in config.keys():
                da_json["named_decoders"] = config["named_decoders"]
                da_json["use_decoders"] = config["use_decoders"]
            else:
                da_json["decoders"] = [config["decoder"]]

            da_json["dmin"] = [0]
            da_json["dmax"] = [1]

            if ("divisor" in config.keys()):
                da_json["divisor"] = config["divisor"]
            else:
                da_json["divisor"] = config["runtime"]

            self.decoder = neuro.DecoderArray(da_json)

        self.n_inputs = self.encoder.get_num_neurons()
        self.n_outputs = self.decoder.get_num_neurons()

        if "fitness_type" in config.keys():
            self.fitness_type = config["fitness_type"]

        if self.config["all_counts_stream"] is not None:
            self.iostream = IO_Stream()
            self.iostream.create_output_from_json(self.config["all_counts_stream"])

        # Change window size for rolling sum from 1000 to 200
        self.window_size = config.get("window_size", 50)
        print(f"Using rolling window size: {self.window_size} packets")

    def identify_attack_events(self, true_labels):
        """
        Identify contiguous blocks of attack packets as individual attack events.
        
        :param true_labels: List of packet-level true labels (0 or 1)
        :return: List of attack events, where each event is a tuple (start_idx, end_idx)
        """
        attack_events = []
        in_attack = False
        start_idx = None
        
        for i, label in enumerate(true_labels):
            if label == 1 and not in_attack:
                # Start of a new attack event
                in_attack = True
                start_idx = i
            elif label == 0 and in_attack:
                # End of an attack event
                in_attack = False
                end_idx = i - 1
                attack_events.append((start_idx, end_idx))
        
        # Handle case where an attack extends to the end of the sequence
        if in_attack:
            attack_events.append((start_idx, len(true_labels) - 1))
            
        return attack_events

    def calculate_metrics(self, decisions, true_labels):
        """
        Calculate metrics using a hybrid approach:
        - Event-based True Positives (TP) and False Negatives (FN)
        - Packet-based False Positives (FP) and True Negatives (TN)
        
        :param decisions: List of packet-level decisions (0 or 1)
        :param true_labels: List of packet-level true labels (0 or 1)
        :return: Dictionary of calculated metrics
        """
        # Find all attack events (contiguous blocks of packets labeled as 1)
        attack_events = self.identify_attack_events(true_labels)
        # Event-based TP and FN
        TP = 0  # Count of attack events with at least one detection
        FN = 0  # Count of attack events with no detection
        
        for start_idx, end_idx in attack_events:
            # Check if any packet in this attack event was detected
            detected = False
            for i in range(start_idx, end_idx + 1):
                if i < len(decisions) and decisions[i] == 1:
                    detected = True
                    break
            
            if detected:
                TP += 1  # Attack event was detected
            else:
                FN += 1  # Attack event was missed
        
        # Packet-based FP and TN
        FP = 0  # Normal packets with a detection
        TN = 0  # Normal packets with no detection
        
        for i, label in enumerate(true_labels):
            if label == 0:  # Normal packet
                if i < len(decisions) and decisions[i] == 1:
                    FP += 1  # False positive
                else:
                    TN += 1  # True negative
        
        # Calculate metrics
        total_attack_events = len(attack_events)
        total_normal_packets = true_labels.count(0) if isinstance(true_labels, list) else np.sum(np.array(true_labels) == 0)
        
        # Event-based TPR
        tpr = TP / total_attack_events if total_attack_events > 0 else 1.0
        fnr = FN / total_attack_events if total_attack_events > 0 else 0.0
        
        # Packet-based FPR
        fpr = FP / total_normal_packets if total_normal_packets > 0 else 0.0
        tnr = TN / total_normal_packets if total_normal_packets > 0 else 1.0
        
        # Calculate precision (TP events / (TP events + FP packets))
        if TP + FP > 0:
            ppv = TP / (TP + FP)
        else:
            ppv = 0.0
            
        # NPV (TN packets / (TN packets + FN events))
        if TN + FN > 0:
            npv = TN / (TN + FN)
        else:
            npv = 0.0

        #Youdans J statistic
        j = tpr + tnr - 1.0 
        # Matthews Correlation Coefficient (MCC)
        mcc = np.sqrt(tpr * tnr * ppv * npv)  # Matthew's Correlation Coefficient
        
        return {
            "j": j, "mcc": mcc, "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr, 
             "tp": TP, "fp": FP, "tn": TN, "fn": FN, "p" : TP + FP, "n": TN + FN,
            "ppv": ppv, "npv": npv,
        }

    def calculate_rolling_sums(self, decisions):
        """
        Calculate rolling sums over a window of packet decisions.
        Each position represents the sum of spikes in a window of the last 'window_size' packets.
        
        :param decisions: List of packet-level decisions (0 or 1)
        :return: List of rolling sums with same length as decisions
        """
        if not decisions:
            return []
            
        rolling_sums = []
        for i in range(len(decisions)):
            # Define the start of the current window
            start_idx = max(0, i - self.window_size + 1)
            # Sum all decisions in current window
            current_sum = sum(decisions[start_idx:i+1])
            rolling_sums.append(current_sum)
            
        return rolling_sums
    
    def apply_threshold(self, rolling_sums, threshold):
        """
        Apply a threshold to rolling sums to get final decisions
        
        :param rolling_sums: List of rolling sum values 
        :param threshold: Threshold value to apply
        :return: List of thresholded decisions (0 or 1)
        """
        return [1 if sum_val >= threshold else 0 for sum_val in rolling_sums]
    
    def find_optimal_threshold(self, raw_decisions, rolling_sums, thresholds, true_labels):
        """
        Find the optimal threshold that maximizes the J score
        
        :param raw_decisions: Original network decisions
        :param rolling_sums: List of rolling sum values
        :param thresholds: List of threshold values to try
        :param true_labels: List of true labels
        :return: Tuple of (best threshold, best J score)
        """
        best_threshold = 0
        best_j = -1.0
        
    
        for threshold in thresholds:
            # Apply threshold to get decisions
            thresholded_decisions = self.apply_threshold(rolling_sums, threshold)
            
            # Calculate metrics
            metrics = self.calculate_metrics(thresholded_decisions, true_labels)
            j_score = metrics["j"]
            
            if j_score > best_j:
                best_j = j_score
                best_threshold = threshold
            
        return best_threshold, best_j

    def fitness(self, net, proc, data_x, data_y, show_progress=False, start_index=None, num_elems=None):
        """
        Calculate the fitness parameters using hybrid metrics approach with rolling window
        
        :param net: a single network
        :param proc: a processor instantiation
        :param data_x: dataset inputs, a list of runs
        :param data_y: dataset target output - packet-level labels
        :param show_progress: True to show progress printouts, False for quiet mode
        :param start_index: index of data run to start at
        :param num_elems: number of 'runs' to process, starting at start_index
        :return: dictionary containing hybrid metrics
        """
        all_raw_decisions = []
        all_labels = []

        if start_index is None:
            start_index = 0
            num_elems = len(data_x)

        for i in tqdm(range(num_elems), disable=not show_progress, leave=False, desc="Data Samples     "):
            data_index = i + start_index
            run_x = data_x[data_index]
            run_y = data_y[data_index]

            # Process data with the network to get packet-level decisions
            raw_decisions = self.process_run(proc, net, run_x)
            
            # Add decisions and true labels for each packet
            all_raw_decisions.extend(raw_decisions)
            all_labels.extend(run_y)

        # Apply rolling window approach
        # Calculate rolling sums
        rolling_sums = self.calculate_rolling_sums(all_raw_decisions)
        
        # Define fixed thresholds directly - for a 200-packet window
        fixed_thresholds = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50
        ]
        
        # Find optimal threshold
        best_threshold, best_j = self.find_optimal_threshold(
            all_raw_decisions, rolling_sums, fixed_thresholds, all_labels)
        
        # Apply the best threshold to get final decisions
        windowed_decisions = self.apply_threshold(rolling_sums, best_threshold)
        
        # Calculate metrics using the windowed decisions
        metrics = self.calculate_metrics(windowed_decisions, all_labels)
        metrics["threshold"] = int(best_threshold)  # Ensure threshold is an integer
        
        return metrics

    def get_prediction(self, proc, net, x, i=None):
        """
        Perform prediction on a single input packet
        """
        for i in range(self.n_outputs):
            proc.track_output_events(i)
            
        if self.config["printing_params"]["show_all_counts"]:
            for n in net.nodes():
                proc.track_neuron_events(n)

        spikes = self.encoder.get_spikes(x)

        proc.apply_spikes(spikes)
        
        proc.run(self.runtime)
        
        spikes = [proc.output_count(i) for i in range(self.n_outputs)]
        
        decision = 1 if np.array(spikes).sum() > 0 else 0

        if self.config["printing_params"]["show_all_counts"]:
            v = proc.neuron_counts()
            if self.config["all_counts_stream"] is not None:
                rv = {}
                net.make_sorted_node_vector()
                ids = [x.id for x in net.sorted_node_vector]
                rv["Neuron Alias"] = ids
                rv["Event Counts"] = v
                self.iostream.write_json(rv)
                time.sleep(0.1)
            else:
                # TODO: Set the timestep correctly
                timestep = -1
                str_output = "Step " + str(timestep) + "Counts:     "
                for i in range(len(v)):
                    str_output += str(v[i]) + " "
                print(str_output)

        return decision

    def process_run(self, proc, net, xdata: np.ndarray):
        """
        Process all packets in a run and return packet-level decisions.
        
        :param proc: processor to use
        :param net: network to use
        :param xdata: input data with shape (31, packet_count)
        :param ydata: ground truth labels with shape (packet_count,)
        :return: List of packet-level decisions
        """
        if net is not None:
            proc.load_network(net)

        if self.temporal:
            proc.clear_activity()
    
        packet_decisions = []
        
        for i in range(xdata.shape[1]):
            x = xdata[:, i]
            if not self.temporal:
                proc.clear_activity()
                
            decision = self.get_prediction(proc, net, x.tolist(), i=None)
            packet_decisions.append(decision)

        return packet_decisions

    def train(self, dataset: IDSDataset, train_config, proc, proc_params, temp_net=None, verbose=False):
        num_epochs = train_config["num_epochs"]

        other = {}
        other["sim_time"] = self.runtime
        other["proc_name"] = self.config["proc_name"]
        other["app_name"] = self.config["app_name"]
        other["temporal"] = self.temporal
        other["dimensionality"] = (self.n_inputs, self.n_outputs)

        processor = proc(proc_params)
        props = processor.get_network_properties()

        self.fitness_type = train_config["fitness"]

        evolver = eons.EONS(train_config["eons_params"])
        pop = self.get_population(evolver, train_config, props, temp_net)

        best_train = -1.0
        best_val = -1.0
        best_f = None
        self.overall_best_net = None
        best_threshold = 0

        t1 = time.time()
        for i in tqdm(range(num_epochs), disable=self.quiet, desc="Epochs           "):
            all_x = dataset.train_x
            all_y = dataset.train_y
            val_x = dataset.val_x
            val_y = dataset.val_y

            train_labels = [run[0] for run in all_x]
            train_inputs = [run[1] for run in all_x]
            val_inputs = [run[1] for run in val_x]

            if self.shuffle:
                has_attack = [np.any(y == 1) for y in all_y]
                attack_dirs = [1 if has else 0 for has in has_attack]
                shuff_indcs = shuffled_batch_indices(self.percent_runs, np.array(attack_dirs), train_labels)
                
                train_inputs = [train_inputs[i] for i in shuff_indcs]
                all_y = [all_y[i] for i in shuff_indcs]
                train_labels = [train_labels[i] for i in shuff_indcs]  

            self.epoch = i + 1
            t0 = t1
            if train_config['num_processes'] > 1:
                with Manager() as manager:
                    shared_all_x = manager.list(train_inputs)
                    shared_all_y = manager.list(all_y)

                    pool = Pool(train_config["num_processes"])
                    
                    bundles = [
                        (
                            self,
                            net.network,
                            proc,
                            proc_params,
                            shared_all_x,
                            shared_all_y,
                            0,
                            len(shared_all_x),
                        )
                        for net in pop.networks
                    ]
                    failed = False
                    try:
                        results = pool.map(mp_fitness, bundles)
                        fitnesses = []
                        thresholds = []
                        for result in results:
                            if "error" in result:
                                raise ValueError(result["error"])
                            fitnesses.append(result[self.fitness_type])
                            if "threshold" in result:
                                thresholds.append(result["threshold"])
                            else:
                                thresholds.append(0)
                    except Exception:
                        failed = True
                        print("Multiprocessing failed, see trace below:")
                        traceback.print_exc()
                    finally:
                        pool.close()
                        pool.join()
                    if failed:
                        exit()
            else:
                # Modified to store thresholds
                fitness_results = [
                    self.fitness(net.network, processor, train_inputs, all_y)
                    for net in tqdm(pop.networks, desc="EONS Population  ", disable=self.quiet, leave=False)
                ]
                
                fitnesses = [result[self.fitness_type] for result in fitness_results]
                thresholds = [result.get("threshold", 0) for result in fitness_results]

            max_fit = max(fitnesses)
            if max_fit > best_train:
                best_train = max_fit
                best_idx = np.argmax(fitnesses)
                self.overall_best_net = pop.networks[best_idx].network
                
                # Store the best threshold with the network
                best_threshold = thresholds[best_idx]
                self.overall_best_net.set_data("proc_params", proc_params)
                self.overall_best_net.set_data("app_params", self.config["app_params"])
                self.overall_best_net.set_data("encoder_array", self.encoder.as_json())
                if self.decoder is not None:
                    self.overall_best_net.set_data("decoder_array", self.decoder.as_json())
                self.overall_best_net.set_data("other", other)
                self.overall_best_net.set_data("spike_threshold", best_threshold)
                
                self.write_network(self.overall_best_net)

                # Process validation data to get raw decisions
                all_val_decisions = []
                all_val_labels = []
                
                for val_input, val_label_set in zip(val_inputs, val_y):
                    val_decisions = self.process_run(processor, self.overall_best_net, val_input)
                    all_val_decisions.extend(val_decisions)
                    all_val_labels.extend(val_label_set)
                
                # Calculate rolling sums and apply the best threshold
                val_rolling_sums = self.calculate_rolling_sums(all_val_decisions)
                val_windowed_decisions = self.apply_threshold(val_rolling_sums, best_threshold)
                
                # Calculate validation metrics
                f = self.calculate_metrics(val_windowed_decisions, all_val_labels)
                
                best_val = f[self.fitness_type]
                best_f = f

            t1 = time.time()

            msg = "Epoch: {:3d}  Time: {:6.1f}  Train: {}  Val: {}  Val TPR: {}  Val FPR: {} Val TNR: {} Val FNR {}  Threshold: {}  Num_Synapses: {}".format(
                i, t1 - t0, best_train, best_val, best_f['tpr'], best_f['fpr'], best_f['tnr'], best_f['fnr'], 
                best_threshold, self.overall_best_net.num_edges())
            
            print(msg)
            with open(self.log_file, 'a') as logfile:
                logfile.write(msg + '\n')
            pop = evolver.do_epoch(pop, fitnesses, train_config["eons_params"])

    def evaluate(self, dataset: IDSDataset, net, proc, proc_params, num_processes=1, verbose=False, output_file=None):
        """
        Evaluate a trained network/processor using hybrid metrics with rolling window approach
        
        :param dataset: dataset to use
        :param net: network
        :param proc: processor instantiation
        :param proc_params: process parameters dictionary
        :param num_processes: number of processors to split the evaluation over
        :param verbose: True to enable progress printouts, False otherwise
        :param output_file: Optional file path to save model decisions
        :return: fitness dictionary containing hybrid metrics
        """
        test_x = dataset.test_x
        test_y = dataset.test_y
        
        test_labels = [run[0] for run in test_x]
        test_inputs = [run[1] for run in test_x]
        
        t0 = time.time()
        
        # Always retrieve the threshold from the trained network
        threshold = net.get_data("spike_threshold") or 0
        
        # We'll only collect the windowed decisions
        all_windowed_decisions = []
        all_true_labels = []
        
        if num_processes > 1 and len(test_inputs) > 0:
            if verbose:
                print(f"Evaluating with {num_processes} processes...")
            
            # Split data into batches for multiprocessing
            batch_size = max(1, len(test_inputs) // num_processes)
            batches = []
            
            for i in range(0, len(test_inputs), batch_size):
                end_idx = min(i + batch_size, len(test_inputs))
                batches.append((
                    self,
                    net,
                    proc,
                    proc_params,
                    test_inputs[i:end_idx],
                    test_y[i:end_idx],
                    i,
                    threshold,
                    self.window_size
                ))
            
            # Process batches in parallel
            with Pool(num_processes) as pool:
                results = pool.map(mp_evaluate, batches)
            
            # Simply collect all decisions and labels from all batches
            for result in results:
                if "error" in result:
                    print(f"Error in batch starting at index {result['start_index']}: {result['error']}")
                    continue
                
                if self.window_size > 0:
                    all_windowed_decisions.extend(result["windowed_decisions"])
                else:
                    all_windowed_decisions.extend(result["decisions"])
                    
                all_true_labels.extend(result["true_labels"])
            
            # Calculate metrics once on the complete dataset
            fitness = self.calculate_metrics(all_windowed_decisions, all_true_labels)
            fitness["threshold"] = threshold
            fitness["windowed_decisions"] = all_windowed_decisions
            
        else:
            # Use single process evaluation (simplified)
            processor = proc(proc_params)
            all_raw_decisions = []
            
            # Process test data with the network to get raw packet-level decisions
            for i, (run_x, run_y) in enumerate(tqdm(zip(test_inputs, test_y), 
                                               total=len(test_inputs), 
                                               disable=not verbose,
                                               desc="Evaluating runs")):
                raw_decisions = self.process_run(processor, net, run_x)
                all_raw_decisions.extend(raw_decisions)
                all_true_labels.extend(run_y)
            
            # Calculate rolling sums and apply threshold
            rolling_sums = self.calculate_rolling_sums(all_raw_decisions)
            all_windowed_decisions = self.apply_threshold(rolling_sums, threshold)
            
            # Calculate metrics using the windowed decisions
            fitness = self.calculate_metrics(all_windowed_decisions, all_true_labels)
            fitness["windowed_decisions"] = all_windowed_decisions
            fitness["threshold"] = threshold
        
        
        return fitness