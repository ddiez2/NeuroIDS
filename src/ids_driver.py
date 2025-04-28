import os
import json
import numpy as np
import matplotlib.pyplot as plt

from py_apps.utils.classify_app import *
from py_apps.utils.common_utils import read_network
from py_apps.utils.common_utils import load_json_arg
from py_apps.utils.neuro_help import add_proc_arguments, add_class_arguments, add_coder_arguments, add_printing_arguments, get_proc_instantiation, setup_config

from neuroids import NeuroIDS
from idsdb import IDSDataset
def main():
    import argparse
    parser = argparse.ArgumentParser(description="NeuroRad Application Driver")
    parser.add_argument("--activity", "-a", required=True, type=str, choices=["train", "evaluate"], help="activity to perform")
    parser.add_argument("--network_filename", default=None, type=str, help="location to store the best network file produced if training or network to load if testing")
    parser.add_argument("--sim_time", default=50, type=float, help="the simulation time for each data instance")
    parser.add_argument("--extra_eons_params", default="{}", type=str, help="JSON file or JSON string updating EONS parameters from configuration file")
    parser.add_argument("--epochs", default=1000, type=int, help="epochs for eons")
    parser.add_argument("--max_fitness", default=1e+06, type=float, help="max fitness for eons")
    parser.add_argument("--processes", "-p", default=1, type=int, help="processes for EONS")
    parser.add_argument("--test_seed", default=1234, type=int, help="testing seed")
    # Beginning of modified params
    parser.add_argument("--fitness", "-f", default="j", type=str, choices=["mcc", "j", "roc_auc"])
    parser.add_argument("--eons_params", default=os.path.join(os.path.dirname(__file__), "config/eons.json"), type=str, help="JSON file with EONS parameters")
    parser.add_argument("--app_name", default="NeuroIDS", type=str)
    # Beginning of custom params
    parser.add_argument("--dataset_path", "-d", type=str, default=None, help="path to directory contaning Numpy files")
    parser.add_argument("--shuffle", action="store_true", help="shuffle data")
    parser.add_argument("--percent_eval", type=float, default=1.0, help="percent of  runs to use for evaluation")
    parser.add_argument("--quiet", action='store_true', help="reduce output by not showing progress")
    parser.add_argument("--temporal", action="store_true", help="Enables temporal network.")
    parser.add_argument("--network_dir", type=str, default=None, help="Directory to store network in, only used with random_netfilename")
    parser.add_argument("--random_netfilename", action="store_true", help="Network filename will be generated randomly")
    parser.add_argument("--save_eval", type=str, default=None, help="Path to directory where to save the evaluation data (only used in evaluate activity)")


    # Proc params
    add_proc_arguments(parser)

    # Encoder information
    add_coder_arguments(parser)

    # Printing params
    add_printing_arguments(parser)

    args = parser.parse_args()
    args.proc_name = "caspian"

    if args.random_netfilename:
        if args.network_dir is None:
            raise ValueError("If random_netfilename is True, then network_dir must be provided.")
    # Read in proc params as necessary
    proc_params = load_json_arg(args.proc_params)

    # Read in EONS params
    eons_params = load_json_arg(args.eons_params)

    # Read in extra EONS params
    extra_eons_params = load_json_arg(args.extra_eons_params)

    # activity we are doing
    activity = args.activity

    for k in extra_eons_params.keys():
        eons_params[k] = extra_eons_params[k]

    proc_instantiation = get_proc_instantiation(args.proc_name)
    config = setup_config(args)

    if activity == "train" and args.random_netfilename:
        import random
        import string
        config["network_filename"] = os.path.join(args.network_dir, ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + '_network.txt')

    # setup the network
    if activity != "train":
        config["runtime"] = args.sim_time
        config["seed"] = 0
        config["encoder_interval"] = args.encoder_interval
        config["num_processes"] = 0

        net = read_network(config["network_filename"])

        data = str(net.get_data("other"))
        # Replace single quotes with double quotes to conform to JSON format
        data = data.replace("'", '"')
        # Replace boolean values accordingly
        data = data.replace("True", "true").replace("False", "false").replace("None", "null")
        other = json.loads(data)

        proc_params = net.get_data("proc_params")

        config["runtime"] = other["sim_time"]

        config["encoder_array"] = net.get_data("encoder_array")
        config["decoder_array"] = net.get_data("decoder_array")

        temporal = other["temporal"]

    # Load the dataset
    dataset = IDSDataset(train_x_path=os.path.join(args.dataset_path, 'train_x.pkl'),
                           train_y_path=os.path.join(args.dataset_path, 'train_y.pkl'),
                           val_x_path=os.path.join(args.dataset_path, 'val_x.pkl'),
                           val_y_path=os.path.join(args.dataset_path, 'val_y.pkl'),
                           test_x_path=os.path.join(args.dataset_path, 'test_x.pkl'),
                           test_y_path=os.path.join(args.dataset_path, 'test_y.pkl'),
                           debug=True)
    if args.activity == "train":
        import time
        st = time.time()
        temporal = args.temporal

        # Instantiate a NeuroRad class
        app = NeuroIDS(config,
                         dataset.get_dminmax(),
                         percent_runs=args.percent_eval,
                         shuffle=args.shuffle,
                         temporal=args.temporal,
                         quiet=args.quiet)

        # Setup training parameters
        train_params = {}
        train_params["eons_params"] = eons_params
        train_params["num_epochs"] = args.epochs
        train_params["num_processes"] = args.processes
        train_params["fitness"] = args.fitness

        # Train the model
        train_start = time.time()
        app.train(dataset, train_params, proc_instantiation, proc_params)
        train_end = time.time()
        elapsed = train_end - train_start
        net = app.overall_best_net
        net.prune()
        print("Network size:", net.num_nodes(), net.num_edges())
        print("Training Time:", elapsed)
        et = time.time()
        duration = (et - st) / 3600.0
        print(f"Performed {args.epochs} epochs in {duration} hours for a rate of {args.epochs / duration} epochs per hour.")

    
    if args.activity == "evaluate":
        # Instantiate a NeuroRad class
        app = NeuroIDS(config,
                         dataset.get_dminmax(),
                         percent_runs=args.percent_eval,
                         shuffle=args.shuffle,
                         temporal=temporal,
                         quiet=args.quiet)

        results = app.evaluate(dataset, net, proc_instantiation, proc_params, args.processes, verbose=not args.quiet)

        print("Results on Test Set")
        print(f"\tP:   {results['p']}")
        print(f"\tN:   {results['n']}")
        print(f"\tTP:  {results['tp']}")
        print(f"\tFP:  {results['fp']}")
        print(f"\tTN:  {results['tn']}")
        print(f"\tFN:  {results['fn']}")
        print(f"\tJ: {results['j']}")
        print(f"\tMCC: {results['mcc']}")
        print(f"\tTPR: {results['tpr']}")
        print(f"\tFPR: {results['fpr']}")
        print(f"\tTNR: {results['tnr']}")
        print(f"\tFNR: {results['fnr']}")
        print(f"\tPPV: {results['ppv']}")
        print(f"\tNPV: {results['npv']}")
        
        # Print threshold information
        stored_threshold = net.get_data("spike_threshold")
        if stored_threshold is not None:
            print(f"\tThreshold from training: {stored_threshold}")
            
        print(f"\tUsed threshold: {results['threshold']}")
        
        if "optimal_test_threshold" in results:
            print(f"\tOptimal threshold for test set: {results['optimal_test_threshold']}")
            print(f"\tMCC with optimal test threshold: {results['optimal_test_mcc']}")
        
        if args.save_eval is not None:
            # Extract just the base filename without path or extension
            base_filename = os.path.splitext(os.path.basename(config["network_filename"]))[0]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.save_eval) if os.path.dirname(args.save_eval) else '.', exist_ok=True)
            
            # If save_eval is a directory, create a file inside it
            if os.path.isdir(args.save_eval) or args.save_eval.endswith('/'):
                # Ensure directory exists
                os.makedirs(args.save_eval, exist_ok=True)
                output_path = os.path.join(args.save_eval, f"{base_filename}_eval.json")
            else:
                # Ensure parent directory exists
                parent_dir = os.path.dirname(args.save_eval)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                # Use the provided path directly
                output_path = args.save_eval
                
                # If it doesn't end with .json, add the extension
                if not output_path.endswith('.json'):
                    output_path += '.json'
            
            # Write results to file
            try:
                with open(output_path, 'w') as sf:
                    # Create a filtered copy of the results without large arrays
                    filtered_results = {k: v for k, v in results.items() if k not in ['true_labels', 'all_spikes']}
                    
                    # Convert results to a fully serializable format
                    serializable_results = convert_to_serializable(filtered_results)
                    json.dump(serializable_results, sf)
                print(f"Evaluation results saved to: {output_path}")
            except Exception as e:
                print(f"Error saving evaluation results: {e}")
                print(f"Attempted to save to: {output_path}")

def convert_to_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable objects to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)  # Convert numpy integers to Python int
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)  # Convert numpy floats to Python float
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively process dicts
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]  # Recursively process lists
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(v) for v in obj)  # Recursively process tuples
    else:
        return obj  # Return other types as-is

if __name__ == '__main__':
    main()
