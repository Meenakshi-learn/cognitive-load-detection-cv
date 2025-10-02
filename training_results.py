import json
import argparse
import pandas as pd
import pprint

def display_results(results, verbose=False):
    """
    Displays model training results in either a detailed or a tabular format.
    
    Args:
        results (dict): The dictionary of results loaded from the JSON file.
        verbose (bool): If True, displays detailed epoch-by-epoch results.
                        If False, displays a concise summary table.
    """
    if verbose:
        print("\n--- Detailed Model Training Results ---")
        pprint.pprint(results, indent=2)
    else:
        # Prepare data for the tabular summary
        table_data = []
        for model_name, data in results.items():
            if 'history' in data and data['history']:
                final_loss = data['history']['loss'][-1]
            else:
                final_loss = 'N/A'
            
            table_data.append({
                'Model': model_name,
                'Validation Accuracy': f"{data['accuracy']:.4f}",
                'Final Loss': f"{final_loss:.4f}"
            })
        
        # Create a DataFrame and display it
        df = pd.DataFrame(table_data)
        print("\n--- Model Training Results Summary ---")
        print(df.to_string())

def main():
    parser = argparse.ArgumentParser(description="View model training results from a JSON file.")
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Display detailed, epoch-by-epoch results instead of a summary table."
    )
    args = parser.parse_args()

    try:
        with open('training_results.json', 'r') as f:
            results = json.load(f)
        
        display_results(results, verbose=args.verbose)
        
    except FileNotFoundError:
        print("Error: The 'training_results.json' file was not found. Please run the model_training.py script first.")
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON format.")

if __name__ == "__main__":
    main()