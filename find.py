import pickle
import sys
from pathlib import Path

def show_best_program(log_dir):
    # Find the most recent backup file in the directory
    path = Path(log_dir)
    backups = list(path.glob("**/*.pickle"))
    
    if not backups:
        print("No results found yet. Let the AI run for a few more minutes!")
        return

    # Get the latest file
    latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
    print(f"Reading from: {latest_backup}\n" + "-"*40)

    try:
        with open(latest_backup, "rb") as f:
            data = pickle.load(f)
            
        # FunSearch saves data as a dictionary of 'islands'
        # We want to find the program with the highest score across all islands
        best_program = None
        best_score = float('-inf')

        # Navigate the internal structure (it varies slightly by version, but usually:)
        # data is often a dict of {island_id: {'programs': {...}}}
        
        # This is a generalized finder for the standard FunSearch pickle structure
        programs_found = 0
        
        # Iterate through whatever structure is in the pickle to find 'programs'
        if hasattr(data, 'values'):
            # It's likely a dictionary of islands
            for island in data.values():
                if isinstance(island, dict) and 'programs' in island:
                    for signature, program_entry in island['programs'].items():
                        programs_found += 1
                        # The entry usually has 'score' and 'code'
                        if program_entry.score > best_score:
                            best_score = program_entry.score
                            best_program = program_entry.code
                            
        if best_program:
            print(f"TOP SCORE: {best_score}")
            print("\n--- BEST ALGORITHM FOUND ---\n")
            print(best_program)
            print("\n----------------------------")
        else:
            print(f"Found {programs_found} programs, but none had a valid score yet.")
            
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        print("Try letting the search run longer first.")

if __name__ == "__main__":
    # Usage: python inspect_results.py ./data_lattice
    if len(sys.argv) < 2:
        print("Usage: python inspect_results.py <path_to_data_folder>")
    else:
        show_best_program(sys.argv[1])