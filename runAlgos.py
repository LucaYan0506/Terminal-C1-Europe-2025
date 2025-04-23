import subprocess
import os

# Path to the parent "algo" folder
algo_folder = "algos"

# Get list of subdirectories (i.e., bots) inside "algo"
bots = [name for name in os.listdir(algo_folder)
        if os.path.isdir(os.path.join(algo_folder, name))]

# Path to the run_match script
run_script = r".\scripts\run_match.py"

# Run all unique pairwise matches
for i in range(len(bots)):
        bot1 = os.path.join(".", "iter7")
        bot2 = os.path.join(".", algo_folder, bots[i])
        print(f"Running match: iter7 vs {bots[i]}")
        subprocess.run(["python", run_script, bot1, bot2])
