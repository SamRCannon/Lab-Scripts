import sys
import subprocess
def installx(package):
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)
installx('scipy')
installx('numpy')
installx('time')
installx('pandas')
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd


print(".")
time.sleep(.5)
print(".")
time.sleep(.5)
print(".")
time.sleep(.5)
print("Before beginning, it's recommended to convert data used into a comma seperated list in a text file for ease of use.")
time.sleep(1)
print(".")
time.sleep(.5)
print(".")
time.sleep(.5)
print(".")
print("If you make any mistakes hit CTRL+C to interrupt the program and restart.")
time.sleep(1)
print(".")
time.sleep(.5)
print(".")
time.sleep(.5)
print(".")
time.sleep(1.5)

# Number of groups prompt
num_groups = int(input("Enter the number of groups to test (including control): "))

#Control info prompt
control_name = input("Enter the name of the control group: ")
control_data = np.array([float(x) for x in input(f"Enter data for {control_name}, separated by commas: ").split(",")])

#Initiate Dictionary
groups_data = {control_name: control_data}

# Collect data for other groups
for i in range(num_groups - 1):
    group_name = input(f"Enter the name of group {i + 1}: ")
    group_data = np.array([float(x) for x in input(f"Enter data for {group_name}, separated by commas: ").split(",")])
    groups_data[group_name] = group_data

# Levene Test for each group vs. control
print("\nLevene test results (p-values):")
results = []
for group_name, group_data in groups_data.items():
    if group_name != control_name:
        stat, p_value = stats.levene(control_data, group_data)
        if p_value < 0.05:
            x = "Welch's t-test"
        else:
            x = "Student's t-test"
        print(f"{control_name} vs {group_name}: p-value = {p_value:.4f}: Recommended Test = {x}")
        results.append({"Control": control_name, "Group": group_name, "p-value": p_value, "Recommended Test": x})

# Save to Excel
results_df = pd.DataFrame(results)
current_date = datetime.now().strftime("%Y-%m-%d")
file_name = f"Levene Test Results {current_date}.xlsx"
results_df.to_excel(file_name, index=False)

print(f"\nLevene test results saved to {file_name}.")

