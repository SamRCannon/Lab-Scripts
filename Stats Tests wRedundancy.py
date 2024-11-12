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
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import api as sm
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd 
from statsmodels.stats.multitest import multipletests

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
all_group_data = {control_name: control_data}

# Collect data for other groups
for i in range(num_groups - 1):
    group_name = input(f"Enter the name of group {i + 1}: ")
    group_data = np.array([float(x) for x in input(f"Enter data for {group_name}, separated by commas: ").split(",")])
    all_group_data[group_name] = group_data

# Levene Test for each group vs. control
print("\nLevene test results (p-values):")
from scipy.stats import ttest_ind, levene
from statsmodels.stats.multitest import multipletests

# Prepare lists to store results for post-correction
all_p_values_student = []
all_p_values_welch = []
comparisons = []

for group_name, group_data in all_group_data.items():
    if group_name != control_name:
        # Levene test to check variance equality
        stat, p_value = levene(control_data, group_data)
        
        # Determine recommended test based on Levene's test
        x = "Welch's t-test" if p_value < 0.05 else "Student's t-test"
        
        # Determine test recommendation based on sample size difference
        largest_set = max(len(control_data), len(group_data))
        Set_Size_Diff = abs(len(control_data) - len(group_data)) / largest_set
        y = "Welch's t-test" if Set_Size_Diff >= 0.6 else "Student's t-test"
        
        # Perform both Student's and Welch's t-tests
        t_stat_student, p_value_student = ttest_ind(control_data, group_data, equal_var=True)
        t_stat_welch, p_value_welch = ttest_ind(control_data, group_data, equal_var=False)
        
        # Store the p-values for correction
        all_p_values_student.append(p_value_student)
        all_p_values_welch.append(p_value_welch)
        comparisons.append({"Control": control_name, "Group": group_name, "Levene p-value": p_value, "Recommended Test Levene": x, "Recommended Test Size Diff": y, "Student's t-test p-value": p_value_student, "Welch's t-test p-value": p_value_welch})

# Apply Bonferroni correction using `multipletests`
_, bonferroni_p_student, _, _ = multipletests(all_p_values_student, alpha=0.05, method='bonferroni')
_, bonferroni_p_welch, _, _ = multipletests(all_p_values_welch, alpha=0.05, method='bonferroni')

# Update comparisons with adjusted p-values
for i, comparison in enumerate(comparisons):
    comparison["Bonferroni Student's Adjusted p-value"] = bonferroni_p_student[i]
    comparison["Bonferroni Welch's Adjusted p-value"] = bonferroni_p_welch[i]

# Convert to DataFrame and save results to Excel
results_df = pd.DataFrame(comparisons)
current_date = datetime.now().strftime("%Y-%m-%d")
file_name = f"Levene_and_t_test_Results_{current_date}.xlsx"
results_df.to_excel(file_name, index=False)

print(f"\nLevene, t-test, and Bonferroni adjusted p-values saved to {file_name}.")

#Tukey
group_labels = []
data_values = []

for group_name, group_data in all_group_data.items():
    group_labels.extend([group_name] * len(group_data))
    data_values.extend(group_data)

# Perform Tukey HSD test
tukey_result = pairwise_tukeyhsd(endog=data_values, groups=group_labels, alpha=0.05)

# Convert Tukey HSD results to DataFrame
tukey_result_df = pd.DataFrame(
    data=tukey_result.summary().data[1:],  # Skip header row
    columns=tukey_result.summary().data[0]  # Use header row for column names
)

# Save Tukey HSD results to a separate Excel file
tukey_file_name = f"Tukey_HSD_Results_{current_date}.xlsx"
tukey_result_df.to_excel(tukey_file_name, index=False)
print(f"Tukey HSD results saved to {tukey_file_name}.")

# Collect data for Dunnett's test
group_labels = []
data_values = []

# Prepare the data and labels for Dunnett's test
for group_name, group_data in all_group_data.items():
    group_labels.extend([group_name] * len(group_data))
    data_values.extend(group_data)

# Prepare the data for MultiComparison
mc = MultiComparison(data_values, group_labels)

# Perform Dunnett's test (comparing each group to the control group)
dunnett_result = mc.tukeyhsd(alpha=0.05)

# Save Dunnett's results to a DataFrame
dunnett_result_df = pd.DataFrame(
    data=dunnett_result.summary().data[1:],  # Skip header row
    columns=dunnett_result.summary().data[0]  # Use header row for column names
)

# Save the results to Excel
dunnett_file_name = f"Dunnett_Test_Results_{current_date}.xlsx"
dunnett_result_df.to_excel(dunnett_file_name, index=False)

print(f"Dunnett test results saved to {dunnett_file_name}.")