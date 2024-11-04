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
installx('pandas')
installx('warnings')
installx('glob')
installx('openpyxl')
installx('matplotlib')
installx('numpy')
installx('sympy')
installx('statsmodels')
installx('os')
installx('scipy')
installx('shutil')
installx('random')
installx('datetime')

import pandas as pd
import warnings  
import glob
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline 

#opt out of future updates
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings("ignore")

#separates even # plates
def evens(df, column_name):
    numeric_column = pd.to_numeric(df[column_name], errors='coerce')
    even_mask = numeric_column % 2 == 0
    filtered_df = df[even_mask]
    return filtered_df
#separates odd # plates
def odds(df, column_name):
    numeric_column = pd.to_numeric(df[column_name], errors='coerce')
    odd_mask = numeric_column % 2 == 1
    filtered_odd_df = df[odd_mask]
    return filtered_odd_df
def singles(df, column_name):
    numeric_column = pd.to_numeric(df[column_name], errors='coerce')
    singles_mask = numeric_column <= addition_num
    filtered_singles_df = df[singles_mask]
    return filtered_singles_df
#rhythmicity testing
def analyze_rhythmicity(df, title):
    rhythmic_replicates = []
    for i, row in df.iterrows():
        data_row = row.iloc[2:].dropna().values
        if len(data_row) < 24:
            continue

        time_index = pd.date_range(start='2024-01-01', periods=len(data_row), freq='h')
        series = pd.Series(data_row, index=time_index)

        decomposition = seasonal_decompose(series, model='additive', period=24)
        seasonal_strength = np.mean(np.abs(decomposition.seasonal.dropna()))
        is_rhythmic = seasonal_strength > threshold

        rhythmic_replicates.append({
            'Row': i,
            'Addition Time': row.iloc[0],
            'Treatment Group': row.iloc[1],
            'Rhythmic': is_rhythmic,
            'Seasonal Strength': seasonal_strength
        })
    
    return pd.DataFrame(rhythmic_replicates)
#mask out arrhythmic reps    
def rythmask(rhythmic_df, column_name, data_df):
    rhythmic_rows = rhythmic_df[rhythmic_df[column_name] == True].index.tolist()
    filtered_data_df = data_df.iloc[rhythmic_rows, :]   
    return filtered_data_df
#mean for groups
def average_by_labels_exnan(df, label_col1, label_col2):
    df['combined_labels'] = df[label_col1].astype(str) + '_' + df[label_col2].astype(str)    
    label_columns = [label_col1, label_col2, 'combined_labels']
    numeric_columns = df.columns.difference(label_columns)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')    
    group_sizes = df.groupby('combined_labels').size()
    valid_groups = group_sizes[group_sizes >= 3].index
    df_filtered = df[df['combined_labels'].isin(valid_groups)]
    non_nan_columns = df_filtered[numeric_columns].dropna(axis=1, how='all').columns
    if not non_nan_columns.empty:
        err_df = df_filtered.groupby('combined_labels')[non_nan_columns].mean().reset_index()
    else:
        err_df = pd.DataFrame(columns=['combined_labels'])
    
    return err_df 
#sem for groups
def err_by_labels_exnan(df, label_col1, label_col2):
    df['combined_labels'] = df[label_col1].astype(str) + '_' + df[label_col2].astype(str)
    label_columns = [label_col1, label_col2, 'combined_labels']
    
    numeric_columns = df.columns.difference(label_columns)    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    group_sizes = df.groupby('combined_labels').size()
    valid_groups = group_sizes[group_sizes >= 3].index
    
    df_filtered = df[df['combined_labels'].isin(valid_groups)]    
        
    non_nan_columns = df_filtered[numeric_columns].dropna(axis=1, how='all').columns
    if not non_nan_columns.empty:
        err_df = df_filtered.groupby('combined_labels')[non_nan_columns].sem().reset_index()
    else:
        err_df = pd.DataFrame(columns=['combined_labels'])
    
    return err_df
#averege (collects CTRL and makes a global average and averages all treatment groups by time addition) 
#Using Smart Range for Evaluating Extrema
def smart_range_extrema(numeric_data_range, range1mean, range1amp, offset, subrange_thresh):
    high_ranges = []
    low_ranges = []
    current_high_idx = []
    current_low_idx = []
    for i, x in enumerate(numeric_data_range):
        original_index = offset + i
        if x >= (range1mean + subrange_thresh * range1amp):
            if not current_high_idx:
                current_high_idx = [(original_index, x)]
            else:
                current_high_idx.append((original_index, x))
        elif current_high_idx:
            max_idx = max(current_high_idx, key=lambda item: item[1])[0]
            high_ranges.append(max_idx)
            current_high_idx = []

        if x <= (range1mean - subrange_thresh * range1amp):
            if not current_low_idx:
                current_low_idx = [(original_index, x)]
            else:
                current_low_idx.append((original_index, x))
        elif current_low_idx:
            min_idx = min(current_low_idx, key=lambda item: item[1])[0]
            low_ranges.append(min_idx)
            current_low_idx = []

    if current_high_idx:
        max_idx = max(current_high_idx, key=lambda item: item[1])[0]
        high_ranges.append(max_idx)

    if current_low_idx:
        min_idx = min(current_low_idx, key=lambda item: item[1])[0]
        low_ranges.append(min_idx)
    
    if len(high_ranges) > 1 and abs(high_ranges[1] - high_ranges[0]) <= 3:
        high_ranges = [high_ranges[0]]

    if len(low_ranges) > 1 and abs(low_ranges[1] - low_ranges[0]) <= 3:
        low_ranges = [low_ranges[0]]
        
    num_extrema1 = min(1, len(high_ranges))
    num_extrema2 = min(1, len(low_ranges))
        
    high_ranges = [h * interval for h in high_ranges[:num_extrema1]]
    low_ranges = [l * interval for l in low_ranges[:num_extrema2]]

    return low_ranges[:num_extrema2], high_ranges[:num_extrema1]
#averege by group
def combined_average_new(df, label_col1, label_col2):
    df['combined_labels'] = df[label_col1].astype(str) + '_' + df[label_col2].astype(str)

    cols = df.columns.tolist()
    combined_label_position = 2
    cols.insert(combined_label_position, cols.pop(cols.index('combined_labels')))
    df = df[cols]

    label_columns = [label_col1, label_col2, 'combined_labels']

    numeric_columns = df.columns.difference(label_columns)

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    group_df = df.copy()
    
    group_sizes = group_df.groupby('combined_labels').size()
    valid_groups = group_sizes[group_sizes >= 3].index
    df_filtered = group_df[group_df['combined_labels'].isin(valid_groups)]
    non_nan_columns = df_filtered[numeric_columns].dropna(axis=1, how='all').columns
    if not non_nan_columns.empty:
        final_df = df_filtered.groupby('combined_labels')[non_nan_columns].mean().reset_index()
    else:
        final_df = pd.DataFrame(columns=['combined_labels'])
    mean_row = final_df[non_nan_columns].mean().to_frame().T
    mean_row['combined_labels'] = 'Overall_Mean'
    final_df_with_mean = pd.concat([final_df, mean_row], ignore_index=True)
    
    return final_df_with_mean    
#sem by group
def combined_sem_new(df, label_col1, label_col2):
    df['combined_labels'] = df[label_col1].astype(str) + '_' + df[label_col2].astype(str)

    cols = df.columns.tolist()
    combined_label_position = 2
    cols.insert(combined_label_position, cols.pop(cols.index('combined_labels')))
    df = df[cols]

    label_columns = [label_col1, label_col2, 'combined_labels']

    numeric_columns = df.columns.difference(label_columns)

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    group_df = df.copy()

    group_sizes = group_df.groupby('combined_labels').size()
    valid_groups = group_sizes[group_sizes >= 3].index
    df_filtered = group_df[group_df['combined_labels'].isin(valid_groups)]
    
    non_nan_columns = df_filtered[numeric_columns].dropna(axis=1, how='all').columns
    if not non_nan_columns.empty:
        final_df = df_filtered.groupby('combined_labels')[non_nan_columns].sem().reset_index()
    else:
        final_df = pd.DataFrame(columns=['combined_labels'])

    overall_sem_row = df_filtered[non_nan_columns].sem().to_frame().T
    overall_sem_row['combined_labels'] = 'Overall_SEM'

    final_df_with_sem = pd.concat([final_df, overall_sem_row], ignore_index=True)
    
    return final_df_with_sem
def combined_sd_new(df, label_col1, label_col2):
    df['combined_labels'] = df[label_col1].astype(str) + '_' + df[label_col2].astype(str)

    cols = df.columns.tolist()
    combined_label_position = 2
    cols.insert(combined_label_position, cols.pop(cols.index('combined_labels')))
    df = df[cols]

    label_columns = [label_col1, label_col2, 'combined_labels']

    numeric_columns = df.columns.difference(label_columns)

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    group_df = df.copy()

    group_sizes = group_df.groupby('combined_labels').size()
    valid_groups = group_sizes[group_sizes >= 3].index
    df_filtered = group_df[group_df['combined_labels'].isin(valid_groups)]
    
    non_nan_columns = df_filtered[numeric_columns].dropna(axis=1, how='all').columns
    if not non_nan_columns.empty:
        final_df = df_filtered.groupby('combined_labels')[non_nan_columns].std().reset_index()
    else:
        final_df = pd.DataFrame(columns=['combined_labels'])

    overall_sem_row = df_filtered[non_nan_columns].std().to_frame().T
    overall_sem_row['combined_labels'] = 'Overall_SD'

    final_df_with_sem = pd.concat([final_df, overall_sem_row], ignore_index=True)
    
    return final_df_with_sem
def replace_addition_times_new(df, column_name, time_dict):
    def replace_time(value):
        if 'Overall' in str(value):
            return value
        if pd.isna(value):
            return value
        parts = value.split('_')
        if len(parts) != 2:
            return value
        addition_time, treatment_group = parts
        replacement_time = time_dict.get(float(addition_time), addition_time)
        return f"{replacement_time}_{treatment_group}"

    df[column_name] = df[column_name].apply(replace_time)
    return df

#READS THE FEEDER FILE

file_list = glob.glob('PRC_Feed_Doc.xlsx')
for file in file_list:
 df1 = pd.read_excel(file)
 dfbug = df1.copy()  
 dfbug.replace({False: 0}, inplace=True)
 labels = df1.iloc[2,0:12].tolist()
 cleaned_labels = [label for label in labels if not pd.isna(label)]
 labels_df = pd.DataFrame(cleaned_labels)
 addition_num = df1.iloc[5,1]
 input_name = df1.iloc[5,5]
 graphing = df1.iloc[5,8]
 graphing1 = df1.iloc[8, 8]
 threshold = df1.iloc[11,5]
 #now a numbered list of labels is made for the sorting process later
 #Below will report and store information from the feeder file
 #if df1.iloc[5,1] == True:
 #   multiplate = True
 #else:
 #   multiplate = False
 if df1.iloc[8,5] == True:
    Mmode = True
 else:
    Mmode = False

#FILE REPORT    

print("FILE REPORT:")    
print("Mammalian Mode", Mmode)
print(labels_df)
interval = df1.iloc[8,1]
print("Interval:", interval, "Hours")
addition_time = pd.DataFrame(dfbug.iloc[12:19,1].tolist())
addition_time = pd.DataFrame(dfbug.iloc[12:(12+addition_num),1].tolist())
print("Number of Additions", addition_num)
print("Addition Times:", addition_time)
print("Threshold for Rhythmicity:", threshold)
print("Input File Label:", input_name)
print("\n"*1)

#ORGANIZING EVEN ODD PLATES AND FIRST ROUND OF LABELLING
 
input_file_list = glob.glob(input_name)
for file in input_file_list:
    df2 = pd.read_excel(file)
    first_col_name = df2.columns[0]
    filtered_singles = singles(df2, first_col_name) 
   
    #renames columns with replicate numbers
    new_column_names = ['plate', 'well'] + list(range(1, 200))
    filtered_singles.columns = new_column_names[:len(filtered_singles.columns)]   
  
    addition_times3 = {}
    for i in range(1, addition_num + 1):
        addition_times3[str(i)] = addition_time.iloc[i-1, 0] 
        
#LABELLING

column_index = 0   
if column_index < len(filtered_singles.columns):
        filtered_singles.iloc[:, column_index] = filtered_singles.iloc[:, column_index].astype(str)
        filtered_singles.iloc[:, column_index] = filtered_singles.iloc[:, column_index].map(addition_times3).fillna(filtered_singles.iloc[:, column_index])         
replacement_labels_singles = [
    labels_df.iloc[0,0], labels_df.iloc[1,0], labels_df.iloc[2,0], 
    labels_df.iloc[3,0], labels_df.iloc[4,0], labels_df.iloc[5,0], 
    labels_df.iloc[6,0], labels_df.iloc[7,0], labels_df.iloc[8,0], 
    labels_df.iloc[9,0], labels_df.iloc[10,0], labels_df.iloc[11,0]
]

#2ND ROUND OF LABELLING

well_labels = [f'{chr(65 + i//12)}{str((i % 12) + 1).zfill(2)}' for i in range(96)]
label_map_singles = {well: replacement_labels_singles[i % 12] for i, well in enumerate(well_labels)}
filtered_singles.iloc[:, 1] = filtered_singles.iloc[:, 1].map(label_map_singles).fillna(filtered_singles.iloc[:, 1])
filtered_singles = filtered_singles.sort_values(by=[filtered_singles.columns[0], filtered_singles.columns[1]], ascending=[True, True])

# ANALYZE RHYTHMICITY FOR FILTERED EVENS AND ODDS

print("TESTING RHYTHMICITY...")
rhythmic_singles_df = analyze_rhythmicity(filtered_singles, "Filtered Singles")
filtered_singles=(rythmask(rhythmic_singles_df, "Rhythmic", filtered_singles))  
averaged_singles_df = average_by_labels_exnan(filtered_singles, "plate" , "well")
averaged_singleserr_df = err_by_labels_exnan(filtered_singles, "plate" , "well")

#COLLECT DATA AND MOVE TO FILE (Report_Avg&SEM.xlsx)

together_at_last3 = [averaged_singles_df, averaged_singleserr_df]
all_singles =pd.concat(together_at_last3)

#GRAPH MAKING

with pd.ExcelWriter('1. Single_Count_Avg&SEM.xlsx') as writer:
        all_singles.to_excel(writer, index=False) 
        
singles_x_values = averaged_singles_df.columns[1:] * interval
plt.figure(figsize=(14, 8))
for index, row in averaged_singles_df.iterrows():
    plt.plot(singles_x_values, row[1:], label=row['combined_labels'])
    
for index, row in averaged_singles_df.iterrows():
    sem_row = averaged_singleserr_df.loc[averaged_singleserr_df['combined_labels'] == row['combined_labels']].iloc[0, 1:]
    plt.errorbar(singles_x_values, row[1:], yerr=sem_row, fmt='none', capsize=2, ecolor='k', elinewidth=1, alpha=.5)

plt.title('Timecourse')
plt.xlabel('Time (Hrs)')
plt.ylabel('Count')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid(True)
plt.savefig('2. averaged_plot.jpg', format='jpg', dpi=300, bbox_inches='tight')  
plt.close()

output_folder = '2. averaged_plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def get_label_prefix(label):
    return label.split('_')[0]
grouped = averaged_singles_df.groupby(averaged_singles_df['combined_labels'].apply(get_label_prefix))
for label_prefix, group_df in grouped:
    plt.figure(figsize=(14, 8))
    
    for index, row in group_df.iterrows():
        singles_x_values = averaged_singles_df.columns[1:] * interval
        def random_color():
            return (np.random.uniform(.25, 0.75),
                    np.random.uniform(.25, 0.75),
                    np.random.uniform(.25, 0.75))                    
        color0 = random_color()
        
        plt.plot(singles_x_values, row[1:], color=color0, label=row['combined_labels'])

        sem_row = averaged_singleserr_df.loc[averaged_singleserr_df['combined_labels'] == row['combined_labels']].iloc[0, 1:]
        plt.errorbar(singles_x_values, row[1:], yerr=sem_row, fmt='none', capsize=2, ecolor=color0, elinewidth=1, alpha=1)

    plt.title(f'Timecourse for {label_prefix}')
    plt.xlabel('Time (Hrs)')
    plt.ylabel('Count')
    x_ticks_interval = 10
    plt.xticks(np.arange(0, max(singles_x_values) + 1, x_ticks_interval))
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.grid(True)
    
    plt.savefig(os.path.join(output_folder, f'singles_plate_plot_{label_prefix}.jpg'), format='jpg', dpi=300, bbox_inches='tight')
    plt.close()   

#FITTING REPLICATES TO FUNCTION FOR PRELIMINARY EXTREMA AND PERIOD -> 
  
time = np.linspace(0, len(filtered_singles), len(filtered_singles)+1)
filtered_singles1 = pd.DataFrame(filtered_singles).reset_index(drop=True)
df = filtered_singles1
colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_singles1)))
t = sp.symbols('t')
start_hour = 10
end_hour = 85
start_index = np.searchsorted(time, start_hour)
end_index = np.searchsorted(time, end_hour)

readable_start = 0
readable_end = len(time)
extrema = {}
max_len = 0 
failed_indices = []

for idx, row_data in filtered_singles1.iterrows():
    if (((idx/df.shape[0]) * 100) % 10) == 0:
        print(f"Processing replicates: {round((idx/df.shape[0]) * 100)} % complete")    

    if idx >= df.shape[0]:
        print(f"Index {idx} is out of bounds for DataFrame with size {df.shape}")
        failed_indices.append(idx)
        continue
    
    row_data_numeric = pd.to_numeric(row_data, errors='coerce')
    time_window = time[start_index:end_index]
    row_data_window = row_data_numeric[start_index:end_index]
    valid_indices = pd.notna(row_data_window) & (row_data_window != 0)
    if not valid_indices.any():
        print(f"No valid data points in the specified time window for replicate {idx}. Skipping.")
        failed_indices.append(idx)
        continue
    if readable_end > len(row_data_window):
        readable_end = len(row_data_window)
    if readable_start >= readable_end:
        raise ValueError("Readable start must be less than readable end.")
    time_valid = time_window[readable_start:readable_end]
    row_data_valid = row_data_window[readable_start:readable_end]
    row_data_smoothed = savgol_filter(row_data_valid, window_length=12, polyorder=10)
    try:
        poly_coeffs = np.polyfit(time_valid, row_data_smoothed, deg=10)
        poly_expr = sum(c * t**i for i, c in enumerate(reversed(poly_coeffs)))
        poly_deriv = sp.diff(poly_expr, t)
        
        extrema_times = sp.solveset(poly_deriv, t, domain=sp.S.Reals)
        if isinstance(extrema_times, sp.FiniteSet):
            extrema_real = [float(sol.evalf()) for sol in extrema_times if sol.is_real]
        else:
            print(f"Non-iterable extrema found for replicate {idx}. Skipping.")
            failed_indices.append(idx)
            continue
        
        extrema_real = [float(sol.evalf()) for sol in extrema_times if sol.is_real]
        extrema_real = [et for et in extrema_real if time_valid[0] <= et <= time_valid[-1]]
        if len(extrema_real) == 0:
            print(f"No extrema found for replicate {idx}. Skipping.")
            failed_indices.append(idx)
            continue
        extrema[idx] = extrema_real
        if len(extrema_real) > max_len:
            max_len = len(extrema_real)
    except np.RankWarning:
        print(f"Warning: Polynomial fitting may be poorly conditioned for replicate {idx}. Skipping.")
        failed_indices.append(idx)
    except Exception as e:
        print(f"Error processing replicate {idx}: {e}")
        failed_indices.append(idx)
extrema_df = pd.DataFrame(columns=['addition time', 'treatment', *[f'extrema_{i}' for i in range(max_len)]], index=filtered_singles1.index)
for idx, extrema_list in extrema.items():
    if idx >= df.shape[0]: 
        continue
    extrema_df.at[idx, 'addition time'] = df.iloc[idx, 0]
    extrema_df.at[idx, 'treatment'] = df.iloc[idx, 1]

    for i, et in enumerate(extrema_list):
        extrema_df.at[idx, f'extrema_{i}'] = et * interval           
plt.figure(figsize=(10, 6))
time = np.linspace(0, (100*interval), 101)        
for idx, row_data in filtered_singles1.iterrows():
    row_data_numeric = pd.to_numeric(row_data, errors='coerce')
    row_data_window = row_data_numeric[start_index:end_index]
    valid_indices = pd.notna(row_data_window) & (row_data_window != 0)
    
    if not valid_indices.any():
        print(f"No valid data points in the specified time window for replicate {idx}. Skipping.")
        continue

    time_valid = time[start_index:end_index][readable_start:readable_end]
    row_data_valid = row_data_window[readable_start:readable_end]
    row_data_smoothed = savgol_filter(row_data_valid, window_length=20, polyorder=5)

    color = colors[idx] if idx < len(colors) else 'black' 
    plt.plot(time_valid, row_data_smoothed, label=f"replicate_{idx}", color=color, alpha=.35, linewidth = .5)

    if idx in extrema:
        extrema_y_values = [np.polyval(np.polyfit(time_valid, row_data_smoothed, deg=8), et) for et in extrema[idx]]
        
        plt.scatter(extrema[idx], extrema_y_values, label=f'replicate_{idx} extrema', color=color, marker='o', alpha=.75)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Data and Extrema for Selected Replicates (Smoothed)')
#plt.show()   #REMOVE "#" TO SHOW FITTED EXTREMA
extrema_df = extrema_df.drop(index=failed_indices)
periods = []

for idx, row in extrema_df.iterrows():
    if (row.iloc[4] - row.iloc[2]) > 19 and (row.iloc[4] - row.iloc[2]) < 30:
        rep_period = row.iloc[4] - row.iloc[2]
        periods.append({'idx': idx, 'period': rep_period})
    elif (row.iloc[5] - row.iloc[3]) > 19 and (row.iloc[5] - row.iloc[3]) < 30:
        rep_period = row.iloc[5] - row.iloc[3]
        periods.append({'idx': idx, 'period': rep_period})
        
period_df = pd.DataFrame(periods)  
smart_period = period_df.iloc[1, 1:].mean()
smart_period = smart_period.round(0)
smart_period = int(smart_period)
print("\n", "Smart Period:", smart_period)

num_extrema = 2

columns = ['addition time', 'treatment'] + [f'HighIndex{i+1}' for i in range(num_extrema)] + [f'LowIndex{i+1}' for i in range(num_extrema)]
extrema_df_new = pd.DataFrame(columns=columns)
if Mmode == True:                                       
    extrema_results_df = pd.DataFrame(columns=['Addition Time', 'Treatment Group', 'Range1_Min', 'Range1_Max', 'Range2_Min', 'Range2_Max', 
                                       'Range3_Min', 'Range3_Max', 'Range4_Min', 'Range4_Max', 'Range5_Min', 'Range5_Max'])
if Mmode == False:
    extrema_results_df = pd.DataFrame(columns=['Addition Time', 'Treatment Group', 'Range1_Max', 'Range1_Min', 'Range2_Max', 'Range2_Min', 
                                       'Range3_Max', 'Range3_Min', 'Range4_Max', 'Range4_Min', 'Range5_Max', 'Range5_Min'])    

                                       
for idx, row in filtered_singles1.iterrows():
    data_row = row[2:]
    label1, label2 = row.iloc[0], row.iloc[1]
    
    subrange_thresh = 0.70
    if Mmode == False:
        spacer = 3
    if Mmode == True:
        spacer = 12
        
    numeric_data_range1 = pd.to_numeric(data_row.iloc[spacer:(spacer+smart_period)], errors='coerce')
    numeric_data_range2 = pd.to_numeric(data_row.iloc[(spacer + smart_period*1):(spacer+smart_period*2)], errors='coerce')
    numeric_data_range3 = pd.to_numeric(data_row.iloc[(spacer + smart_period*2):(spacer+smart_period*3)], errors='coerce')
    numeric_data_range4 = pd.to_numeric(data_row.iloc[(spacer + smart_period*3):(spacer+smart_period*4)], errors='coerce')
    numeric_data_range5 = pd.to_numeric(data_row.iloc[(spacer + smart_period*4):(spacer+smart_period*5)], errors='coerce')

    R2Spacer = (spacer + smart_period*1)
    R3Spacer = (spacer + smart_period*2)
    R4Spacer = (spacer + smart_period*3)
    R5Spacer = (spacer + smart_period*4)
    
    range1mean = numeric_data_range1.mean()
    range1amp = (numeric_data_range1.max() - numeric_data_range1.min()) / 2
    range2mean = numeric_data_range2.mean()
    range2amp = (numeric_data_range2.max() - numeric_data_range2.min()) / 2
    range3mean = numeric_data_range3.mean()
    range3amp = (numeric_data_range3.max() - numeric_data_range3.min()) / 2
    range4mean = numeric_data_range4.mean()
    range4amp = (numeric_data_range4.max() - numeric_data_range4.min()) / 2
    range5mean = numeric_data_range5.mean()
    range5amp = (numeric_data_range5.max() - numeric_data_range5.min()) / 2      
    
    range1data = smart_range_extrema(numeric_data_range1, range1mean, range1amp, spacer, subrange_thresh=0.75)
    range2data = smart_range_extrema(numeric_data_range2, range2mean, range2amp, R2Spacer, subrange_thresh=0.75)
    range3data = smart_range_extrema(numeric_data_range3, range3mean, range3amp, R3Spacer, subrange_thresh=0.75)
    range4data = smart_range_extrema(numeric_data_range4, range4mean, range4amp, R4Spacer, subrange_thresh=0.5)
    range5data = smart_range_extrema(numeric_data_range5, range5mean, range5amp, R5Spacer, subrange_thresh=0.5)
    
    if Mmode == True:
        # Initialize lists for all range values
        all_range_values = []

        # Collect and sort values for each range
        for range_data in [range1data, range2data, range3data, range4data, range5data]:
            range_values = []
            if range_data[0]:  # Check if range_data has values
                range_values.append(range_data[0][0])  # Add the first value
            if range_data[1]:  # Check for the second value
                range_values.append(range_data[1][0])  # Add the second value
            all_range_values.extend(range_values)  # Add values to the all_range_values list

        # Sort the collected values
        all_range_values = sorted(all_range_values)

        # Insert blanks for gaps greater than 15
        for i in range(len(all_range_values) - 1):
            # Check the gap between consecutive values
            if all_range_values[i + 1] is not None and all_range_values[i] is not None:
                if all_range_values[i + 1] - all_range_values[i] > 15:
                    all_range_values.insert(i + 1, None)  # Insert None after the current index
                    # To maintain the size of 10, break if we reach 10 values
                    if len(all_range_values) >= 10:
                        break

        # Ensure the length of all_range_values is 10, filling with None if necessary
        while len(all_range_values) < 10:
            all_range_values.append(None)

        # Create the replicate_row with sorted values and gaps filled
        replicate_row = [
            label1, label2,
            *all_range_values[:10]  # Take the first 10 values
        ]


  
           
    else:
        replicate_row = [
            label1, label2, 
            range1data[0][0] if range1data[0] else None, range1data[1][0] if range1data[1] else None, 
            range2data[0][0] if range2data[0] else None, range2data[1][0] if range2data[1] else None, 
            range3data[0][0] if range3data[0] else None, range3data[1][0] if range3data[1] else None, 
            range4data[0][0] if range4data[0] else None, range4data[1][0] if range4data[1] else None, 
            range5data[0][0] if range5data[0] else None, range5data[1][0] if range5data[1] else None
        ]       
    extrema_results_df.loc[len(extrema_results_df)] = replicate_row 
    
# AVERAGING AND SEM OF EXTREMA -> REMOVING ABNORMAL RESPONSE REPLICATES -> REAVERAGE
all_together_now5 = combined_average_new(extrema_results_df, "Addition Time", "Treatment Group")
all_together_now6 = combined_sem_new(extrema_results_df, "Addition Time", "Treatment Group")
all_together_now7 = combined_sd_new(extrema_results_df, "Addition Time", "Treatment Group")

with pd.ExcelWriter('3.1 extrema_individual_reps.xlsx') as writer:
        extrema_results_df.to_excel(writer, index=False)

mask = np.ones(len(extrema_results_df), dtype=bool) 
excluded_reps_count = 0
for index, row in extrema_results_df.iterrows():
    combined_label = row['combined_labels']
    
    mean_row = all_together_now5.loc[all_together_now5['combined_labels'] == combined_label, 'Range3_Min']
    if not mean_row.empty:
        sem_row = all_together_now7.loc[all_together_now7['combined_labels'] == combined_label, 'Range3_Min']
        extrema_value = row['Range3_Min']
    else:
        mean_row = all_together_now5.loc[all_together_now5['combined_labels'] == combined_label, 'Range4_Max']    
        sem_row = all_together_now7.loc[all_together_now7['combined_labels'] == combined_label, 'Range4_Max']  
        extrema_value = row['Range4_Max']        
    if not mean_row.empty and not sem_row.empty:
        mean = mean_row.iloc[0]
        sd = sem_row.iloc[0]  
        upper_bound = mean + 2 * sd
        lower_bound = mean - 2 * sd

        if extrema_value is not None:
            if lower_bound <= extrema_value <= upper_bound:
                continue 
            else:
                idx = all_together_now5[all_together_now5['combined_labels'] == combined_label].index
                mask[idx] = False 
                excluded_reps_count += 1
        else:
            idx = all_together_now5[all_together_now5['combined_labels'] == combined_label].index
            mask[idx] = False 
            excluded_reps_count += 1
            
extrema_results_df = extrema_results_df[mask].reset_index(drop=True)
all_together_now5 = combined_average_new(extrema_results_df, "Addition Time", "Treatment Group")
all_together_now6 = combined_sem_new(extrema_results_df, "Addition Time", "Treatment Group")
print("\n")
print(f"Number of replicates excluded: {excluded_reps_count}")

# CALCULATE AND SAVE PERIOD ANALYSIS
period3 = all_together_now5.iloc[-1]

# CHECK IF TROUGH 1 TIME OR TROUGH 2 TIME IS MISSING OR ZERO
if (period3.loc['Range2_Max'] - period3.loc['Range1_Max']) >= 19:
    period3 = period3.loc['Range2_Max'] - period3.loc['Range1_Max']
else:
    period3 = period3.loc['Range3_Max'] - period3.loc['Range2_Max']
print("\n", "PERIOD REPORT")
print('period:', period3, 'Hrs') 
    
period4 = all_together_now6.iloc[-1]    
if period3 >= 19:
    period4 = (period4.loc['Range2_Max']**2 + period4.loc['Range1_Max']**2)**0.5
else:
    period4 = (period4.loc['Range3_Max']**2 + period4.loc['Range2_Max']**2)**0.5
print('period SEM:', period4, 'Hrs') 
print("\n"*2, "Period After Addition")

rows = []
for idx, row in all_together_now5.iterrows():
    period_calc = row.loc['Range4_Max'] - row.loc['Range3_Max']
    sem_row = all_together_now6.loc[idx]
    period_calc_SEM = (sem_row.loc['Range4_Min']**2 + sem_row.loc['Range3_Min']**2)**0.5    
    rows.append({
        'combined_labels': row['combined_labels'],
        'Period': period_calc,
        'SEM': period_calc_SEM
    })

period_df = pd.DataFrame(rows)
print(period_df,
"\n")
output_file = '9. Post-Addition_Period_Report.xlsx'
period_df.to_excel(output_file, index=False)
    
#CT CALCULATIONS

CT_list = []
if Mmode == False:
    for value in addition_time[0]: 
        rand_value = value/period3*24

        if rand_value >= 60:
            rand_value -= 60
        if rand_value >= 36:
            rand_value -= 36
        if rand_value >= 24:
            rand_value -= 12
        if rand_value == 0:
            rand_value += 12
        CT_list.append(rand_value.round(4))
else:
    for value in addition_time[0]: 
        rand_value = value/period3*24
        if rand_value >= 72:
            rand_value -= 72
        if rand_value >= 48:
            rand_value -= 48
        if rand_value >= 24:
            rand_value -= 24
        if rand_value == 0:
            rand_value = 0       
        CT_list.append(rand_value)            
CT_list = pd.DataFrame(CT_list)
    
#CT RELABELLING

time_dict = {add_time: ct_time for add_time, ct_time in zip(addition_time.iloc[:, 0], CT_list.iloc[:, 0])}

print("Addition Time to CT Mapping")
print(time_dict)
all_together_now5.insert(1, 'add_times', all_together_now5['combined_labels'])
all_together_now6.insert(1, 'add_times', all_together_now6['combined_labels'])
    
all_together_now5 = replace_addition_times_new(all_together_now5, 'combined_labels', time_dict)
all_together_now6 = replace_addition_times_new(all_together_now6, 'combined_labels', time_dict) 
       
#EXPORT EXTREMA

with pd.ExcelWriter('3. all_extrema.xlsx') as writer:
        all_together_now5.to_excel(writer, index=False)
with pd.ExcelWriter('4. all_extrema_sem.xlsx') as writer:
        all_together_now6.to_excel(writer, index=False)    

control_keywords = ['CT', 'CTRL', 'Control']

# CONVERT THE 'COMBINED_LABELS' COLUMN TO A STRING AND FIND ROWS CONTAINING CONTROL KEYWORDS
control_rows = all_together_now5[all_together_now5['combined_labels'].str.contains('|'.join(control_keywords), case=False, na=False)].copy()

# SAVE THE CONTROL ROWS AND REMOVE THEM FROM THE DATAFRAME
all_together_now5 = all_together_now5[~all_together_now5['combined_labels'].str.contains('|'.join(control_keywords), case=False, na=False)].copy()

# PERFORM SUBTRACTION FOR MATCHING TREATMENT GROUPS BY ADDITION TIME
for timepoint in control_rows['combined_labels'].unique():
    control_row = control_rows[control_rows['combined_labels'] == timepoint].iloc[0, 2:]
    for index, row in all_together_now5.iterrows():
        if timepoint.split('_')[0] == row['combined_labels'].split('_')[0]:  
            for col in all_together_now5.columns[2:]:
                #Conversion to CT
                all_together_now5.at[index, col] = (((control_row[col] - row[col])/period3)*24)
        
# CONVERT THE 'COMBINED_LABELS' COLUMN TO A STRING AND FIND ROWS CONTAINING CONTROL KEYWORDS
control_rows = all_together_now6[all_together_now6['combined_labels'].str.contains('|'.join(control_keywords), case=False, na=False)].copy()

# SAVE THE CONTROL ROWS AND REMOVE THEM FROM THE DATAFRAME
all_together_now6 = all_together_now6[~all_together_now6['combined_labels'].str.contains('|'.join(control_keywords), case=False, na=False)].copy()

# PERFORM SUBTRACTION FOR MATCHING TREATMENT GROUPS BY ADDITION TIME
for timepoint in control_rows['combined_labels'].unique():
    control_row = control_rows[control_rows['combined_labels'] == timepoint].iloc[0, 2:]
    for index, row in all_together_now6.iterrows():
        if timepoint.split('_')[0] == row['combined_labels'].split('_')[0]: 
            for col in all_together_now6.columns[2:]:
                #SEM also converted to CT
                all_together_now6.at[index, col] = ((((control_row[col]**2 + row[col]**2)**0.5)/period3)*24)

with pd.ExcelWriter('5. all_extrema_PS.xlsx') as writer:
        all_together_now5.to_excel(writer, index=False)
with pd.ExcelWriter('6. all_extrema_PS_sem.xlsx') as writer:
        all_together_now6.to_excel(writer, index=False)
            
output_dir = '7. PRC Plot Data'
os.makedirs(output_dir, exist_ok=True)

# PROCESSES AND SAVES DATAFRAMES TO EXCEL FILES BASED ON DRUG DISTINCTION
df_names = []
df_names_SEM = []
for label in labels_df.iloc[:, 0].unique():
    treatment_df = all_together_now5[all_together_now5['combined_labels'].apply(lambda x: x.split('_')[1] == label)].copy()
    treatment_df = all_together_now5[all_together_now5['add_times'].apply(lambda x: x.split('_')[1] == label)].copy()
    treatment_df.loc[:, 'combined_labels'] = treatment_df['combined_labels'].apply(lambda x: x.split('_', 1)[0] if len(x.split('_')) > 1 else x)
    treatment_df.loc[:, 'add_times'] = treatment_df['add_times'].apply(lambda x: x.split('_', 1)[0] if len(x.split('_')) > 1 else x)
    treatment_df['combined_labels'] = pd.to_numeric(treatment_df['combined_labels'], errors='coerce')    
    treatment_df = treatment_df.sort_values(by='combined_labels')

    df_name = f"{label}"
    df_names.append(df_name)
    
    file_name = f"{label}_PS.xlsx"
    file_path = os.path.join(output_dir, file_name)
    with pd.ExcelWriter(file_path) as writer:
        treatment_df.to_excel(writer, index=False)
    
    globals()[df_name] = treatment_df  

for label in labels_df.iloc[:, 0].unique():
    treatment_df_SEM = all_together_now6[all_together_now6['combined_labels'].apply(lambda x: x.split('_')[1] == label)].copy()
    treatment_df_SEM = all_together_now6[all_together_now6['add_times'].apply(lambda x: x.split('_')[1] == label)].copy()
    treatment_df_SEM.loc[:, 'combined_labels'] = treatment_df_SEM['combined_labels'].apply(lambda x: x.split('_', 1)[0] if len(x.split('_')) > 1 else x)
    treatment_df_SEM.loc[:, 'add_times'] = treatment_df_SEM['add_times'].apply(lambda x: x.split('_', 1)[0] if len(x.split('_')) > 1 else x)
    treatment_df_SEM['combined_labels'] = pd.to_numeric(treatment_df_SEM['combined_labels'], errors='coerce')    
    treatment_df_SEM = treatment_df_SEM.sort_values(by='combined_labels')

    df_name = f"{label}_SEM"
    df_names_SEM.append(df_name)
    
    file_name = f"{label}_PS_SEM.xlsx"
    file_path = os.path.join(output_dir, file_name)
    with pd.ExcelWriter(file_path) as writer:
        treatment_df_SEM.to_excel(writer, index=False)
    
    globals()[df_name] = treatment_df_SEM        
    
print("\n")
print("PHASE SHIFT ANALYSIS")

#SORTS DATAFRAME FOR CHOICE EXTREMA TO BUILD PRC

for df_name in df_names:
    df = globals()[df_name]
    #print("\n")
    #print(df)
    #print("\n")    
    phase_shifts = []
    
    for index, row in df.iterrows():
        try:
            add_time_value = pd.to_numeric(row['add_times'], errors='coerce')
            
            if pd.notna(add_time_value):
                division_result = add_time_value / period3
                
                if Mmode == False:
                    if division_result < 1:
                        phase_shift_value = row['Range1_Max']
                    elif 1 < division_result < 1.5:
                        phase_shift_value = row['Range2_Max']
                    elif 1.5 < division_result < 2:
                        phase_shift_value = row['Range3_Min']
                    elif 2 < division_result:
                        phase_shift_value = row['Range3_Max']
                    else:
                        phase_shift_value = None
                else:        
                    if division_result < 1:
                        phase_shift_value = row['Range3_Max']
                    elif 1 < division_result < 1.5:
                        phase_shift_value = row['Range3_Max']
                    elif 1.5 < division_result < 2:
                        phase_shift_value = row['Range4_Max']
                    elif 2 <= division_result :
                        phase_shift_value = row['Range4_Max']                        
                    else:
                        phase_shift_value = None
                    
            else:
                phase_shift_value = None
        except (TypeError, ValueError):
            phase_shift_value = None
        
        phase_shifts.append(phase_shift_value)

    df['Phase Shift'] = phase_shifts

    #Double Plotting    
    first_phase_shift = df['Phase Shift'].iloc[0] if not df['Phase Shift'].empty else None
    first_add = df['combined_labels'].iloc[0] if not df['combined_labels'].empty else None
    second_phase_shift = df['Phase Shift'].iloc[1] if not df['Phase Shift'].empty else None    
    second_add = df['combined_labels'].iloc[1] if not df['combined_labels'].empty else None
    last_phase_shift = df['Phase Shift'].iloc[-1] if not df['Phase Shift'].empty else None
    last_add = df['combined_labels'].iloc[-1] if not df['combined_labels'].empty else None

    if first_add is not None:
        new_combined_label = first_add + 24
        new_row = pd.DataFrame({'combined_labels': [new_combined_label], 'Phase Shift': [first_phase_shift]})        
        df = pd.concat([df, new_row], ignore_index=True)        
        if new_combined_label == 24:
            new_combined_label1 = second_add + 24
            new_row = pd.DataFrame({'combined_labels': [new_combined_label1], 'Phase Shift': [second_phase_shift]})
            df = pd.concat([df, new_row], ignore_index=True)                    


    if last_add is not None:
        new_combined_label = last_add - 24 
        new_row = pd.DataFrame({'combined_labels': [new_combined_label], 'Phase Shift': [last_phase_shift]})
        
        df = pd.concat([new_row, df], ignore_index=True)   
    print(df)
    result_df_name = f"{df_name}_PRC"
    globals()[result_df_name] = df[['combined_labels', 'Phase Shift']].copy()
    
plot_dir = '8. PRC Plots'
os.makedirs(plot_dir, exist_ok=True)

#PROCESS REPEATED FOR SEM

for df_name in df_names_SEM:
    df = globals()[df_name]
    #print("\n")
    #print(df)
    #print("\n") 
    phase_shifts_SEM = []
    
    for index, row in df.iterrows():
        try:
            add_time_value = pd.to_numeric(row['add_times'], errors='coerce')
            
            if pd.notna(add_time_value):
                division_result = add_time_value / period3
                
                if not Mmode:
                    if division_result < 1:
                        phase_shift_SEM_value = row['Range1_Max']
                    elif 1 < division_result < 1.5:
                        phase_shift_SEM_value = row['Range2_Max']
                    elif 1.5 < division_result < 2:
                        phase_shift_SEM_value = row['Range3_Min']
                    elif 2 < division_result:
                        phase_shift_SEM_value = row['Range3_Max']
                    else:
                        phase_shift_SEM_value = None
                else:        
                    if division_result < 1:
                        phase_shift_SEM_value = row['Range2_Max']
                    elif 1 < division_result < 1.5:
                        phase_shift_SEM_value = row['Range3_Max']
                    elif 1.5 < division_result < 2:
                        phase_shift_SEM_value = row['Range4_Max']
                    elif 2 <= division_result:
                        phase_shift_SEM_value = row['Range4_Max']                        
                    else:
                        phase_shift_SEM_value = None
                    
            else:
                phase_shift_SEM_value = None
        except (TypeError, ValueError):
            phase_shift_SEM_value = None
        phase_shifts_SEM.append(phase_shift_SEM_value)
        
    df['Phase Shift SEM'] = phase_shifts_SEM
    
    #Double Plotting
    first_phase_shift_sem = df['Phase Shift SEM'].iloc[0] if not df['Phase Shift SEM'].empty else None
    second_phase_shift_sem = df['Phase Shift SEM'].iloc[1] if not df['Phase Shift SEM'].empty else None
    last_phase_shift = df['Phase Shift SEM'].iloc[-1] if not df['Phase Shift SEM'].empty else None
    last_add = df['combined_labels'].iloc[-1] if not df['combined_labels'].empty else None
    
    if first_add is not None:
        new_combined_label = first_add + 24
        new_row = pd.DataFrame({'combined_labels': [new_combined_label], 'Phase Shift SEM': [first_phase_shift_sem]})        
        df = pd.concat([df, new_row], ignore_index=True)   
        if new_combined_label == 24:
            new_combined_label1 = second_add + 24
            new_row = pd.DataFrame({'combined_labels': [new_combined_label1], 'Phase Shift SEM': [second_phase_shift_sem]})
            df = pd.concat([df, new_row], ignore_index=True)  

    if last_add is not None:
        new_combined_label = last_add - 24 
        new_row = pd.DataFrame({'combined_labels': [new_combined_label], 'Phase Shift SEM': [last_phase_shift]})
        
        df = pd.concat([new_row, df], ignore_index=True)
        
    result_df_name = f"{df_name}_PRC"
    globals()[result_df_name] = df[['combined_labels', 'Phase Shift SEM']].copy()
   
#MAKING AND SAVING OF PRC'S

from matplotlib import pyplot as pl
colors = {}
prc_df_names = [name for name in globals() if name.endswith('_PRC') and 'SEM' not in name]

for result_df_name in prc_df_names:
    df = globals()[result_df_name]
    print(result_df_name)
    print(df)
    print("\n")
    
    sem_df_name = result_df_name.replace('_PRC', '_SEM_PRC') 

    if sem_df_name in globals():
        sem_df = globals()[sem_df_name]  
        
        if result_df_name not in colors:
            colors[result_df_name] = (np.random.random(), np.random.random(), np.random.random())
        
        color1 = colors[result_df_name]

        if not df.empty and 'combined_labels' in df.columns and 'Phase Shift' in df.columns:
            color1 = colors[result_df_name]
            if len(df['combined_labels']) > 0 and len(df['Phase Shift']) > 0:
                # Create the figure for plotting
                plt.figure(figsize=(10, 6))

                # Plot the raw data
                plt.plot(df['combined_labels'], df['Phase Shift'], color=color1, marker='o', linestyle='-', 
                         linewidth=2.5, label=f'{result_df_name} Phase Shift')

                # Add error bars if SEM data is present
                if graphing == True:
                    plt.errorbar(df['combined_labels'], df['Phase Shift'], yerr=sem_df['Phase Shift SEM'], 
                                 fmt='none', capsize=5, ecolor='k', elinewidth=.75, alpha=1)
                elif graphing == False:
                    error = sem_df['Phase Shift SEM'].values  # Use SEM as error
                    plt.fill_between(df['combined_labels'], 
                                     df['Phase Shift'] - error, 
                                     df['Phase Shift'] + error,
                                     alpha=0.25, edgecolor=color1, facecolor=color1, label='SEM Range')

                # Create the spline interpolation
                X_Y_Spline = make_interp_spline(df['combined_labels'], df['Phase Shift'])
                X = np.linspace(df['combined_labels'].min(), df['combined_labels'].max(), 500)
                Y = X_Y_Spline(X)
                
                if graphing1 == True:
                    plt.plot(X, Y, color=color1, linewidth=1.5, alpha=.75)
                plt.title(result_df_name)
                plt.xlabel('CT Hrs')
                plt.ylabel('Phase Shift')

                if Mmode == True:
                    plt.ylim(-4, 4)
                plt.xlim(-1, 25)
                plt.xticks(np.arange(0, 28, 4))
                plt.grid(True)
                
                plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)                
                ax = plt.gca() 
                ax.axhline(0, color='black', linewidth=1.5)
                
                plot_filename = os.path.join(plot_dir, f"{result_df_name}.jpg")
                plt.savefig(plot_filename)
                plt.close()
            else:
                print(f"Data in {result_df_name} is empty or insufficient for interpolation.")
        else:
            print(f"Required columns missing or empty in {result_df_name}.")
    else:
        print(f"SEM DataFrame {sem_df_name} not found.")
        
# Combined plot for all datasets except "UT"
plt.figure(figsize=(10, 6))
for result_df_name in prc_df_names:
    if "UT" not in result_df_name:
        df = globals()[result_df_name]
        if not df.empty and 'combined_labels' in df.columns and 'Phase Shift' in df.columns:
            color1 = colors[result_df_name]
            plt.plot(df['combined_labels'], df['Phase Shift'], color=color1, marker='o', linestyle='-', 
                     linewidth=2.5, label=f'{result_df_name} Phase Shift')
            
            if graphing == True:
                plt.errorbar(df['combined_labels'], df['Phase Shift'], yerr=sem_df['Phase Shift SEM'], 
                                 fmt='none', capsize=5, ecolor='k', elinewidth=.75, alpha=1)
            elif graphing == False:
                error = sem_df['Phase Shift SEM'].values 
                plt.fill_between(df['combined_labels'], 
                                 df['Phase Shift'] - error, 
                                 df['Phase Shift'] + error,
                                 alpha=0.25, edgecolor=color1, facecolor=color1, label='SEM Range')
            
            # Optional spline interpolation
            X_Y_Spline = make_interp_spline(df['combined_labels'], df['Phase Shift'])
            X = np.linspace(df['combined_labels'].min(), df['combined_labels'].max(), 500)
            Y = X_Y_Spline(X)
            if graphing1 == True:
                plt.plot(X, Y, color=color1, linewidth=1.5, alpha=.75)
plt.title('Combined PRC')
plt.xlabel('CT Hrs')
plt.ylabel('Phase Shift')
plt.xlim(-.5, 24.5)
plt.ylim(-4, 4)
plt.xticks(np.arange(-4, 28, 4))
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.6)

# Darker y=0 line
ax = plt.gca()
ax.axhline(0, color='black', linewidth=1.5)
plt.legend(loc='upper right')

# Save the combined plot
combined_plot_filename = os.path.join(plot_dir, "combined_plot_excluding_UT.jpg")
plt.savefig(combined_plot_filename)
plt.close()
        
#FILE ORGANIZATION

import shutil
import random
from datetime import datetime

source_file1 = "1. Single_Count_Avg&SEM.xlsx"
source_file2 = "2. averaged_plots"
source_file3 = "2. averaged_plot.jpg"
source_file4 = "3. all_extrema.xlsx"
source_file5 = "4. all_extrema_sem.xlsx"
source_file6 = "5. all_extrema_PS.xlsx"
source_file7 = "6. all_extrema_PS_sem.xlsx"
source_file8 = "7. PRC Plot Data"
source_file9 = "8. PRC Plots"
source_file10 = "9. Post-Addition_Period_Report.xlsx"
source_file11 = "3.1 extrema_individual_reps.xlsx"
source_file12 = "PRC_Feed_Doc.xlsx"

file_list = [source_file1, source_file2, source_file3, source_file4, source_file5, source_file6, source_file7, source_file8, source_file9, source_file10, source_file11]
current_time = datetime.now().strftime("%Y%m%d_%H%M")

target_directory1 = f"{input_name} PRC {current_time}"

if not os.path.exists(target_directory1):
    os.makedirs(target_directory1)

for file_name in file_list:
    if os.path.exists(file_name):
        target_path = os.path.join(target_directory1, file_name)
        shutil.move(file_name, target_path)
        print(f"File {file_name} moved to {target_directory1}" )

#COPY FEEDERFILE
if os.path.exists(source_file12):
    destination_path = os.path.join(target_directory1, source_file12)
    shutil.copy2(source_file12, destination_path)
    print(f"File {source_file12} copied to {destination_path}")
else:
    print(f"File {source_file12} does not exist, skipping copy.")

target_directory2 = "Stored Results"
if not os.path.exists(target_directory2):
    os.makedirs(target_directory2)

final_target_path = os.path.join(target_directory2, target_directory1)
shutil.move(target_directory1, final_target_path)
print(f"Directory {target_directory1} moved to {target_directory2}")
