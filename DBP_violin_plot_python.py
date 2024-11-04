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
installx('seaborn')
installx('glob')
installx('openpyxl')
installx('matplotlib')

import pandas as pd
import seaborn as sb
import glob
import openpyxl
import matplotlib.pyplot as plt

#Be sure all module above are downloaded

# Lists excel files in directory
file_list = glob.glob('*.xlsx')

# Reads excel files
for file in file_list:
    df = pd.read_excel(file)
    #debug info/ gives details on column labels
    print(df.head())
    print(df.columns)
    #making the plot now that data has been imported
    #x in data should be left most column
    sb.set(style = 'whitegrid') 
    Plot = sb.violinplot(x = df.columns[0], 
			y = df.columns[1], 
            hue = df.columns[0], 
			data=df) 
    Plot.set_xlabel('')
plt.show()

