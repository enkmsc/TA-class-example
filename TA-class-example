# you need to install matplotlib, numpy, pandas modules first.
# pip install matplotlib numpy pandas or conda install matplotlib numpy pandas

*******************************
# Line Plot 
*******************************

import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label='data')

# Label the axes and add a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Show plot
plt.show()


*******************************
# Scatter Plot 
*******************************
import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a scatter plot
plt.scatter(x, y, color='r', marker='o', label='data')

# Label the axes and add a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')

# Add legend
plt.legend(loc="lower right")

# Add grid
plt.grid(True)

# Save the figure
plt.savefig('path_to_save_figure.png', dpi=300)

# Show plot
plt.show()


*******************************
Plot Multiple Lines
******************************* 

import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20

# Sample data for multiple lines
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

# Create line plots
plt.plot(x, y1, marker='o', linestyle='-', color='b', label='Line 1')
plt.plot(x, y2, marker='s', linestyle='--', color='r', label='Line 2')

# Label the axes and add a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Line Plot Example')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Save the figure
plt.savefig('multiple_line_plot.png')

# Show plot
plt.show()



*******************************
Plot Numerous Points 
*******************************
import numpy as np
import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20

# Generate x values
x = np.linspace(0, 10, 100)  # Generating 100 points between 0 and 10

# Calculate corresponding y values using the function y = x^2
y = x ** 2

# Create a plot
plt.plot(x, y, linewidth=4, label='y = x^2')

# Label the axes and add a title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot using linspace')

# Add legend
plt.legend()

# Add grid
plt.grid(True)

# Show plot
plt.show()




*******************************
Plot Fitting Lines
*******************************
import numpy as np
import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 16

# Define the original data points (x and y values)
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 7, 8, 10, 13])

# Specify the degree of the polynomial you want to fit
degree = 2

# Perform polynomial fitting using np.polyfit
coefficients = np.polyfit(x, y, degree)

# Create a polynomial function using the coefficients
polynomial_func = np.poly1d(coefficients)

# Generate a range of x values for prediction
x_new = np.linspace(0, 7, 100)

# Use the polynomial function to predict y values for the new x values
y_predicted = polynomial_func(x_new)

# Create a scatter plot of the original data points
plt.scatter(x, y, label='Original Data')

# Plot the fitted polynomial curve using the new x and predicted y values
plt.plot(x_new, y_predicted, label='Fitted Polynomial', color='red')

# Label the axes and add a title
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Fitting Degree = " + str(degree))

# Show plot
plt.show()


*******************************
Plot Fitting Lines
*******************************
import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel("/Users/emilydai/Downloads/book1.xlsx")

# Set font properties and figure settings
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (8, 6)

# Define markers, labels, and colors
markers = ['o', 's', '^']
labels = ['F', 'H2O', 'EtOH']
colors = ['b', 'g', 'r'] 


# Plot each line with the specified marker, label, and color using enumerate
for i, label in enumerate(labels):
    plt.plot(df['stream'], df[label], marker=markers[i], label=label, color=colors[i])

# Add title and labels to the plot
plt.title('Stream vs F, H2O, EtOH')
plt.xlabel('Stream')
plt.ylabel('Values')

# Show legend and grid for the plot
plt.legend()
plt.grid(True)
plt.show()

*******************************
Create Dataframes 
*******************************
import pandas as pd 
import numpy as np
data = np.array([[1, 2, 3], 
                 [4, 5, 6], 
                 [7, 8, 9]])
df = pd.DataFrame(data)
print (df)

*******************************
Create Dataframes from Dic
*******************************
import pandas as pd
data = {'name': ['Alice', 'Bob', 'Charlie'],
        'age':[25, 30, 35]}
        
df = pd. DataFrame(data)
print(df)



*******************************
Create Dataframes from Lists
*******************************
import pandas as pd
# define the column and index labels
A = ["a", "b", "c", "d"]
B = ["Q", "S", "A"]
     
# create a DataFrame with NaN values
df = pd.DataFrame (index=B, columns=A)
     
# print the empty DataFrame
print (df)

*******************************
Shape Function
*******************************
import pandas as pd
# Create a sample DataFrame
df = pd.DataFrame(index = [1,2,3],columns = [4,5,6,7])

# Get the dimensions of the DataFrame 
shape = df.shape
print(df)
print (shape)



*******************************
Read Excel Files and Plot 
*******************************
import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel("/Users/emilydai/Downloads/book1.xlsx")

# Set font properties and figure settings
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (8, 6)

# Define markers, labels, and colors
markers = ['o', 's', '^']
labels = ['F', 'H2O', 'EtOH']
colors = ['b', 'g', 'r'] 


# Plot each line with the specified marker, label, and color using enumerate
for i, label in enumerate(labels):
    plt.plot(df['stream'], df[label], marker=markers[i], label=label, color=colors[i])

# Add title and labels to the plot
plt.title('Stream vs F, H2O, EtOH')
plt.xlabel('Stream')
plt.ylabel('Values')

# Show legend and grid for the plot
plt.legend()
plt.grid(True)
plt.show()

*******************************
Read CSV Files and Plot 
*******************************
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/emilydai/Downloads/xrd.dat", sep=',')

# Set font properties and figure settings
plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 300
plt.rcParams["figure.figsize"] = (8, 5)

plt.plot(df['Angle'], df['PSD'], linestyle='-', linewidth=2)
plt.xlim(10, 70)

plt.xlabel('2theta (Degree)')
plt.ylabel('Intensity (a.u.)')
plt.yticks([])
plt.show()
