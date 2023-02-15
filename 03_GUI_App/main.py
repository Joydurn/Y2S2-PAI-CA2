import tkinter as tk
from tkinter import ttk, filedialog
from ttkthemes import ThemedTk

from pandastable import Table, config
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
import pickle
from PIL import ImageTk, Image

import numpy as np
import pandas as pd



#read data
X_train = pd.read_csv('assets/X_train.csv')
observed_cluster = pd.read_csv('assets/observed_cluster.csv')
# Gradient Boosting Classifier
with open('./models/XBoostingTuned.pkl', 'rb') as file:
    model = pickle.load(file)

# Scaler to prepare data for clustering
with open('./models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# K-Prototype clustering algorithm
with open('./models/unsupervised_model.pkl', 'rb') as file:
    unsupervised_model = pickle.load(file)

# PCA to visualize data in 2D
with open('./models/pca.pkl', 'rb') as file:
    pca = pickle.load(file)

#PCA loading (Weights)
pca_loadings = pca.components_

root=ThemedTk(theme="arc") #themed root
# Import the tcl file
root.tk.call('source', 'forest-light.tcl')
# Set the theme with the theme_use method
style=ttk.Style()
style.theme_use('forest-light')
style.configure('.', font=('Calibri', 14))
style.configure('TLabel', font=('Calibri', 20))

#window title
root.title('Trip Safety Classifier')
root.geometry("1600x900")
root.resizable(True, True)

# configure row weights for resizing the 3 rows (header, info, form part)
root.rowconfigure(0, weight=0)
root.rowconfigure(1, weight=0)
root.rowconfigure(2, weight=3)
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=0)

#HEADER FRAME
headFrame = ttk.Frame(root)
headFrame.grid(padx=100,row=0,column=0)
#logo
img = Image.open("./JustTaxi.png")
img=img.resize((140,140), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

imgLabel = ttk.Label(headFrame, image=img,font=("Calibri", 12))
imgLabel.grid(padx=150,row=0,column=0)

#title
title = ttk.Label(headFrame, text="TRIP SAFETY CLASSIFIER",font=("Calibri", 36),anchor="center").grid(padx=0,row=0,column=1)

#info FRAME
infoFrame=ttk.Frame(root,height=80)
infoFrame.grid(padx=75,row=1,column=0)
infoText = ttk.Label(infoFrame,font=("Calibri", 14),
text=f'''Welcome to JustTaxi's safety classifier! As a taxi driver, you can put in your driving data from your taxi's sensors to predict whether\n your driving is dangerous or safe. Some things to take note:
> All input boxes only take in float/integer values
''',borderwidth=1, relief="groove"
)
infoText.pack(side='bottom')

#CREATING TABS IN NOTEBOOK

#create Notebook widget
nb = ttk.Notebook(root)

#create two tabs
formTab = ttk.Frame(nb)
importTab = ttk.Frame(nb)

#add tabs to Notebook
nb.add(formTab, text="Data Form")
nb.add(importTab, text="Import from CSV")

#pack Notebook
nb.grid(row=2,column=0
,sticky="nsew"
)

#FORM TAB, contains 2 frames, form frame and graph frame
#FORM FRAME
formTab.columnconfigure(0, weight=1)
formTab.columnconfigure(1, weight=10)


formFrame=ttk.Frame(formTab)
formFrame.grid(row=0,column=0
,sticky="nsew"
)

# configure column weights for resizing between Data Form and Graph
formFrame.columnconfigure(0, weight=1)
formFrame.columnconfigure(1, weight=1)

# ============
# Form Inputs
# ============

# Question 1: Speed (Standard Deviation)
qn1 = ttk.Label(formFrame, text="Speed (Standard Deviation)",font=("Calibri", 14))
qn1.grid(row=0,column=0)
qn1Entry = ttk.Entry(formFrame)
qn1Entry.grid(row=1,column=0,pady=10)

# Question 2: Acceleration Z (Standard Deviation)
qn2 = ttk.Label(formFrame, text="Acceleration Z (Standard Deviation)",font=("Calibri", 14))
qn2.grid(row=0,column=1)
qn2Entry = ttk.Entry(formFrame)
qn2Entry.grid(row=1,column=1,pady=10)

# Question 3: Speed (Skewness)
qn3 = ttk.Label(formFrame, text="Speed (Skewness)",font=("Calibri", 14))
qn3.grid(row=2,column=0)
qn3Entry = ttk.Entry(formFrame)
qn3Entry.grid(row=3,column=0,pady=10)

# Question 4: Bearing (Standard Deviation)
qn4 = ttk.Label(formFrame, text="Bearing (Standard Deviation)",font=("Calibri", 14))
qn4.grid(row=2,column=1)
qn4Entry = ttk.Entry(formFrame)
qn4Entry.grid(row=3,column=1,pady=10)

# Question 5: Seconds (Max)
qn5 = ttk.Label(formFrame, text="Duration of ride (Second)",font=("Calibri", 14))
qn5.grid(row=4,column=0)
qn5Entry = ttk.Entry(formFrame)
qn5Entry.grid(row=5,column=0,pady=10)

# Question 6: Car Model
qn6 = ttk.Label(formFrame, text="Car Model",font=("Calibri", 14))
qn6.grid(row=4,column=1)
qn6Entry = tk.StringVar(formFrame)
qn6Menu = ttk.OptionMenu(formFrame, qn6Entry, "Toyota", "Mercedes-Benz", "Hyundai", "Volkswagen", "BMW", "Mazda", "Nissan", "Honda", "Chevrolet", "Ford")
qn6Entry.set("Toyota") # default value
qn6Menu.grid(row=5,column=1)


class InputRangeError(Exception):
    pass

def describe_results(cluster):
    #Give description
    description = f''' 
In our training, these were our findings for each cluster:
- Cluster 4: Most likely to have dangerous trips 
(73% dangerous trips recorded)
- Cluster 1: Next most likely to have dangerous trips 
(30% dangerous trips)
- Cluster 5: Extreme values (Considered outliers)
- Cluster 2 & 3: Average driver, not likely to have
dangerous trips
- Cluster 0: Safest drivers, very unlikely to have
dangerous trips

What affects PC1 the most? - Weightage Below
> | Acceleration (Std): {pca_loadings[0][0]:.2f} | Speed (Std): {pca_loadings[0][1]:.2f} | 
> | Speed (Skew): {pca_loadings[0][2]:.2f} | Bearing (Std): {pca_loadings[0][3]:.2f} |

What affects PC2 the most? - Weightage Below
> | Acceleration (Std) {pca_loadings[1][0]:.2f} | Speed (Std) {pca_loadings[1][1]:.2f} | 
> | Bearing (Std) {pca_loadings[1][3]:.2f} | Duration of ride {pca_loadings[0][4]:.2f} |
'''


    description =ttk.Label(predictGraphFrame, text=description,font=("Calibri", 15))
    description.grid(row=1,column=1)

    

def create_plot(data, figsize=(6, 6), batch_pred=False):
    # Scale data
    num_cols = ['acceleration_z (Std)', 'speed (Std)', 'speed (Skew)', 'bearing (Std)', 'second (Max)']
    scaled_data = scaler.transform(data[num_cols])

    # Predict cluster of data
    data = pd.concat([pd.DataFrame(scaled_data,columns=num_cols), data[['car_model', 'label']]],axis=1)
    category_position = [5,6]
    data['Cluster'] = unsupervised_model.predict(data, categorical=category_position)

    # Calculate and plot PC scores of new data
    pc_scores = pca.transform(data[num_cols])
    pc_scores_df = pd.DataFrame(pc_scores, columns=["PC1", "PC2"])
    pc_scores_df['New'] = 1
    pc_scores_df['Cluster'] = data['Cluster']

    # Plot scatter plot (show clusters)
    colors = ['green', 'orange', 'blue', 'magenta', 'red','cyan']
    fig, ax = plt.subplots(figsize=figsize)
    observed_cluster['New'] = 0

    # Plot trained clusters
    sns.scatterplot(data=observed_cluster, x='PC1', y='PC2', hue='Cluster', palette=colors, alpha=0.2, ax=ax)

    # Plot new prediction clusters
    for idx, record in pc_scores_df.iterrows():
        pc1 = [record['PC1']]
        pc2 = [record['PC2']]
        sns.scatterplot(x=pc1, y=pc2, color=colors[int(record['Cluster'])], edgecolor='black', linewidth=0.8, s=100, marker='X', ax=ax)

    plt.ylim(-6,6)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Clusters (PCA)')
    sns.move_legend(ax, "upper right", bbox_to_anchor=(1, 1))
    
    if batch_pred:
        distribution = []
        total = len(data)
        for i in range(6):
            distribution.append(len(data[data['Cluster']==i]))
        distribution.append(total)
        return fig, distribution

    return fig, pc_scores_df['Cluster']

# Predictions
def predictForm():
    # Prediction result
    resultFrame = ttk.Frame(formFrame)
    resultFrame.grid(row=7, column=0,columnspan=2,sticky="nsew", pady=20)
    try:
        # Form values
        acceleration_z_std = float(qn1Entry.get())
        speed_std = float(qn2Entry.get())
        speed_skew = float(qn3Entry.get())
        bearing_std = float(qn4Entry.get())
        second_max = int(qn5Entry.get())
        car_model = qn6Entry.get()
        dict={
            'Acceleration Z (Standard Deviation)':acceleration_z_std,
            'Speed (Standard Deviation)':speed_std,
            'Speed (Skewness)':speed_skew,
            "Bearing (Standard Deviation)":bearing_std,
            'Duration of ride (Second)':second_max
        }
        validRanges=[
            (0,100), #acc Z std
            (0,100), #speed std
            (-5,5), # speed skew
            (0,200), #bearing std
            (0,2000)  #second_max
        ]
        problemList=[]
        for entryName,entry,rangeTuple in zip(dict,dict.values(),validRanges):
            # check min and max values
            if entry>rangeTuple[0] and entry<rangeTuple[1]:
                continue 
            else:
                problemList.append([entryName,entry,rangeTuple])
        
        if problemList: #
            errorString='\n'
            for entryName,entry,rangeTuple in problemList:
                errorString+=f'{entryName}: Value {entry} out of range {rangeTuple}\n'
            raise InputRangeError(errorString)

        # Model Prediction
        X_test = pd.DataFrame([[acceleration_z_std, speed_std, speed_skew, bearing_std, second_max, car_model]], columns=['acceleration_z (Std)', 'speed (Std)', 'speed (Skew)', 'bearing (Std)', 'second (Max)', 'car_model'])
        pred = model.predict(X_test)
        
        # Safe trips
        if pred[0] == 0:
            predictionLabel=ttk.Label(resultFrame, text=f"Predicted: Safe Trip!",font=("Calibri", 16),foreground='green')
        else: # Dangerous trips
            predictionLabel=ttk.Label(resultFrame, text=f"Predicted: Dangerous Trip!",font=("Calibri", 16),foreground='red')
        predictionLabel.pack(side="bottom")

        # Plot data 
        X_test['label'] = pred
        fig, cluster = create_plot(X_test, figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=predictGraphFrame)
        canvas.get_tk_widget().grid(row=1,column=0)

        # Describe result
        describe_results(cluster)

    except InputRangeError as e:
        errorLabel=ttk.Label(resultFrame, text=f"Invalid Inputs! {e}",font=("Calibri", 12),foreground='orange')
        errorLabel.pack(side="bottom")

    except Exception as e:
        if str(e) == "could not convert string to float: ''":
            errorLabel=ttk.Label(resultFrame, text=f"Empty Inputs!",font=("Calibri", 14),foreground='orange')
            errorLabel.pack(side="bottom")
            return
        errorLabel=ttk.Label(resultFrame, text=f"Invalid Inputs! Only numbers allowed, no letters",font=("Calibri", 14),foreground='orange')
        errorLabel.pack(side="bottom")

# Submit button
predictButtonFrame = ttk.Frame(formFrame)
predictButtonFrame.grid(row=6,column=0,columnspan=2,sticky="nsew",pady=(50,0))
predict_button = ttk.Button(predictButtonFrame, text="Predict", command=predictForm)
predict_button.pack(side="top")

predictGraphFrame=ttk.Frame(formTab)
predictGraphFrame.grid(row=0,column=1
,sticky="nsew"
)
# label2=ttk.Label(predictGraphFrame, text="Graph stuff",font=("Calibri", 14))
# label2.grid(row=0,column=0)
# create_plot()
# configure column weights for resizing
formTab.columnconfigure(0, weight=1)
formTab.columnconfigure(1, weight=1)

#================
# IMPORT CSV TAB
#================
# configure column weights for resizing between Data Form and Graph

importTab.columnconfigure(0, weight=1)
importTab.columnconfigure(1, weight=1)

#IMPORT FRAME
importFrame=ttk.Frame(importTab)
importFrame.grid(row=0,column=0 
# ,sticky="nsew"
)
importFrame.columnconfigure(0, weight=1)
# Title
label1 = ttk.Label(importFrame, text="Import CSV File",font=("Calibri", 14))
label1.grid(row=0,column=0)

# Do prediction on CSV file
def read_file(event=None):
    # Read file
    filename = filedialog.askopenfilename()
    fileSmallName=filename.split('/')[-1]
    
    try:
        # Predict and save result in new csv file '[original_name]_predicted.csv'
        X_test = pd.read_csv(filename,index_col=False)
        X_test = X_test[['acceleration_z (Std)', 'speed (Std)','speed (Skew)','bearing (Std)','second (Max)','car_model']] #ensure order of columns is correct
        predictions = model.predict(X_test)
        X_test['label'] = predictions
        filename = filename[:-4] + '_predicted.csv'
        X_test.to_csv(filename, index=False)
        label1['text']=f"Loaded File: '{fileSmallName}'\nSaved new file with predictions as '{filename.split('/')[-1]}'"\

        # Plot pie chart
        graphLabel = ttk.Label(importGraphFrame, text="Distribution in Batch Data",font=("Calibri", 14))
        graphLabel.grid(row=0,column=1)
        count = X_test.groupby(['label'], as_index=False).size()
        labels = ['Safe', 'Dangerous']
        fig, ax = plt.subplots(figsize=[2,1.5])
        ax.pie(x=count['size'], autopct="%.1f%%", labels=labels, pctdistance=0.5, textprops={'fontsize': 8})
        canvas3 = FigureCanvasTkAgg(fig, master=importGraphFrame)
        canvas3.get_tk_widget().grid(row=1,column=1)
        
        #Show pandas df in table
        tableFrame=ttk.Frame(importTab)
        tableFrame.grid(row=1,column=0
        )
        pt = Table(tableFrame, dataframe=X_test, width=500,height=250,
                                        showstatusbar=False,editable=False)
        pt.show()

        options=config.load_options()
        options={'fontsize':9}
        config.apply_options(options,pt)

        for column in X_test.columns:
            pt.columnwidths[column] = 75
        
        # Plot scatter plot
        fig, dist = create_plot(X_test,figsize=(4,4), batch_pred=True)
        canvas4 = FigureCanvasTkAgg(fig, master=importGraphFrame)
        scatter=canvas4.get_tk_widget()
        scatter.grid(row=2,column=0)

        analysis = cluster_analysis(dist)
        #Give description
        description = f''' 
In our training, these were our findings for each cluster:
- Cluster 4: Most likely to have dangerous trips 
(73% dangerous trips recorded)
- Cluster 1: Next most likely to have dangerous trips 
(30% dangerous trips)
- Cluster 0: Safest drivers, very unlikely to have
dangerous trips
- Cluster 2 & 3: 'Average' drivers, not likely to have
dangerous trips
- Cluster 5: Extreme values (Considered outliers)

Cluster Stats:
Cluster 0: {dist[0]/dist[-1] * 100:.2f}% | Cluster 1: {dist[1]/dist[-1] * 100:.2f}% |
Cluster 2: {dist[2]/dist[-1] * 100:.2f}% | Cluster 3: {dist[3]/dist[-1] * 100:.2f}% | 
Cluster 4: {dist[4]/dist[-1] * 100:.2f}% | Cluster 5: {dist[5]/dist[-1] * 100:.2f}% |

- Cluster Analysis: 
{analysis}
    '''

        descriptionLabel['text']=description
    except Exception as e:
        label1['text']="ERROR: Invalid file was loaded, \nplease ensure it is a csv or a txt file with comma seperated values following the same format as the example X_test.csv"


def cluster_analysis(dist):
    cluster_max = np.argmax(dist[:6])

    # Generally the safest drivers (Cluster 0)
    if cluster_max == 0:
        return 'Nice! Looks like most of the trips fall within Cluster 0,\nthis cluster consists of drivers that are careful and meticulous!\nIt also has the least number of dangerous trips reported by passengers.\nKeep it up!'
    # Showing dangerous driving behavior (Cluster 1)
    elif cluster_max == 1:
        return 'Warning! Looks like most of the trips fall within Cluster 1,\nthis cluster consists of drivers that are likely driving more dangerously,\ndrivers from this cluster should try accelerating slower, turning slower\nand have a more stable speed distribution across their ride!'
    # Average (Cluster 2 or 3)
    elif cluster_max == 2 or cluster_max == 3:
        return 'Nothing too worrying. Looks like most of the trips fall within Cluster 2 or 3,\nthese drivers seem pretty average within the distribution. Drivers in these clusters\nshould try aiming for Cluster 0, which are the safest drivers, they can do so by accelerating\nslower, turning slower or have a more stable speed distribution across their ride.'
    # Very Dangerous Driver (Cluster 4)
    elif cluster_max == 4:
        return 'Warning! Looks like most of the trips fall within Cluster 4! Drivers and trips\nin this cluster are driving dangerously! It is important to try accelerating slower,\nturning slower and have a more stable speed distribution across their ride!'
    # Extreme Outliers (Cluster 5) - Only 3/20000 in this cluster during training
    else:
        return 'Warning! This cluster consist of drivers or trips that do not\nfall within the norms of any drivers or trips,make sure to check the\nsensors of these drivers. They might also be driving very abnormally!'

# File input button
read_file_button = ttk.Button(importFrame, text='Select File', command=read_file)
read_file_button.grid(row=2,column=0, pady=(20, 0))

#GRAPH FRAME
importGraphFrame=ttk.Frame(importTab)
importGraphFrame.grid(row=0,column=1,rowspan=2
,sticky="nsew"
)
importGraphFrame.columnconfigure(0, weight=1)
importGraphFrame.columnconfigure(1, weight=1)

descriptionLabel=ttk.Label(importGraphFrame , text='',font=("Calibri", 12))
descriptionLabel.grid(row=2,column=1)
# Pie chart on distribution of training data
graphLabel = ttk.Label(importGraphFrame, text="Distribution in Training Data:",font=("Calibri", 14))
graphLabel.grid(row=0,column=0)
count = X_train.groupby(['label'], as_index=False).size()
labels = ['Safe', 'Dangerous']
fig, ax = plt.subplots(figsize=[2,1.5])
ax.pie(x=count['size'], autopct="%.1f%%", labels=labels, pctdistance=0.5,  textprops={'fontsize': 8})
canvas2 = FigureCanvasTkAgg(fig, master=importGraphFrame)
canvas2.get_tk_widget().grid(row=1,column=0)

root.mainloop()