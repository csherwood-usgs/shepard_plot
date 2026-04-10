![Example ternary diagram (Shepard plot)](./Example_Grain_Size_Gravel.png)

##  shepard_plot
Plots sand/silt/clay ternary plot (Shepard diagram)

Created with the help of ChatGPT 5.2

### Notebook version

`shepard_plot_classify.ipynb`

### Standalone version

`shep.py`

#### Environment
argparse  
warnings  
pathlib  
matplotlib  
numpy  
pandas  

#### Usage
`python shep.py Example_Grain_Size_gravel.csv`  

or
`python shep.py Example_Grain_Size.csv --labels` 

##### Optional flags
--labels            Add analysis_number labels to plotted points  
--figsize 8 8       Set figure size in inches  
--dpi 300           Set output PNG resolution  

#### Input 
Required columns:
    analysis_number, sand, silt, clay

Optional column:
    gravel

#### Output files
`<input_stem>_classes.csv` - Input plus additional columns with QA info and classification  
`<input_stem>.png`         - Shepard diagram  

#### Notes
The gravel column is optional. If present, the percentages of sand, silt, and clay are normalized to 100% and used in the plots and classification, and the gravel percentage is indicated in color on the Shepard plot.  

A check is performed to ensure the columns add up to 100% +/-1.5%. If they don't, an "N" is written in the `QA_flag` column. Otherwise, that column contains "A". The values are always renormalized so the columns sum to 100%.  



