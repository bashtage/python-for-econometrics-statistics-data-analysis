#!/usr/bin/env python
# coding: utf-8

# # Example: Exporting to $\LaTeX$
# 
# The first code block contains the imports needed and defines a flag which determines whether the 
# output $\LaTeX$ should be compiled.

# In[ ]:


# imports
import numpy as np
import subprocess

# Flag to compile output tables
compile_latex = False


# The next code block loads the npz file created using the output from the Fama-MacBeth example.
# The second part shows a generic method to restore all variables. The loaded data is in a dictionary,
# and so iterating over the keys and using `globals()` (a dictionary) in the main program.

# In[ ]:


# Load variables
f = np.load('fama-macBeth-results.npz')
data = f.items()
# Manually load parameters and std errors
arp = f['arp']
arp_se = f['arp_se']
beta = f['beta']
beta_se = f['beta_se']
J = f['J']
Jpval = f['Jpval']

# Generic restore of all data in a npz file
for key in f.keys():
    globals()[key] = f[key]
f.close()


# The document is be stored in a list. The first few lines contain the required header for a
# $\LaTeX$ document, including some packages used to improve table display and to select a custom font.
#     

# In[ ]:


# List to hold table
latex = []
# Initializd LaTeX document
latex.append(r'\documentclass[a4paper]{article}')
latex.append(r'\usepackage{amsmath}')
latex.append(r'\usepackage{booktabs}')
latex.append(r'\usepackage[adobe-utopia]{mathdesign}')
latex.append(r'\usepackage[T1]{fontenc}')
latex.append(r'\begin{document}')


# Table 1 is stored in its own list, and then extend will be used to add it to the main list. 

# In[ ]:


# Table 1
table1 = []
table1.append(r'\begin{center}')
table1.append(r'\begin{tabular}{lrrr} \toprule')
# Header
colNames = [r'VWM$^e$','SMB','HML']
header = ''
for cName in colNames:
    header += ' & ' + cName

header += r'\\ \cmidrule{2-4}'
table1.append(header)
# Main row
row = ''
for a,se in zip(arp,arp_se):
    row += r' & $\underset{{({0:0.3f})}}{{{1:0.3f}}}$'.format(se,a)
table1.append(row)
# Blank row
row = r'\\'
table1.append(row)
# J-stat row
row = r'J-stat: $\underset{{({0:0.3f})}}{{{1:0.1f}}}$ \\'.format(float(Jpval),float(J))
table1.append(row)
table1.append(r'\bottomrule \end{tabular}')
table1.append(r'\end{center}')
# Extend latex with table 1
latex.extend(table1)
latex.append(r'\newpage')


# Table 2 is a bit more complex, and uses loops to iterate over the rows of the arrays containing
# the $\beta$s and their standard errors.

# In[ ]:


# Format information for table 2
sizes = ['S','2','3','4','B']
values = ['L','2','3','4','H']
# Table 2 has the same header as table 1, copy with a slice
table2 = table1[:3]
m = 0
for i in range(len(sizes)):
    for j in range(len(values)):
        row = 'Size: {:}, Value: {:} '.format(sizes[i],values[j])
        b = beta[:,m]
        s = beta_se[m,1:]
        for k in range(len(b)):
            row += r' & $\underset{{({0:0.3f})}}{{{1: .3f}}}$'.format(s[k],b[k])
        row += r'\\ '
        table2.append(row)
        m += 1
    if i<(len(sizes)-1):
        table2.append(r'\cmidrule{2-4}')

table2.append(r'\bottomrule \end{tabular}')
table2.append(r'\end{center}')
# Extend with table 2
latex.extend(table2)


# The penultimate block finished the document, and uses write to write the lines to the $\LaTeX$ file.

# In[ ]:


# Finish document   
latex.append(r'\end{document}')
# Write to table
fid = open('latex.tex','w')
for line in latex:
    fid.write(line + '\n')
fid.close()


# Finally, if the flag is set, subprocess is used to compile the LaTeX.
#     

# In[ ]:


# Compile if needed
if compile_latex:
    exit_status = subprocess.call(r'pdflatex latex.tex', shell=True)
else:
    print("\n".join(latex))

