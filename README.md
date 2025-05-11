# law-admissions-calculator

This is a machine learning-based calculator for law admissions. Download the following files to use.

main.ipynb: the code file, written in Python using Jupyter Notebook.

lsdata1.csv, lsdata2.csv: the data files, split into two due to Github's file size limit.

Ensure that all files are in the same directory. You may run the code via localhost or a code editor
(e.g. Visual Studio Code) if you are familiar with programming; otherwise, I would recommend
Google Colab (colab.research.google.com). For Colab, import the main.ipynb file and add the data files.

Modify the information at the top of the very last code block according to your current or expected stats. I left 
comments above each variable value you should change. Afterwards, run each code block sequentially.

Note that the current code is restricted to the 30 most commonly applied schools. You may modify this in the relevant
code block, with the disclaimer that the model accuracy may be worse for other schools.

Limitations:
1. The data files are taken from lsd.law, which is a naturally self-selective site that does not encompass all law applicants.
   This means that the data results may be skewed according to the difference between the lsd.law candidate pool and the actual
   law school candidate pool.
2. Due to the highly complex nature of law school admissions, the calculator itself isn't the most accurate. However, given the
   complexity of the task, I find it accurate enough, especially for "accepted" and "rejected" predictions. Unfortunately, waitlists
   are right between the other two, and thus are the hardest to predict for this model as well. You may check the relevant
   validation statistics and confusion matrix right above the last code block for reference. 
