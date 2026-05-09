Publically available dataset featuring video clips can be found at: https://huggingface.co/datasets/multimodalemotionnlp/Neurodiverse_Multimodal_Dataset

Filtered Audio Clips can be found at:
https://huggingface.co/datasets/multimodalemotionnlp/cleanedAudioFiles

HDF5 Files can be found at: 
https://huggingface.co/datasets/multimodalemotionnlp/ResearchProjectHDF5Files

Models used for this research project can be found at:
https://huggingface.co/multimodalemotionnlp/Neurodiverse_NLP_Models

Note: Make sure all folders and files are saved to the save directory.

Each of the four files has the global variable, path_to_folder, near the top. You can change this path to the directory in which the program is saved in. 
Each of the four files has the global variable, emotions_task, near the top as well. 
If you want to run the emotions task, make sure you set emotions_task = True
If you want to run the emotional dimensions task, make sure you set emotions_task = False

You have two ways of running the program:
1) Running through the command terminal
2) Running using a Python application (ie. Pycharm, etc.). If running using an application, make sure to comment out the main guard. 

In the cmd terminal, you can edit the PATH and emotions_task:
cd path/to/folder/

After pressing enter, you can enter any of the four main files (Audio_Processing.py, Text_Processing.py, Video_Processing.py, Multimodal.py) to run, 
followed by: 



Please install all necessary Python packages (as seen in the Requirements.txt file) before running any programs. 



