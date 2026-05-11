# Neurodiverse Dataset

Our goal is to make a novel neurodiverse dataset featuring neurodivergent and neurotypical individuals, for the ultimate purpose of training and deploying AI models that can be used in mobile or web-based applications to assist neurodivergent individuals in daily tasks. Currently, our dataset consists of 1000 video clips. 

The publically available dataset featuring video clips can be found at: https://huggingface.co/datasets/multimodalemotionnlp/Neurodiverse_Multimodal_Dataset

There are four options to choose from for running a program. 

### Audio Processing
Audio_Processing.py
> Filtered audio clips can be found at:
https://huggingface.co/datasets/multimodalemotionnlp/cleanedAudioFiles

## Textual Processing
Text_Processing.py
- Text Preprocessing file can be found in this GitHub repository (Miscellaneous/Text_Preprocessing.txt)

## Visual Processing
Video_Processing.py
- HDF5 Files can be found at: 
https://huggingface.co/datasets/multimodalemotionnlp/ResearchProjectHDF5Files

## Multimodal Fusion
Multimodal.py

\
\
\

Note: Make sure all folders and files are saved to the save directory.

There are two ways of running a program:
1) Running through the command terminal

In the cmd terminal, navigate to the directory that the program exists in.
Example (if running Multimodal.py and performing the emotional dimensions classification task and your path_to_folder is path/to/folder/):

```
python Multimodal.py --emotions_task False --path_to_folder path/to/folder/

--emotions_task:  True  - Emotion Classification Task
                  False - Emotional Dimension Classification Task

--path_to_folder: working directory where the program is saved
```

2) Running using a Python application (ie. Pycharm, etc.).
If running using an application, make sure to comment out the main guard before running the program and change the path_to_folder global variable near the top of the program to your working directory in which the program is saved in and the emotions_task global variable to True if performing the Emotion Classification Task or False if performing the Emotional Dimension Classification Task.

If you plan to run all three modalities (Audio, Text and Video), make sure to run the embeddings files (found in this GitHub repository at Embeddings/Embeddings Programs) to store the embeddings from each modality before finally running Multimodal.py. 

Models used for this research project can be found at:
https://huggingface.co/multimodalemotionnlp/Neurodiverse_NLP_Models

Label Files can be found in this GitHub repository: 
- For the emotional classification task: Labels_File/New_Labels_By_Classification_Emotions_Threshold15.npy
- For the emotional dimensions classification task: Labels_File/Revised_New_Labels_By_Classification_Attributes.npy

Note: Please install all necessary Python packages (as seen in the requirements.txt file) before running any programs. 





