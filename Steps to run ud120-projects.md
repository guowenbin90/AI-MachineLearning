# Run ud120-projects in Anaconda step by step
Sometimes because of the system environment or path issues, we can't run the code directly. 
Therefore, I used Anaconda to install the virutal environment to run the code. 
Here are the procedures to run the code step by step. 
## Install Anaconda
Download [Anaconda](https://www.anaconda.com/products/individual) from the official website to install it based on the system requirements.
## Create virtual environment to run the python code
- (Windows): Anaconda Powershell Prompt
- (Mac): Terminal  

1. Check the python version: ```python --version```  
2. Here is the [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to create the virtual environment.  
3. My virtual environment (udacity): ```conda create --name udacity```  
4. Enter the vitural environemnt: ```conda activate udacity```
5. Install Scikit-Learn using conda: ```conda install scikit-learn```
6. Install other modules using ```conda install ...``` if it has 'ModuleNotFoundError'. 
7. Download the [ud120-projects from github](https://github.com/guowenbin90/ud120-projects.git)
8. Enter the directory of the project, such as ```..\ud120-projects-master\naive_bayes```
9. Run the python code: ```python nb_author_id.py```
10. It will show the results of in the terminal: ```No. of Chris training emails :  7936  
No. of Sara training emails :  7884```

## Edit the code with Visual Studio Code
- Download [Visual Studio Code](https://code.visualstudio.com/)  
- Select the folder to edit the python code 
