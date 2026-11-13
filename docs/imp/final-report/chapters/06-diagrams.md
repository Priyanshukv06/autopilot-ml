# Proposed flowchart, block diagram and DFD

This chapter shows the structure of AutoPilot-ML as a block diagram of its layers, data flow diagrams at levels 0 and 1, and a flowchart of the Session loop. All diagrams were drawn for this report from the code. Boxes drawn in grey are libraries used as they are; white boxes are parts built in this project.

## Block diagram

Figure @fig:block shows the layers. Users reach the system through the Streamlit dashboard, the command-line scripts or the local web API. All three call the same services, because no service imports the user interface; the same code therefore runs from the dashboard, the scripts and the tests. The Session layer holds the autonomous loop and its components. The Experiment layer runs one search and evaluates it: it calls FLAML, which in turn trains LightGBM, XGBoost and scikit-learn models. Results are stored as JSON records, model files, MLflow runs and the memory file.

::figure fig:block | figures/fig_block.png | 15.5 | Block diagram of AutoPilot-ML; grey boxes are external libraries used unchanged

## Data flow diagram, level 0

Figure @fig:dfd0 is the context diagram. The user gives a CSV dataset, the target column and a budget, and receives models, test scores, Diagnoses, decisions and explanations. OpenML supplies the public datasets of the benchmark, and a local language model server (Ollama) receives only facts from a finished Session record and returns a summary text.

::figure fig:dfd0 | figures/fig_dfd0.png | 15 | Data flow diagram, level 0 (context diagram)

## Data flow diagram, level 1

Figure @fig:dfd1 opens the system into its main processes and data stores. Process 1 splits and profiles the data; processes 2 to 6 form the loop of search, Assessment, Diagnosis, Proposal and Judgement; process 7 controls the budget; process 8 makes the final evaluation and the explanations. The Session records, the model files, the MLflow runs and the Experiment_Memory are written by the loop, and only the memory is read back by a later Session.

::figure fig:dfd1 | figures/fig_dfd1.png | 15.5 | Data flow diagram, level 1

## Flowchart of the Session loop

Figure @fig:flow follows one Session from the split to the final evaluation. It is the same loop as Algorithm @alg:session, with the Judge's decision of Algorithm @alg:judge inside it. The loop leaves through the stop check of the Resource_Manager only, so every Session ends with a Final_Evaluation and a complete record, also when the user presses Stop.

::figure fig:flow | figures/fig_flow.png | 12 | Flowchart of one Session
