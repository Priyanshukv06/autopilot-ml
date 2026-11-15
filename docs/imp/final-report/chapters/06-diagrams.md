# Proposed flowchart, block diagram and DFD

This chapter shows the structure of AutoPilot-ML as a block diagram of its layers, data flow diagrams at levels 0 and 1, and a flowchart of the session loop. We drew all of these diagrams for this report from the code. Boxes drawn in grey are libraries used as they are, and white boxes are parts we built in this project.

## Block diagram

Figure @fig:block shows the layers of the system. Users reach it through the Streamlit dashboard, the command-line scripts or the local web API, and all three call the same services. Since no service imports the user interface, the same code runs from the dashboard, the scripts and the tests. The session layer holds the autonomous loop and its components. The experiment layer runs one search and evaluates it: it calls FLAML, which in turn trains LightGBM, XGBoost and scikit-learn models. The results are stored as JSON records, model files, MLflow runs and the memory file.

::figure fig:block | figures/fig_block.png | 15.5 | Block diagram of AutoPilot-ML; grey boxes are external libraries used unchanged

## Data flow diagrams

Figure @fig:dfd0, the context diagram (level 0), shows what crosses the boundary of the system. The user provides a CSV dataset, the target and a budget, and gets back models, test scores, diagnoses, decisions and explanations. OpenML supplies the benchmark datasets, and a local language model server (Ollama) only turns facts from a finished record into a summary, so no data leaves the laptop. Figure @fig:dfd1 (level 1) opens the system up into its processes and data stores: 1.0 prepares the data, 2.0 to 4.0 form the loop of search, judgement, diagnosis and proposal, 5.0 controls the budget and 6.0 makes the final evaluation. It also shows that the test part flows only into 6.0, so no decision in the loop can see it, and that the memory is the only store that a later session reads back.

::figure fig:dfd0 | figures/fig_dfd0.png | 15 | Data flow diagram, level 0 (context diagram)

::figure fig:dfd1 | figures/fig_dfd1.png | 15.5 | Data flow diagram, level 1

## Flowchart of the session loop

Figure @fig:flow follows one session from the split to the final evaluation. It is the same loop as Algorithm @alg:session, with the Judge's decision from Algorithm @alg:judge inside it. The only way out of the loop is through the stop check of the Resource Manager, so every session ends with a final evaluation and a complete record, even when the user presses Stop.

::figure fig:flow | figures/fig_flow.png | 12 | Flowchart of one session
