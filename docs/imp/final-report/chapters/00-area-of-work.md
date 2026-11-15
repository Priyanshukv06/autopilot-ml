Our project belongs to automated machine learning (AutoML) for tabular data, the kind of data kept in spreadsheets and databases. It draws on machine learning, statistics and software engineering.

On the machine learning side, the work deals with choosing a model and its hyperparameters, and with recognising the usual weaknesses of a model, such as overfitting, class imbalance, missing values and misleading or leaking columns. It also covers the changes to the data and the settings that fix these weaknesses.

On the statistics side, it deals with comparing two models fairly with repeated cross-validation and the corrected resampled t-test, and with comparing several systems over many datasets through average ranks and bootstrap intervals. It also uses ideas from decision making under uncertainty: two multi-armed bandit methods, Thompson sampling and UCB1, learn from earlier experiments which changes are worth trying first.

On the software side, the result is a tested, modular Python system with a web dashboard, a command line, a local web API and a Docker image. It was built from a written specification and runs fully offline on a laptop.
