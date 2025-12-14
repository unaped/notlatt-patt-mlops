# Notlatt Patt's ITU BDS MLOPS'25 Project

## About the structure

We follow the approach to standard data science MLOps project structure recommended by CCDS, comprising several components: 


- `.dvc/`: DVC configuration for data version control
- `.github/workflows/`: Stores GitHub Actions for CI/CD 
  - `ml_pipeline.yml`: ML Pipeline's action configuration and definitions
- `dagger/`:  Go-based Dagger workflow for ML pipeline orchestration
  - `main.go`: Dagger pipeline implementation with modular stage functions
  - `go.mod`: Go module dependencies declaration
  - `go.sum`: Go dependency checksums for reproducibility
- `data/`: Data directory following CCDS approach
  - `raw/`: Stores original, immutable datasets
    - `raw_data.csv.dvc`: DVC pointer to raw training data
  - `interim/`: Stores intermediate, transformed data and preprocessing artifacts
  - `processed/`: Stores final, cleaned datasets ready for modeling
- `docs/`: Project documentation files
- `models/`: Stores trained model artifacts and preprocessing objects
- `src/`: Source code for the ML pipeline
  - `modeling/`: Decomposition of model building files
    - `train.py`: Model training orchestration with MLflow logging
    - `models.py`: Model definitions and custom wrappers
    - `evaluate.py`: Model evaluation metrics and reporting
    - `select.py`: Model selection and MLflow registry management
    - `predict.py`: Model inference logic
    - `deploy.py`: MLflow model stage transitions and deployment
  - `config.py`: Centralized configuration and constants
  - `preprocessing.py`: Data cleaning, feature selection, and standardization
  - `features.py`: Feature engineering and transformation logic

- `.gitignore`: Git ignore patterns
- `requirements.txt`: Python dependencies for the project

## How to run the code

Running the machine learning pipeline implemented by this project can be done in two simple ways.

### 1. Locally 

Simply clone the repository with your terminal using your preferred method (i.e. HTTPS), by running: `git clone https://github.com/unaped/notlatt-patt-mlops.git` 

Then, go to the newly created directory: `cd notlatt-patt-mlops/dagger`, and lastly, run: `go run main.go`

This will kick off the Dagger workflow that orchestrates the source Python code to create the model artifact we are interested in, and store it under `models/`. 

>**Note:** You will need Docker installed and running in your device, as well as Go 1.24+ installed on it in order to be able to run the code locally without conflicts. 

### 2. Remotely

Alternatively, you can also take advantage of the implemented CI/CD action that triggers and executes the entire pipeline too, whenever code is pushed to a remote Github repository (or to a pull request for that repository). 

To do so, first fork this repository and clone it in your local machine, for instance, as described in the above section. Then, work on your machine to add whatever you deem necessary and push your changes. 

After doing that you will see a Github Action running and its logs, which you can check for any potential errors or conflicts that may arise due to your changes. 

If the pipeline doesn't crash due to newly introduced conflicts, you will be able to download the model artifact by going to the Summary tab within the Action run. 

## Conclusion

In short, it can be concluded that the project successfully applied the suggested architecture for implementing a machine learning pipeline capable of building a model artifact, as such: 

![Project architecture](./docs/project-architecture.png)
