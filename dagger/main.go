package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

// Prepare the source directory
func setupSource(client *dagger.Client) *dagger.Directory {
	source := client.Host().Directory("..", dagger.HostDirectoryOpts{
		Exclude: []string{
			"dagger/",
			"__pycache__/",
			"*.pyc",
			".pytest_cache/",
			"mlruns/",
			".venv/",
		},
	})

	return source
}

// Build the base Python container with dependencies
func buildPythonContainer(client *dagger.Client, source *dagger.Directory) *dagger.Container {
	// Build base image with pip upgrade
	pythonBase := client.Container().
		From("python:3.11.5").
		WithExec([]string{"pip", "install", "--upgrade", "pip"})

	// Install dependencies (cached layer)
	pythonWithDeps := pythonBase.
		WithFile("/tmp/requirements.txt", client.Host().File("../requirements.txt")).
		WithExec([]string{"pip", "install", "-r", "/tmp/requirements.txt"})

	// Mount source code and pull data
	pythonContainer := pythonWithDeps.
		WithDirectory("/pipeline", source).
		WithWorkdir("/pipeline").
		WithExec([]string{"dvc", "update", "data/raw/raw_data.csv.dvc"})

	return pythonContainer
}

// Execute the data preprocessing step
func runPreprocessing(ctx context.Context, container *dagger.Container) (*dagger.Container, error) {
	fmt.Println("\nStep 1: Data Preprocessing")

	preprocessContainer, err := container.
		WithExec([]string{"python", "-m", "src.preprocessing"}).
		Sync(ctx)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}

	return preprocessContainer, nil
}

// Run the model training step
func runTraining(ctx context.Context, container *dagger.Container) (*dagger.Container, error) {
	fmt.Println("\nStep 2: Model Training")

	trainContainer, err := container.
		WithExec([]string{"python", "-m", "src.modeling.train"}).
		Sync(ctx)
	if err != nil {
		return nil, fmt.Errorf("training failed: %w", err)
	}

	return trainContainer, nil
}

// Execute the model selection step
func runSelection(ctx context.Context, container *dagger.Container) (*dagger.Container, error) {
	fmt.Println("\nStep 3: Model Selection")

	selectContainer, err := container.
		WithExec([]string{"python", "-m", "src.modeling.select"}).
		Sync(ctx)
	if err != nil {
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	return selectContainer, nil
}

// Execute the model deployment step
func runDeployment(ctx context.Context, container *dagger.Container) (*dagger.Container, error) {
	fmt.Println("\nStep 4: Model Deployment")

	deployContainer, err := container.
		WithExec([]string{"python", "-m", "src.modeling.deploy"}).
		Sync(ctx)
	if err != nil {
		return nil, fmt.Errorf("deployment failed: %w", err)
	}

	return deployContainer, nil
}

// Export models and data from the container
func exportArtifacts(ctx context.Context, container *dagger.Container) error {
	fmt.Println("\nExporting Artifacts...")

	// Export trained models
	modelsDir := container.Directory("/pipeline/models")
	if _, err := modelsDir.Export(ctx, "../models"); err != nil {
		return fmt.Errorf("failed to export models: %w", err)
	}

	// Export processed data
	dataDir := container.Directory("/pipeline/data")
	if _, err := dataDir.Export(ctx, "../data"); err != nil {
		return fmt.Errorf("failed to export data: %w", err)
	}

	return nil
}

// Orchestrate the entire ML pipeline
func Run() error {
	ctx := context.Background()

	// Initialize Dagger client
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stdout))
	if err != nil {
		return fmt.Errorf("failed to connect to Dagger: %w", err)
	}
	defer client.Close()

	fmt.Println("Starting ML Pipeline...")

	// Setup
	source := setupSource(client)
	container := buildPythonContainer(client, source)

	// Execute pipeline stages
	container, err = runPreprocessing(ctx, container)
	if err != nil {
		return err
	}

	container, err = runTraining(ctx, container)
	if err != nil {
		return err
	}

	container, err = runSelection(ctx, container)
	if err != nil {
		return err
	}

	container, err = runDeployment(ctx, container)
	if err != nil {
		return err
	}

	// Export results
	if err := exportArtifacts(ctx, container); err != nil {
		return err
	}

	fmt.Println("\nPipeline completed successfully!")
	return nil
}

func main() {
	if err := Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
