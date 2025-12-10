import time
from mlflow.tracking import MlflowClient
from src.config import MODEL_NAME, DEFAULT_MODEL_VERSION

def wait_for_deployment(model_name, model_version, stage='Staging'):
    """Wait for model version to transition to specified stage."""
    client = MlflowClient()
    status = False
    
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name, version=model_version)
        )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    
    return status

def transition_model_stage(model_name, model_version, target_stage='Staging', archive_existing=True):
    """Transition a model version to a new stage"""
    client = MlflowClient()
    
    model_version_details = dict(
        client.get_model_version(name=model_name, version=model_version)
    )
    
    if model_version_details['current_stage'] != target_stage:
        print(f"Transitioning model {model_name} v{model_version} to {target_stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=target_stage,
            archive_existing_versions=archive_existing
        )
        model_status = wait_for_deployment(model_name, model_version, target_stage)
        return model_status
    else:
        print(f'Model already in {target_stage}')
        return True

def deploy_model_to_staging(model_name, model_version):
    """Deploy a specific model version to staging."""
    return transition_model_stage(
        model_name=model_name,
        model_version=model_version,
        target_stage='Staging',
        archive_existing=True
    )

def deploy_model_to_production(model_name, model_version):
    """Deploy a specific model version to production."""
    return transition_model_stage(
        model_name=model_name,
        model_version=model_version,
        target_stage='Production',
        archive_existing=True
    )

if __name__ == "__main__":
    # Example usage
    
    success = deploy_model_to_staging(MODEL_NAME, DEFAULT_MODEL_VERSION)
    if success:
        print(f"Model {MODEL_NAME} v{DEFAULT_MODEL_VERSION} successfully deployed to staging")