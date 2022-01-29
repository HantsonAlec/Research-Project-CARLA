import os
import sys
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import ScriptRunConfig
from azureml.core.environment import Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Run, Experiment, Workspace, Dataset

from dotenv import load_dotenv

load_dotenv()


def prepareEnv(ws, env_name):
    env = Environment(env_name)
    cd = CondaDependencies(
        conda_dependencies_file_path="train/lstr-azure/environment.yml")
    env.python.conda_dependencies = cd

    # Register environment to re-use later
    env.register(workspace=ws)

    return env


def prepareMachines(ws):
    # choose a name for your cluster
    compute_name = os.environ.get("AML_COMPUTE_NAME")
    compute_min_nodes = int(os.environ.get("AML_COMPUTE_MIN_NODES"))
    compute_max_nodes = int(os.environ.get("AML_COMPUTE_MAX_NODES"))
    vm_size = os.environ.get("AML_VM_SIZE")

    if compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[compute_name]
        if compute_target and type(compute_target) is AmlCompute:
            print(f"Found compute cluster with name '{compute_name}'.")

    else:
        print(
            f"No existing compute cluster found with name {compute_name}, creating new compute cluster...")
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes, identity_type='SystemAssigned')
        compute_target = ComputeTarget.create(
            ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(
            show_output=True, min_node_count=None, timeout_in_minutes=20)

    return compute_target


def prepareTraining(dataset, model_name, script_folder, compute_target, env):
    args = ['LSTR', '--data-folder', dataset.as_mount()]
    src = ScriptRunConfig(source_directory=script_folder, script='train_azure.py',
                          compute_target=compute_target, environment=env, arguments=args)
    return src


def main():
    experiment_name = os.environ.get("AML_EXPERIMENT_NAME")
    subscription_id = os.environ.get("AML_SUB_ID")
    resource_group = os.environ.get("AML_RESOURCE_GROUP")
    workspace_name = os.environ.get("AML_WORKSPACE_NAME")
    env_name = os.environ.get("AML_ENV_NAME")
    model_name = os.environ.get("AML_MODEL_NAME")
    dataset_name = os.environ.get("AML_DATASET_NAME")
    # Use env for training and testing
    script_folder = 'LSTR'
    cli_auth = AzureCliAuthentication()

    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    # Prepare!
    compute_target = prepareMachines(ws)
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    env = prepareEnv(ws, env_name)
    src = prepareTraining(dataset, model_name,
                          script_folder, compute_target, env)

    # Start training
    print('Starting training on cluster')
    exp = Experiment(workspace=ws, name=experiment_name)
    run = exp.submit(config=src)

    run.wait_for_completion(show_output=True)

    print('Training on cluster complete')


if __name__ == '__main__':
    main()
