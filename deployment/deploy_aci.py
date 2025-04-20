from azureml.core import Workspace, Model, Environment, InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

ws = Workspace.from_config(path=".azureml/config.json")
model = Model(ws, name="pointnet-semseg")

env = Environment(name="pointnet-infer-env")
env.python.conda_dependencies.add_pip_package("torch")
env.python.conda_dependencies.add_pip_package("fastapi")
env.python.conda_dependencies.add_pip_package("uvicorn")
env.python.conda_dependencies.add_pip_package("pandas")

inference_config = InferenceConfig(
    entry_script="app/azureml/score.py",
    environment=env,
    source_directory="."
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={"area": "semantic-segmentation"},
    description="PointNet semantic segmentation model deployed to ACI"
)

service = Model.deploy(
    workspace=ws,
    name="pointnet-seg-aci",
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"ACI Endpoint: {service.scoring_uri}")
