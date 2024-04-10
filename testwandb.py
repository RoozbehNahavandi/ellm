import wandb

# Initialize a new run
wandb.init(project="test-project", entity="roozbeh-n99")

# Log a simple value
wandb.log({"test_value": 1})

# Finish the run
wandb.finish()
