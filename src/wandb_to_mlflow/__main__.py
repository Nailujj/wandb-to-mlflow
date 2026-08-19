"""``python -m wandb_to_mlflow`` -- the form MLproject entry points use."""

from wandb_to_mlflow.cli import app

if __name__ == "__main__":
    app()
