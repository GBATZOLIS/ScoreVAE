# models/__init__.py
import importlib

def get_model(config):
    try:
        # Extract the model class name from the configuration
        model_name = config.model.network

        # Dynamically import the module containing the model
        module = importlib.import_module(f'.{model_name}', package='models')

        # Get the model class from the imported module
        model_class = getattr(module, model_name)

        # Initialize and return the model with the provided configuration
        return model_class(config.model)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}'. Make sure the module and class are correctly named and placed in the 'models' directory.") from e
