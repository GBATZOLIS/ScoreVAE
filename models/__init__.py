# models/__init__.py
import importlib

def get_model(model_config):
    try:
        # Extract the model class name from the configuration
        model_name = model_config.network
        print(f"Model name from config: {model_name}")

        # Dynamically import the module containing the model
        module_name = f'models.{model_name}'
        print(f"Importing module: {module_name}")
        module = importlib.import_module(module_name)

        # Get the model class from the imported module
        print(f"Getting model class: {model_name} from module")
        model_class = getattr(module, model_name)

        # Initialize and return the model with the provided configuration
        print(f"Initializing model {model_name} with config: {model_config}")
        model_instance = model_class(model_config)
        print(f"Model {model_name} initialized successfully")

        return model_instance
    except ImportError as e:
        print(f"ImportError: {e}")
        raise ImportError(f"Could not import module '{model_name}'. Ensure the module is correctly named and placed in the 'models' directory.") from e
    except AttributeError as e:
        print(f"AttributeError: {e}")
        raise ImportError(f"Could not find model class '{model_name}' in the module. Ensure the class name matches the module file.") from e
    except Exception as e:
        print(f"Exception: {e}")
        raise e
