import torch
import os
from torchinfo import summary
from model import (
    IntraEncoder,
    InterEncoder,
    ProteinInteractionNet,
    Trainer,
    Tester
)
from uncertaintyAwareDeepLearn import VanillaRFFLayer
from optimizer import initialize_scheduler
from utils import (
    load_configuration,
    initialize_logging,
    set_random_seed,
    train_and_validate_model
)

def main():
    # --- Pre-Training Setup ---
    # Load configs. Use config file to change hyperparameters.
    config = load_configuration("config.yaml")
    model_config = config['model']
    training_config = config['training']

    # Set up logging to save output to a text file
    os.makedirs("output", exist_ok=True)
    initialize_logging("output/results.txt")
    
    # Set random seed for reproducibility
    set_random_seed(config['other']['random_seed'])
    
    # Determine the computation device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Model Initialization ---
    # Initialize the Encoder, Decoder, and overall model
    intra_encoder = IntraEncoder(
        model_config['base_size'],
        model_config['hidden_size'],
        model_config['n_layers'],
        model_config['n_heads'],
        model_config['intermediate_size'],
        model_config['dropout'],
        model_config['activation_function'],
    )
    
    inter_encoder = InterEncoder(
        model_config['base_size'],
        model_config['hidden_size'],
        model_config['n_layers'],
        model_config['n_heads'],
        model_config['intermediate_size'],
        model_config['dropout'],
        model_config['activation_function'],
    )

    gp_layer = VanillaRFFLayer(
        in_features=model_config['hidden_size'],
        RFFs=model_config['gp_layer']['rffs'],
        out_targets=model_config['gp_layer']['out_targets'],
        gp_cov_momentum=model_config['gp_layer']['gp_cov_momentum'],
        gp_ridge_penalty=model_config['gp_layer']['gp_ridge_penalty'],
        likelihood=model_config['gp_layer']['likelihood_function'],
        random_seed=config['other']['random_seed']
    )

    model = ProteinInteractionNet(intra_encoder, inter_encoder, gp_layer, device).to(device)
    summary(model)
    
    # Initialize the training and testing modules
    trainer = Trainer(model, training_config['learning_rate'], training_config['weight_decay'], training_config['batch_size'])
    tester = Tester(model)
    scheduler = initialize_scheduler(trainer, config['optimizer'])
    
    # --- Training and Validation ---
    # Perform training and validation
    train_and_validate_model(config, trainer, tester, scheduler, model, device)


# Execute the main function when the script is run
if __name__ == "__main__":
    main()
