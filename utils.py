import torch
import random
import calibration as cal
import logging
import yaml
import numpy as np
import timeit
import os
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import PPIDataset, PPICollator, get_data
from metrics import calculate_metrics
from plots import plot
from pauc.pauc import plot_roc_with_ci


# ------------------- Initialization and Configuration -------------------
# Set up logging to a specific file
def initialize_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
    logging.info("Epoch Time              Train Loss          Test Loss           Accuracy            Recall              Precision           F1                  MCC                 AUC                 Max AUC")


# Log and save metrics
def log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, accuracy, recall, precision, f1, mcc, auc, max_auc):
    metrics = [epoch, time, total_loss_train/total_train_size, total_loss_test/total_test_size, accuracy, recall, precision, f1, mcc, auc, max_auc]
    logging.info('\t'.join(map(str, metrics)))


# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Load configuration settings from a YAML file
def load_configuration(config_file):
    with open(config_file, 'r') as config_file:
        return yaml.safe_load(config_file)


# ------------------- Model Training and Testing -------------------
# Train the model for one epoch
def train_epoch(dataset, emb_dict, trainer, config, device, last_epoch):
    total_loss = 0
    total_samples = 0
    batch_size = config['training']['batch_size']
    max_seq_length = config['model']['max_sequence_length']
    base_size = config['model']['base_size']    
    
    dataset = PPIDataset(dataset, emb_dict)
    total_samples += len(dataset)
    data_collator = PPICollator(max_length=max_seq_length, base_size=base_size, test=False)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4 if os.cpu_count() >= 8 else 0
    )
    
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_loss = trainer.train(batch, last_epoch)
        total_loss += batch_loss * len(batch['x_a'])

    return total_loss, total_samples


# Test the model for one epoch
def test_epoch(dataset, emb_dict, tester, config, device, last_epoch, batch_size=1):
    y_true, y_pred, probs = [], [], []
    total_loss = 0
    total_samples = 0
    max_seq_length = config['model']['max_sequence_length']
    base_size = config['model']['base_size']
    dataset = PPIDataset(dataset, emb_dict)
    total_samples += len(dataset)
    data_collator = PPICollator(max_length=max_seq_length, base_size=base_size, test=True)
    dev_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4 if os.cpu_count() >= 8 else 0
    )

    if last_epoch: 
        for batch in tqdm(dev_loader, desc="Testing", total=len(dev_loader)):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_loss, t, y, s = tester.test(batch, last_epoch)
            y_true.extend(t)
            y_pred.extend(y)
            probs.extend(s)
            total_loss += batch_loss * len(batch['x_a'])
            
        return y_true, y_pred, probs, total_loss, total_samples

    else:
        for batch in tqdm(dev_loader, desc="Testing", total=len(dev_loader)):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_loss, t, y, s = tester.test(batch, last_epoch)
            y_true.extend(t)
            y_pred.extend(y)
            probs.extend(s)
            total_loss += batch_loss * len(batch['x_a'])
            
        return y_true, y_pred, probs, total_loss, total_samples


# Train and validate the model across multiple epochs
def train_and_validate_model(config, trainer, tester, scheduler, model, device):
    best_val_loss = float('inf')
    best_model_path = "output/best_model.pt"
    num_epochs = config['training']['epochs']
    max_auc = 0.0

    for epoch in range(num_epochs):
        total_loss_train, total_train_size = train_epoch(
            get_data(config, device)[1], get_data(config, device)[0], trainer, config, device, last_epoch=False
        )
        y_true, y_pred, probs, total_loss_test, total_test_size = test_epoch(
            get_data(config, device)[2], get_data(config, device)[0], tester, config, device, last_epoch=False
        )
        val_loss = total_loss_test / total_test_size

        # Calculate metrics for this epoch
        accuracy, recall, precision, f1, mcc, auc = calculate_metrics(y_true, y_pred, probs)
        
        # Update max AUC
        if auc > max_auc:
            max_auc = auc

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()
        log_and_save_metrics(
            epoch,
            timeit.default_timer(),
            total_loss_train, total_train_size, total_loss_test, total_test_size,
            accuracy, recall, precision, f1, mcc, auc,
            max_auc
        )
        plot(config['directories']['metrics_output'])

    # Load best model weights before final evaluation
    model.load_state_dict(torch.load(best_model_path))


def evaluate(config, tester, device, batch_size=1, bugfix=False):
    embedding_dict, _, _, test_data = get_data(config, device)
    if bugfix:
        print("Testing evaluation to doublecheck metric calculation")
        test_data = test_data.shuffle().select(range(100))

    # correct labels, predictions, raw scores
    y_true, y_pred, probs, total_loss_test, total_test_size = test_epoch(
        test_data,
        embedding_dict,
        tester,
        config,
        device,
        last_epoch=True,
        batch_size=batch_size
    )

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    probs = np.array(probs).astype(float)

    accuracy, recall, precision, f1, mcc, auc = calculate_metrics(y_true, y_pred, probs)
    
    # Print and write results to file
    test_results = [
        f'Test loss: {total_loss_test / total_test_size}',
        f'Test accuracy: {accuracy}',
        f'Test recall: {recall}',
        f'Test precision: {precision}',
        f'Test F1: {f1}',
        f'Test MCC: {mcc}',
        f'Test AUC: {auc}'
    ]
    
    # Print results
    for result in test_results:
        print(result)
    
    # Calculate Expected Calibration Error
    ece = cal.get_ece(probs, y_true)
    ece_result = f"Expected Calibration Error (ECE): {ece}"
    print(ece_result)
    test_results.append(ece_result)
    
    # Calculate uncertainty
    uncertainty = (1 - probs) * (probs) / 0.25

    for cutoff in [0.2, 0.4, 0.6, 0.8]:
        filtered_indices = uncertainty < cutoff
        y_true_filtered = y_true[filtered_indices]
        y_pred_filtered = y_pred[filtered_indices]
        true_positives = sum((y_true_filtered == 1) & (y_pred_filtered == 1))
        precision_filtered = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
        cutoff_result = f"Uncertainty Cutoff {cutoff}: Precision - {precision_filtered}, True Positives - {true_positives}"
        print(cutoff_result)
        test_results.append(cutoff_result)
    
    # Append all results to output file
    with open(config['directories']['metrics_output'], 'a') as f:
        f.write('\n')
        for result in test_results:
            f.write(result + '\n')

    # test_data has columns A, B, SeqA, SeqB, labels
    test_interactions = test_data.to_pandas()
    # Add S and uncertainty columns to test_interactions DataFrame
    test_interactions['probs'] = probs.tolist()
    test_interactions['labels'] = y_true.tolist()
    test_interactions['uncertainty'] = uncertainty

    # Saving to TSV
    test_interactions.to_csv('evaluation_results.tsv', sep='\t', index=False)

    # when pauc has saving
    print(y_true.shape, probs.shape)
    plot_roc_with_ci(y_true, probs, save_path='output/roc_curve.png')


# Save model state to file
def save_model(model, filename):
    torch.save(model.state_dict(), filename)