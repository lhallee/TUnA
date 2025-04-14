import torch
import random
import calibration as cal
import logging
import yaml
import numpy as np
import timeit
import os
from pauc import plot_roc_with_ci
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data import PPIDataset, PPICollator, get_data
from metrics import calculate_metrics
from plots import plot


# ------------------- Initialization and Configuration -------------------
# Set up logging to a specific file
def initialize_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(message)s')
    logging.info("Epoch Time              Train Loss          Test Loss           AUC                 PRC                 Accuracy            Sensitivity         Specificity         Precision           F1                  MCC                 Max AUC")


# Log and save metrics
def log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev):
    metrics = [epoch, time, total_loss_train/total_train_size, total_loss_test/total_test_size, AUC_dev, PRC_dev,accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev]
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
    data_collator = PPICollator(max_length=max_seq_length, base_size=base_size, device=device, test=False)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4 if os.cpu_count() >= 8 else 0
    )
    
    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        # batch is alread on device
        batch_loss = trainer.train(batch, base_size, device, last_epoch)
        
        total_loss += batch_loss

    return total_loss, total_samples


# Test the model for one epoch
def test_epoch(dataset, emb_dict, tester, config, device, last_epoch):
    T, Y, S = [], [], []
    total_loss = 0
    total_samples = 0
    max_seq_length = config['model']['max_sequence_length']
    base_size = config['model']['base_size']
    batch_size = config['training']['batch_size']
    dataset = PPIDataset(dataset, emb_dict)
    total_samples += len(dataset)
    data_collator = PPICollator(max_length=max_seq_length, base_size=base_size, device=device, test=True)
    dev_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4 if os.cpu_count() >= 8 else 0
    )
    
    if last_epoch: 
        for proteinA, proteinB, labels in tqdm(dev_loader, desc="Testing", total=len(dev_loader)):
            dataset_batch = list(zip(proteinA, proteinB, labels))
            batch_loss, t, y, s = tester.test(dataset_batch, max_seq_length, base_size, last_epoch)
            T.extend(t)
            Y.extend(y)
            S.extend(s)
            total_loss += batch_loss
            
        return T, Y, S, total_loss, total_samples

    else:
        for proteinA, proteinB, labels in tqdm(dev_loader, desc="Testing", total=len(dev_loader)):
            dataset_batch = list(zip(proteinA, proteinB, labels))
            batch_loss, t, y, s = tester.test(dataset_batch, max_seq_length, base_size, last_epoch)
            T.extend(t)
            Y.extend(y)
            S.extend(s)
            total_loss += batch_loss
            
        return T, Y, S, total_loss, total_samples


# Train and validate the model across multiple epochs
def train_and_validate_model(config, trainer, tester, scheduler, model, device):
    max_AUC_dev = 0

    embedding_dict, train_data, valid_data, _ = get_data(config, device)

    start = timeit.default_timer()
    for epoch in range(1, config['training']['iteration'] + 1):
        if epoch != (config['training']['iteration']):
            total_loss_train, total_train_size = train_epoch(train_data, embedding_dict, trainer, config, device, last_epoch=False)
            T, Y, S, total_loss_test, total_test_size = test_epoch(valid_data, embedding_dict, tester, config, device, last_epoch=False)
            
            end = timeit.default_timer()
            time = end - start

            AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T,Y,S)
            
            if AUC_dev > max_AUC_dev:
                save_model(model, "output/model")
                max_AUC_dev = AUC_dev
            
            log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev)
            scheduler.step()
            plot(config['directories']['metrics_output'])
        
        if epoch == (config['training']['iteration']):
            total_loss_train, total_train_size = train_epoch(train_data, embedding_dict, trainer, config, device, last_epoch=True)
            
            T, Y, S, total_loss_test, total_test_size = test_epoch(valid_data, embedding_dict, tester, config, device, last_epoch=True)
            AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T,Y,S)
            
            end = timeit.default_timer()
            time = end - start
            log_and_save_metrics(epoch, time, total_loss_train, total_train_size, total_loss_test, total_test_size, AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc, max_AUC_dev)
            plot(config['directories']['metrics_output'])
            save_model(model, "output/model")


def evaluate(config, tester, device, batch_size=1):
    embedding_dict, _, _, test_data = get_data(config, device)

    T, Y, S, total_loss_test, total_test_size = test_epoch(test_data, embedding_dict, tester, config, last_epoch=True, batch_size=batch_size)
    AUC_dev, PRC_dev, accuracy, sensitivity, specificity, precision, f1, mcc = calculate_metrics(T, Y, S)
    
    # Print and write results to file
    test_results = [
        f'Test loss: {total_loss_test / total_test_size}',
        f'Test AUC: {AUC_dev}',
        f'Test PRC: {PRC_dev}',
        f'Test accuracy: {accuracy}',
        f'Test sensitivity: {sensitivity}',
        f'Test specificity: {specificity}',
        f'Test precision: {precision}',
        f'Test F1: {f1}',
        f'Test MCC: {mcc}'
    ]
    
    # Print results
    for result in test_results:
        print(result)
    
    # Calculate Expected Calibration Error
    ece = cal.get_ece(S, T)
    ece_result = f"Expected Calibration Error (ECE): {ece}"
    print(ece_result)
    test_results.append(ece_result)
    
    # Calculate uncertainty
    uncertainty = (1 - np.array(S)) * (np.array(S)) / 0.25

    for cutoff in [0.2, 0.4, 0.6, 0.8]:
        filtered_indices = uncertainty < cutoff
        T_filtered = np.array(T)[filtered_indices]
        Y_filtered = np.array(Y)[filtered_indices]
        true_positives = sum((T_filtered == 1) & (Y_filtered == 1))
        precision_filtered = precision_score(T_filtered, Y_filtered, zero_division=0)
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
    test_interactions['S'] = S
    test_interactions['uncertainty'] = uncertainty

    # Saving to TSV
    test_interactions.to_csv('evaluation_results.tsv', sep='\t', index=False)

    # when pauc has saving
    plot_roc_with_ci(Y, S, save_path='output/roc_curve.png')


# Save model state to file
def save_model(model, filename):
    torch.save(model.state_dict(), filename)