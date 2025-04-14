import os
import csv
import matplotlib.pyplot as plt


# Plot metrics across epochs
def plot(directory, train=True):
    train_loss = []
    test_loss = []
    AUC_dev = []
    PRC_dev = []
    acc = []
    sens = []
    spec = []
    prec = []
    f1 = []
    mcc = []
    max_auc = []
    
    # Open output and extract columns
    with open(directory, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)
        for row in reader:
            train_loss.append(float(row[2]))
            test_loss.append(float(row[3]))
            AUC_dev.append(float(row[4]))
            PRC_dev.append(float(row[5]))
            acc.append(float(row[6]))
            sens.append(float(row[7]))
            spec.append(float(row[8]))
            prec.append(float(row[9]))
            f1.append(float(row[10]))
            mcc.append(float(row[11]))
            max_auc.append(float(row[12]))

    # Get the current directory name
    current_dir = os.path.basename(os.getcwd())
    plt.close()
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    # Plot loss
    plt.plot(train_loss, label='Training Loss')
    if train:
        plt.plot(test_loss, label='Validation Loss')
    else: 
        plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.title(current_dir[:2])  # Print only the first two letters
    plt.ylabel('Loss')
    plt.grid()
    plt.tick_params(labelbottom=False)

    # Plotting metrics
    plt.subplot(2, 1, 2)
    plt.plot(AUC_dev, label='AUC')
    plt.plot(PRC_dev, label='PRC')
    plt.plot(acc, label='Accuracy')
    plt.plot(sens, label='Sensitivity')
    plt.plot(spec, label='Specificity')
    plt.plot(prec, label='Precision')    
    plt.plot(f1, label='F1')
    plt.plot(mcc, label='MCC')
    plt.plot(max_auc, label='Max Auc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid()

    plt.tight_layout()
    plt.savefig(current_dir[:2] + '.png', dpi=300)
    plt.close()