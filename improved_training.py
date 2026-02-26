# Add this as a new cell AFTER cell 17 to see detailed metrics and continue training

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_model(model, test_loader, device):
    """Detailed evaluation with confusion matrix"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, 
                                target_names=['N', 'S', 'V', 'F']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['N', 'S', 'V', 'F'],
                yticklabels=['N', 'S', 'V', 'F'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return all_preds, all_targets

# Evaluate current model
evaluate_model(model, test_loader, device)
