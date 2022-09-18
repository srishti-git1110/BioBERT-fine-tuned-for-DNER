def evaluate_test_data(batch_generator, classifier, compute_accuracy, criterion, train_state):
  classifier.eval()
  running_mean_loss = 0.0
  running_mean_accuracy = 0.0
  for batch_idx, batch_dict in enumerate(batch_generator):
    y_pred = classifier(batch_dict['x'])
    loss = criterion(y_pred, batch_dict['y'])
    batch_loss = loss.item()
    batch_acc = compute_accuracy(y_pred, batch_dict['y'])
    
    running_mean_loss += (batch_loss - running__mean_loss) / (batch_idx + 1)
    running_mean_accuracy += (batch_loss - running_mean_accuracy) / (batch_idx + 1)

  train_state['test_loss'] = running_mean_loss
  train_state['test_acc'] = running_mean_accuracy
