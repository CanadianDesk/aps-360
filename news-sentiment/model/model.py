# This file contains the new model for the news sentiment analysis project
# It uses a pre-trained GloVe word embedding model as opposed to training
# a new model using Word2Vec from scratch for embedding the headlines
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import torchtext
torchtext.disable_torchtext_deprecation_warning()  # silencce the annoying ahh warning
import torchtext.vocab as vocab


# downloading the tokenizer
nltk.download('punkt', quiet=False)
# nltk.download('punkt_tab', quiet=False)
gemini = True


class HeadlineDataset(Dataset):
  def __init__(self, headlines, labels, glove_vectors, max_length=128):
    self.headlines = headlines
    self.labels = labels
    self.glove_vectors = glove_vectors
    self.max_length = max_length
    self.vector_size = self.glove_vectors['the'].shape[0]  # get dimension from a common word

  def __len__(self):
    return len(self.headlines)

  def __getitem__(self, idx):
    headline = self.headlines[idx]
    sentiment = self.labels[idx]
    
    # tokenize
    tokens = word_tokenize(headline.lower())
    
    # embed using GloVe
    vectors = []
    for token in tokens[:self.max_length]:
      if token in self.glove_vectors:
        vectors.append(self.glove_vectors[token].numpy())
      else:
        vectors.append(np.zeros(self.vector_size))
    
    # pad if needed
    if len(vectors) < self.max_length:
      vectors.extend([np.zeros(self.vector_size)] * (self.max_length - len(vectors)))
    
    # convert to tensors
    vectors = torch.FloatTensor(vectors)
    sentiment = torch.FloatTensor([sentiment])
    
    return vectors, sentiment


# load pre-trained GloVe vectors
def load_glove_vectors(dim=200):
    return vocab.GloVe(name='6B', dim=dim)

class BiLSTMSentiment(nn.Module):
  '''
  Class for the BiLSTM model.
  '''  
  def __init__(self, embedding_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.55):
    super(BiLSTMSentiment, self).__init__()
    
    self.lstm = nn.LSTM(embedding_dim, 
                        hidden_dim,
                        num_layers=num_layers,
                        bidirectional=True,
                        dropout=dropout if num_layers > 1 else 0,
                        batch_first=True)
    
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
  def forward(self, text):
    # text shape: [batch size, sequence length, embedding dim]

    lstm_output, (hidden, cell) = self.lstm(text) # lstm layers
    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) # dropout layer
    
    # output layer is tanh to get output in range [-1, 1]
    return torch.tanh(self.fc(hidden))
  

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=50, patience=6, csv_index=0):
  '''
  Funciton for training the model. 
  '''
  # load onto cuda or mps if available
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model.to(device)
  print("Training model on device: ", device)
  
  best_val_loss = float('inf')
  epochs_no_improve = 0  # Counter for early stopping
  early_stop = False
  
  # For tracking metrics
  train_losses = []
  val_losses = []
  epochs_completed = 0

  for epoch in range(num_epochs):
    
    model.train()
    train_loss = 0

    # training loop
    for batch_idx, (headlines, labels) in enumerate(train_loader):
        
      # move everything to gpu
      headlines, labels = headlines.to(device), labels.to(device)

      # forward pass
      optimizer.zero_grad()
      outputs = model(headlines)
      loss = loss_fn(outputs, labels)

      # backward pass
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
      for headlines, labels in val_loader:

        # move everything to gpu
        headlines, labels = headlines.to(device), labels.to(device)

        outputs = model(headlines)
        loss = loss_fn(outputs, labels)
        val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    epochs_completed = epoch + 1

    print("\n\n" + "="*10 + f" Epoch {epoch+1} " + "="*10)
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    # Check if this is the best model
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      state_folder = 'gemini-states' if gemini else 'states'
      torch.save(model.state_dict(), f"./{state_folder}/model_{csv_index}.pt")
      print(f'\t Best model saved at epoch {epoch+1}')
      epochs_no_improve = 0  # Reset counter
    else:
      epochs_no_improve += 1
      print(f'\t No improvement for {epochs_no_improve} epochs')
      
    # Check early stopping condition
    if epochs_no_improve >= patience:
      print(f'\n Early stopping triggered after {epoch+1} epochs')
      early_stop = True
      break
  
  # Create and save the loss plot with actual epochs completed
  epochs = list(range(1, epochs_completed+1))
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, train_losses, 'b-', label='Training Loss')
  plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
  plt.title('Training and Validation Loss Over Time')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plot_folder = 'gemini-plots' if gemini else 'plots'
  plt.savefig(f'./{plot_folder}/loss_plot_{csv_index}.png')
  plt.close()
  
  if early_stop:
    print("Training stopped early due to no improvement in validation loss")
  print(f"Loss plot saved to 'loss_plot.png'")
  print(f"Best validation loss: {best_val_loss:.4f}")
  
  return train_losses, val_losses, best_val_loss  


def test_model(model, test_loader, loss_fn, csv_index=0):
  '''
  Function for testing the model.
  '''
  # load onto cuda or mps if available
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  # load best trained model
  state_folder = "gemini-states" if gemini else "states"
  model.load_state_dict(torch.load(f'./{state_folder}/model_{csv_index}.pt'))
  model.to(device)
  model.eval()

  predictions = []
  actual = []
  test_loss = 0

  with torch.no_grad():
    for headlines, labels in test_loader:
      headlines, labels = headlines.to(device), labels.to(device)

      outputs = model(headlines)
      loss = loss_fn(outputs, labels)

      test_loss += loss.item()

      predictions.extend(outputs.squeeze().cpu().numpy())
      actual.extend(labels.squeeze().cpu().numpy())

  test_loss /= len(test_loader)

  # get metrics
  mse = np.mean((np.array(predictions) - np.array(actual)) ** 2)
  mae = np.mean(np.abs(np.array(predictions) - np.array(actual)))

  print(f'Test Loss: {test_loss:.4f}')
  print(f'MSE: {mse:.4f}')
  print(f'MAE: {mae:.4f}')
  
  return predictions, actual, test_loss, mae

def run_experiments():
  # check if record.csv exists and load it
  csv_file_path = './gemini_record.csv' if gemini else './record.csv'
  if not os.path.exists(csv_file_path):
    print(f"Error: ./{csv_file_path} not found.")
    return
  
  record_df = pd.read_csv(csv_file_path)

  print("record df", record_df)
  
  # find the first row with missing evaluation metrics
  next_row = None
  for idx, row in record_df.iterrows():
    if pd.isna(row['val_loss']) or pd.isna(row['test_loss']) or pd.isna(row['mae']):
      next_row = idx
      break
  
  if next_row is None:
    print("All hyperparameter configurations have already been evaluated.")
    return
  
  print(f"Starting hyperparameter evaluations from row {next_row}")

  # in the new model, we are replacing with GloVe vectors
  glove_vectors = load_glove_vectors(200)  
  
  # process each row that needs evaluation
  for idx in range(next_row, len(record_df)):
    row = record_df.iloc[idx]
    
    
    # extract hyperparameters
    lr = row['lr']
    weight_decay = row['weight_decay']
    batch_size = int(row['batch_size'])
    dropout = row['dropout']
    hidden_dim = int(row['hidden_dim'])
    num_layers = int(row['num_layers'])
    
    print(f"\n\n{'='*50}")
    print(f"Evaluating configuration {idx+1}/{len(record_df)}:")
    print(f"lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}, dropout={dropout}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print('='*50)
    
    # loading the data
    label_file_name = "gemini" if gemini else "data"
    df = pd.read_csv(f'../data-collection/{label_file_name}.csv')

    # tokenize the headlines for word2vec to process
    tokenized_headlines = [word_tokenize(str(headline).lower()) for headline in df['headline']]  

    # train word2vec model
    # word2vec_model = Word2Vec(sentences=tokenized_headlines, vector_size=200, window=4, min_count=1, workers=6, epochs=15, sg=1)



    # split data
    X_train, X_temp, y_train, y_temp = train_test_split(df['headline'], df['label'], test_size=0.3, random_state=7415)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=7415)

    # dataset creation
    train_dataset = HeadlineDataset(X_train.values, y_train.values, glove_vectors)
    valid_dataset = HeadlineDataset(X_valid.values, y_valid.values, glove_vectors)
    test_dataset = HeadlineDataset(X_test.values, y_test.values, glove_vectors)

    # dataloader creation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # initialize model with current hyperparameters
    embedding_dim = 200 # hard coded for now but probably should be dynamic
    model = BiLSTMSentiment(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    # loss function & optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Train model with fixed patience of 6
    _, _, best_val_loss = train_model(model, train_loader, valid_loader, 
                                      optimizer, loss_fn, num_epochs=50, patience=6, csv_index=idx)

    # Test the model and get metrics
    _, _, test_loss, mae = test_model(model, test_loader, loss_fn, idx)
    
    # Update the CSV with results
    record_df.at[idx, 'val_loss'] = round(best_val_loss, 6)
    record_df.at[idx, 'test_loss'] = round(test_loss, 6)
    record_df.at[idx, 'mae'] = round(mae, 6)
    
    # Save after each evaluation to preserve progress
    record_df.to_csv(csv_file_path, index=False)
    print(f"\nResults for configuration {idx+1}:")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Results saved to {csv_file_path}")
  
  print("\nAll hyperparameter configurations have been evaluated!")


def perform_inference(model_number, headline_text, gemini=True):
  
  # device
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  

  csv_file_path = './gemini_record.csv' if gemini else './record.csv'
  if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Record file not found: {csv_file_path}")
  
  record_df = pd.read_csv(csv_file_path)
  model_params = record_df.iloc[model_number]
  

  hidden_dim = int(model_params['hidden_dim'])
  num_layers = int(model_params['num_layers'])
  dropout = model_params['dropout']
  

  glove_vectors = load_glove_vectors(200)
  embedding_dim = 200 
  
  model = BiLSTMSentiment(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
  
  state_folder = "gemini-states" if gemini else "states"
  model_path = f'./{state_folder}/model_{model_number}.pt'
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
  
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  
  max_length = 128 
  tokens = word_tokenize(headline_text.lower())
  
  vectors = []
  for token in tokens[:max_length]:
    if token in glove_vectors:
      vectors.append(glove_vectors[token].numpy())
    else:
      vectors.append(np.zeros(embedding_dim))
  
  if len(vectors) < max_length:
    vectors.extend([np.zeros(embedding_dim)] * (max_length - len(vectors)))
  
  headline_tensor = torch.FloatTensor(vectors).unsqueeze(0).to(device)
  
  with torch.no_grad():
    prediction = model(headline_tensor)
  
  return prediction.item()

run_experiments()