# This file contains the new model for the news sentiment analysis project
# It uses a pre-trained GloVe word embedding model as opposed to training
# a new model using Word2Vec from scratch for embedding the headlines
# This is the Optuna version of the model
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
torchtext.disable_torchtext_deprecation_warning()  # silence the annoying warning
import torchtext.vocab as vocab
import optuna  # Add Optuna for hyperparameter optimization
import json


# downloading the tokenizer
nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=False)
gemini = True

# Define paths
TRIALS_CSV = './optuna_trials.csv' if gemini else './optuna_trials_ngemini.csv'
HISTORY_CSV = './optuna-history.csv' if gemini else './optuna-history-ngemini.csv'
STATE_FOLDER = './optuna-gemini-states' if gemini else 'states'
PLOT_FOLDER = './optuna-gemini-plots' if gemini else 'plots'
DATA_FILE = '../data-collection/gemini.csv' if gemini else '../data-collection/data.csv'

# Paths for inference
ARTICLES_DIR = '../data-collection/articles'
LABELED_ARTICLES_DIR = '../data-collection/labeled-articles'

# Ensure folders exist
os.makedirs(STATE_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)


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
    
    # convert to tensors - fixed to use np.array first
    vectors = torch.FloatTensor(np.array(vectors))
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


# Function to save validation loss history to CSV
def save_loss_history(trial_number, train_losses, val_losses):
    """Save the training and validation loss history for a trial"""
    # Convert lists to strings for storage in CSV
    train_losses_str = json.dumps(train_losses)
    val_losses_str = json.dumps(val_losses)
    
    history_data = {
        'trial_number': trial_number,
        'train_losses': train_losses_str,
        'val_losses': val_losses_str
    }
    
    history_df = pd.DataFrame([history_data])
    
    if os.path.exists(HISTORY_CSV):
        # Check if file is empty
        if os.path.getsize(HISTORY_CSV) == 0:
            # Create new file with header
            history_df.to_csv(HISTORY_CSV, index=False)
            return
            
        try:
            # Check if this trial already exists
            existing_df = pd.read_csv(HISTORY_CSV)
            if trial_number in existing_df['trial_number'].values:
                # Update existing row
                existing_df.loc[existing_df['trial_number'] == trial_number] = history_df.iloc[0]
                existing_df.to_csv(HISTORY_CSV, index=False)
            else:
                # Append new row
                history_df.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # Handle corrupted files by overwriting
            print(f"Warning: CSV file {HISTORY_CSV} exists but is invalid. Creating a new file.")
            history_df.to_csv(HISTORY_CSV, index=False)
    else:
        # Create new file
        history_df.to_csv(HISTORY_CSV, index=False)

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=50, patience=6, trial_number=None):
  '''
  Function for training the model. 
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
      # Save model with trial number if provided
      model_path = f"./{STATE_FOLDER}/model_trial_{trial_number}.pt" if trial_number is not None else f"./{STATE_FOLDER}/model_best.pt"
      torch.save(model.state_dict(), model_path)
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
  
  # Save the loss history for this trial
  if trial_number is not None:
    save_loss_history(trial_number, train_losses, val_losses)
  
  # Create and save the loss plot with actual epochs completed
  if trial_number is not None:
    epochs = list(range(1, epochs_completed+1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Trial {trial_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{PLOT_FOLDER}/loss_plot_trial_{trial_number}.png')
    plt.close()
  
  if early_stop:
    print("Training stopped early due to no improvement in validation loss")
  print(f"Best validation loss: {best_val_loss:.4f}")
  
  return train_losses, val_losses, best_val_loss  


def test_model(model, test_loader, loss_fn, trial_number=None):
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
  model_path = f'./{STATE_FOLDER}/model_trial_{trial_number}.pt' if trial_number is not None else f'./{STATE_FOLDER}/model_best.pt'
  model.load_state_dict(torch.load(model_path))
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


# CSV for optuna trials management
def load_previous_trials():
    """Load previous trials from CSV file"""
    if not os.path.exists(TRIALS_CSV):
        return []
    
    # Check if file is empty
    if os.path.getsize(TRIALS_CSV) == 0:
        return []
    
    try:
        trials_df = pd.read_csv(TRIALS_CSV)
        previous_trials = []
        
        for _, row in trials_df.iterrows():
            params = {
                'lr': row['lr'],
                'weight_decay': row['weight_decay'],
                'batch_size': int(row['batch_size']),
                'dropout': row['dropout'],
                'hidden_dim': int(row['hidden_dim']),
                'num_layers': int(row['num_layers'])
            }
            previous_trials.append({
                'trial_number': int(row['trial_number']),
                'params': params,
                'value': row['val_loss'],
                'test_loss': row['test_loss'],
                'mae': row['mae']
            })
        
        return previous_trials
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # Handle both empty file and parsing errors
        print(f"Warning: CSV file {TRIALS_CSV} exists but is empty or invalid. Starting fresh.")
        return []
    

def save_trial(trial_number, params, val_loss, test_loss, mae):
    """Save trial results to CSV file"""
    trial_data = {
        'trial_number': trial_number,
        'lr': params['lr'],
        'weight_decay': params['weight_decay'],
        'batch_size': params['batch_size'],
        'dropout': params['dropout'],
        'hidden_dim': params['hidden_dim'],
        'num_layers': params['num_layers'],
        'val_loss': val_loss,
        'test_loss': test_loss,
        'mae': mae
    }
    
    trial_df = pd.DataFrame([trial_data])
    
    if os.path.exists(TRIALS_CSV):
        # Check if file is empty
        if os.path.getsize(TRIALS_CSV) == 0:
            # Create new file with header
            trial_df.to_csv(TRIALS_CSV, index=False)
            return
            
        try:
            # Check if this trial already exists
            existing_df = pd.read_csv(TRIALS_CSV)
            if trial_number in existing_df['trial_number'].values:
                # Update existing row
                existing_df.loc[existing_df['trial_number'] == trial_number] = trial_df.iloc[0]
                existing_df.to_csv(TRIALS_CSV, index=False)
            else:
                # Append new row
                trial_df.to_csv(TRIALS_CSV, mode='a', header=False, index=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # Handle corrupted files by overwriting
            print(f"Warning: CSV file {TRIALS_CSV} exists but is invalid. Creating a new file.")
            trial_df.to_csv(TRIALS_CSV, index=False)
    else:
        # Create new file
        trial_df.to_csv(TRIALS_CSV, index=False)

def create_datasets_and_loaders(batch_size, glove_vectors):
    """Create datasets and dataloaders for training"""
    # Load data
    df = pd.read_csv(DATA_FILE)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(df['headline'], df['label'], test_size=0.3, random_state=7415)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=7415)
    
    # Create datasets
    train_dataset = HeadlineDataset(X_train.values, y_train.values, glove_vectors)
    valid_dataset = HeadlineDataset(X_valid.values, y_valid.values, glove_vectors)
    test_dataset = HeadlineDataset(X_test.values, y_test.values, glove_vectors)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader


def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    
    print(f"\n\n{'='*50}")
    print(f"Trial {trial.number}:")
    print(f"lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}, dropout={dropout}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    print('='*50)
    
    # Load GloVe vectors
    glove_vectors = load_glove_vectors(200)
    
    # Create datasets and loaders
    train_loader, valid_loader, test_loader = create_datasets_and_loaders(batch_size, glove_vectors)
    
    # Create model
    embedding_dim = 200  # Fixed dimension for GloVe vectors
    model = BiLSTMSentiment(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    # Train model
    _, _, best_val_loss = train_model(model, train_loader, valid_loader, 
                                    optimizer, loss_fn, num_epochs=50, patience=6, trial_number=trial.number)
    
    # Test model
    _, _, test_loss, mae = test_model(model, test_loader, loss_fn, trial_number=trial.number)
    
    # Save trial results
    save_trial(trial.number, 
               {'lr': lr, 'weight_decay': weight_decay, 'batch_size': batch_size, 
                'dropout': dropout, 'hidden_dim': hidden_dim, 'num_layers': num_layers}, 
               best_val_loss, test_loss, mae)
    
    return best_val_loss


def run_optuna_optimization(n_trials=10):
  """Run Optuna hyperparameter optimization with CSV persistence"""
  # Load previous trials
  previous_trials = load_previous_trials()
  
  # Create study
  study = optuna.create_study(direction='minimize')
  
  # Add previous trials
  for trial_info in previous_trials:
    study.add_trial(
      optuna.trial.create_trial(
        params=trial_info['params'],
        value=trial_info['value']
      )
    )
  
  print(f"Loaded {len(previous_trials)} previous trials")
  if previous_trials:
    best_trial = min(previous_trials, key=lambda x: x['value'])
    print(f"Best previous trial: {best_trial['trial_number']} with val_loss: {best_trial['value']:.6f}")
    print(f"Parameters: {best_trial['params']}")

  # Run optimization
  study.optimize(objective, n_trials=n_trials)
  
  # Print results
  print("\nStudy complete!")
  print(f"Best trial: {study.best_trial.number}")
  print(f"Best value: {study.best_value:.6f}")
  print(f"Best parameters: {study.best_params}")
  
  # Create importance plot
  if len(study.trials) >= 5:  # Need enough trials for meaningful plot
    try:
      importance = optuna.importance.get_param_importances(study)
      importance_df = pd.DataFrame(importance.items(), columns=['Parameter', 'Importance'])
      importance_df = importance_df.sort_values('Importance', ascending=False)
      
      plt.figure(figsize=(10, 6))
      plt.bar(importance_df['Parameter'], importance_df['Importance'])
      plt.title('Hyperparameter Importance')
      plt.xlabel('Parameter')
      plt.ylabel('Importance')
      plt.xticks(rotation=45)
      plt.tight_layout()
      plt.savefig(f'./{PLOT_FOLDER}/parameter_importance.png')
      plt.close()
      print(f"Parameter importance plot saved to '{PLOT_FOLDER}/parameter_importance.png'")
    except Exception as e:
      print(f"Could not create importance plot: {e}")
  
  return study


def perform_inference(trial_number, headline_text):
  """Perform inference using a trained model from a specific trial"""
  # Load trial data
  if not os.path.exists(TRIALS_CSV):
    raise FileNotFoundError(f"Trials file not found: {TRIALS_CSV}")
  
  trials_df = pd.read_csv(TRIALS_CSV)
  if trial_number not in trials_df['trial_number'].values:
    raise ValueError(f"Trial {trial_number} not found in trials file")
  
  trial_data = trials_df[trials_df['trial_number'] == trial_number].iloc[0]
  
  # Get device
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  
  # Load model parameters
  hidden_dim = int(trial_data['hidden_dim'])
  num_layers = int(trial_data['num_layers'])
  dropout = trial_data['dropout']
  
  # Load GloVe vectors
  glove_vectors = load_glove_vectors(200)
  embedding_dim = 200
  
  # Create model
  model = BiLSTMSentiment(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
  
  # Load model state
  model_path = f'./{STATE_FOLDER}/model_trial_{trial_number}.pt'
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
  
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  
  # Process headline
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
  
  headline_tensor = torch.FloatTensor(np.array(vectors)).unsqueeze(0).to(device)
  
  with torch.no_grad():
    prediction = model(headline_tensor)
  
  return prediction.item()


# Function to generate loss plots from history CSV
def generate_loss_plots_from_history():
  """Generate plots from the saved loss history CSV"""
  if not os.path.exists(HISTORY_CSV):
    print(f"History CSV file not found: {HISTORY_CSV}")
    return
  
  history_df = pd.read_csv(HISTORY_CSV)
  for _, row in history_df.iterrows():
    trial_number = int(row['trial_number'])
    train_losses = json.loads(row['train_losses'])
    val_losses = json.loads(row['val_losses'])
    
    # Create plot
    epochs = list(range(1, len(val_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Trial {trial_number}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{PLOT_FOLDER}/history_loss_plot_trial_{trial_number}.png')
    plt.close()
  
  print(f"Generated {len(history_df)} loss plots from history")



def get_color(value):
  # If value is very close to zero, use default color
  if abs(value) < 0.05:
    return "\033[0m"  # Reset to default
  
  # For positive values (green gradient)
  elif value > 0:
    # Calculate intensity (0 to 1), capped at 0.5
    intensity = min(value / 0.5, 1.0)
    # Darker green has less red and blue components
    # 255 -> 0 for intensity 0 -> 1
    brightness = int(255 * (1 - intensity))
    return f"\033[38;2;{brightness};255;{brightness}m"
  
  # For negative values (red gradient)
  else:
    # Calculate intensity (0 to 1), capped at 0.5
    intensity = min(abs(value) / 0.5, 1.0)
    # Darker red has less green and blue components
    # 255 -> 0 for intensity 0 -> 1
    brightness = int(255 * (1 - intensity))
    return f"\033[38;2;255;{brightness};{brightness}m"
  

def label_csv_headlines_with_inference():
  '''
  This function goes thorugh all CSV files in ARTICLES_DIR and performs inference on each headline.
  It then saves the CSV with the label column filled in with values
  '''
  trial_number = 25

  # load the model
  if not os.path.exists(TRIALS_CSV):
    raise FileNotFoundError(f"Trials file not found: {TRIALS_CSV}")
  
  trials_df = pd.read_csv(TRIALS_CSV)
  if trial_number not in trials_df['trial_number'].values:
    raise ValueError(f"Trial {trial_number} not found in trials file")
  
  trial_data = trials_df[trials_df['trial_number'] == trial_number].iloc[0]
  
  # Get device
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  
  print(f"Using device: {device}") 
  # Load model parameters
  hidden_dim = int(trial_data['hidden_dim'])
  num_layers = int(trial_data['num_layers'])
  dropout = trial_data['dropout']
  
  # Load GloVe vectors
  glove_vectors = load_glove_vectors(200)
  embedding_dim = 200
  
  # Create model
  model = BiLSTMSentiment(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)  

  # Load model state
  model_path = f'./{STATE_FOLDER}/model_trial_{trial_number}.pt'
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")  
  
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()  

  max_length = 128

  for filename in os.listdir(ARTICLES_DIR):
    if filename.endswith('.csv'):
      print(f"Processing CSV file: {filename}")

      # Load CSV
      df = pd.read_csv(os.path.join(ARTICLES_DIR, filename))

      # Perform inference on each headline
      for index, row in df.iterrows():
        headline = row['headline']

        # process headline
        tokens = word_tokenize(headline.lower())
        vectors = []
        for token in tokens[:max_length]:
          if token in glove_vectors:
            vectors.append(glove_vectors[token].numpy())
          else:
            vectors.append(np.zeros(embedding_dim))
        
        if len(vectors) < max_length:
          vectors.extend([np.zeros(embedding_dim)] * (max_length - len(vectors)))

        headline_tensor = torch.FloatTensor(np.array(vectors)).unsqueeze(0).to(device)

        with torch.no_grad():
          prediction = model(headline_tensor)

        df.at[index, 'label'] = prediction.item()
        print(f"\rProcessing {filename.replace('.csv', '')}: {index+1}/{len(df)}", end='', flush=True)
      
      print(f"\nComplete. Saving CSV file: {LABELED_ARTICLES_DIR}/{filename}")

      # Save CSV with label column filled in
      df.to_csv(os.path.join(LABELED_ARTICLES_DIR, filename), index=False)
    else:

      print(f"Non-CSV file found: {filename}")
      continue
  return

if __name__ == "__main__":
  # Run optimization with 50 trials
  # run_optuna_optimization(n_trials=50)

  # best trial number:
  # best_trial = 25

  # strings = [
  #   "Stock market soars, Tesla expected to beat earnings.",
  #   "Jonah Diamond Inc. fulfills record high deliveries in Q4",
  #   "Apple Inc. stock soars after iPhone 14 launch",
  #   "Kyle Dyer Corp. sued for over $30 million over drone death",
  #   "Exxon Mobil Corporation CEO Stephen Hahn to step down after scandals and controversies",
  #   "Charles Inc. CEO resigns amid sexual misconduct allegations",
  #   "Bhandafri & Co. CEO shot in back by police",
  #   "Tesla Inc. CEO Elon Musk to step down as company's chairman to pursue dream of theater",
  #   "Amazon CEO Mark Zuckerberg demoted to Vice President of Corporate Affairs",
  #   "Real estate firms to pay $1.5 billion in fines over allegedly illegal rent-to-own",
  #   "George Li Inc. purchases Earth at a low cost $1 million", 
  #   "Darsan Qi Corp. secures monopoly on Chinese pharmaceuticals",
  #   "Mark Carney and the Liberals win Canadian election",
  #   "Pierre Poillievre and the Conservatives win Canadian election",
  #   "Elon Musk might be a Nazi"
  # ]

  # for string in strings:
  #   inference = perform_inference(best_trial, string)
  #   color = get_color(inference)
  #   reset = "\033[0m"
  #   print(f"{string}: {color}{inference:.3f}{reset}")

  label_csv_headlines_with_inference()