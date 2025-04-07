from dataapi import *
from model import *
import torch
# from torchmetrics.classification import BinaryAccuracy
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    data_instance = DataApi([2017, 2018, 2019, 2020, 2021, 2022])
    seasons_data: pd.DataFrame = data_instance.run()
    seasons_data = process_data(seasons_data)
    seasons_data['team_winner'] = seasons_data['team_winner'].astype(int)
    print(seasons_data.info())

    model = NBAtorchnnModel().to(device)

    y = seasons_data['team_winner'].to_numpy()
    X = seasons_data.drop(columns=['team_winner']).to_numpy()

    # print("X type:", type(X), "dtype:", X.dtype, "shape:", X.shape)
    # print("y type:", type(y), "dtype:", y.dtype, "shape:", y.shape)

    # print(X.shape)

    # Create training data, and create a temp in order to create training/validation data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.2, random_state=43, stratify= y)

    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=.5, random_state = 43, stratify= y_temp)

    train_dataset = convert_to_tensor_dataset(X_train, y_train)
    test_dataset = convert_to_tensor_dataset(X_test, y_test)
    val_dataset = convert_to_tensor_dataset(X_val, y_val)

    torch.manual_seed(42)

    # Helps the data be loaded in batches smoothly
    train_loader = DataLoader(train_dataset, batch_size = 50, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 50, shuffle = True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    # metric = BinaryAccuracy().to(device)

    #Initialize Variables for EarlyStopping
    best_loss: float = float('inf')
    best_model_weights = None
    best_val_acc: float = float('inf')
    patience: int = 10
    best_epoch: int = 0

    num_epochs = 100
    final_epochs = 0

    acc_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        val_loss, correct_val, total_val = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) 
            # Forward pass
            y_logits = model(X_batch).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))

            # Compute loss
            loss = criterion(y_logits, y_batch)
            train_loss += loss.item()

            # Accuracy
            correct_train += (y_pred == y_batch).sum().item()
            total_train += y_batch.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = 100 * correct_train / total_train

            # Validation phase
            model.eval()
            

            with torch.inference_mode():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_logits = model(X_batch).squeeze()
                    val_pred = torch.round(torch.sigmoid(val_logits)).squeeze()
                    # metric.update(val_pred, y_batch)

                    loss = criterion(val_logits, y_batch)
                    val_loss += loss.item()

                    correct_val += (val_pred == y_batch).sum().item()
                    total_val += y_batch.size(0)

            val_acc = 100 * correct_val / total_val

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience = 10  # Reset patience counter
        else:
            patience -= 1
        # Print the stats before breaking out of the loop
        if patience == 0:
            print(f"Epoch {best_epoch}: "
            f"Best Val Loss: {best_loss/len(val_loader):.5f}, Val Acc: {best_val_acc:.2f}%")
            break


        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.5f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss/len(val_loader):.5f}, Val Acc: {val_acc:.2f}%")
            
        acc_list.append(val_acc)
        # metric.reset
        final_epochs += 1

    
    epochs = range(1, final_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png", dpi=300, bbox_inches='tight')
        
if __name__ == "__main__":
    main()
