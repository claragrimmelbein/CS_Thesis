import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformerPredictor

def train_model():
    # load the prepared data
    X = torch.load('X_train.pt')
    y = torch.load('y_train.pt')

    # reshape X to (Batch_Size, Sequence_Length, Input_Dim)
    # our window size was 10, and we are using 1 feature (watch time)
    X = X.unsqueeze(-1) 

    # initialize the Model
    # input_dim=1, model_dim=32 (this is the size of the internal vectors)
    model = SimpleTransformerPredictor(input_dim=1, model_dim=32)
    
    # define Loss Function (Mean Squared Error) and Optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training Loop
    epochs = 10 # We'll start with 10 passes through the data
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() # Reset the gradients
        
        # forward pass: Get predictions
        predictions = model(X)
        
        # calculate the error (how far off we are from actual watch time)
        loss = criterion(predictions.squeeze(), y)
        
        # backward pass: The model learns from its mistakes
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # save the trained "Brain"
    torch.save(model.state_dict(), 'tiktok_model.pth')
    print("Training Complete! Model saved as 'tiktok_model.pth'")

if __name__ == "__main__":
    train_model()