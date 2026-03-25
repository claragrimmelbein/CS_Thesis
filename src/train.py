import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTransformerPredictor

def train_model():
    # load the prepared data
    X = torch.load('X_train.pt')
    y = torch.load('y_train.pt')


    # initialize the Model
    # input_dim=1, model_dim=32 (size of the internal vectors)
    model = SimpleTransformerPredictor(input_dim=3, model_dim=32)
    
    # define Loss Function (Mean Squared Error) and Optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training Loop
    epochs = 10 # start with 10 passes through the data
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() # Reset the gradients
        
        # forward pass: get predictions
        predictions = model(X)
        
        # calculate the error (how far off  from actual watch time)
        loss = criterion(predictions.squeeze(), y)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # save 
    torch.save(model.state_dict(), 'tiktok_model.pth')
    print("Training Complete! Model saved as 'tiktok_model.pth'")

if __name__ == "__main__":
    train_model()