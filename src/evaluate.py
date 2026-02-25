import torch
import pandas as pd
from model import SimpleTransformerPredictor

def evaluate():
    # load model and data
    model = SimpleTransformerPredictor(input_dim=1, model_dim=32)
    model.load_state_dict(torch.load('tiktok_model.pth'))
    model.eval() # set to evaluation mode
    
    X = torch.load('X_train.pt').unsqueeze(-1)
    y = torch.load('y_train.pt')

    # sample sequences to test
    with torch.no_grad():
        predictions = model(X)
    
    # create a comparison table
    results = pd.DataFrame({
        'Actual_Watch_Time': y.numpy(),
        'Predicted_Watch_Time': predictions.squeeze().numpy()
    })
    
    # show the first 10 predictions
    print("Sample Predictions vs Actual:")
    print(results.head(10))
    
    # save results for thesis charts
    results.to_csv('model_results_comparison.csv', index=False)
    print("\nResults saved to 'model_results_comparison.csv'")

if __name__ == "__main__":
    evaluate()