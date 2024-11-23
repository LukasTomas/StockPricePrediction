import torch
import torch.nn as nn
    

class StockPricePredictor(nn.Module):
    
    def __init__(self, input_size, hidden_size=128, num_layers=1, output_size=1):
        super(StockPricePredictor, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm_2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout_2 = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_N) = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        
        last_hidden_state = self.dropout(last_hidden_state)
        
        lstm_out_2, (h_n_2, c_N_2) = self.lstm_2(lstm_out)
        last_hidden_state_2 = lstm_out_2[:, -1, :]

        last_hidden_state_2 = self.dropout_2(last_hidden_state_2)

        linear_out = self.fc_2(last_hidden_state_2)
        out = torch.sigmoid(linear_out)
        return out
