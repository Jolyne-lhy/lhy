import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
device = get_default_device()

# 1D-CAE
class Encoder(nn.Module):
  def __init__(self, feature_size):
    super().__init__()
    self.conv1 = nn.Conv1d(feature_size, 32, kernel_size=7)
    self.conv2 = nn.Conv1d(32, 16, kernel_size=7)
    self.pool = nn.AvgPool1d(kernel_size=2)
        
  def forward(self, x):
    # input x: (batch_size, feature_num, seq_len)

    x = torch.tanh(self.conv1(x))

    x = self.pool(x)

    x = torch.tanh(self.conv2(x))

    x = self.pool(x)
    return x
    
class Decoder(nn.Module):
  def __init__(self, feature_size, seq_len):
    super().__init__()
    self.t_conv1 = nn.ConvTranspose1d(16, 16, kernel_size=7, stride=4)
    self.t_conv2 = nn.ConvTranspose1d(16, 32, kernel_size=7, stride=4)
    self.pool = nn.AvgPool1d(kernel_size=2)
    self.fc = nn.Linear(8672, feature_size * seq_len)
        
  def forward(self, x):
    x = torch.tanh(self.t_conv1(x))

    x = self.pool(x)

    x = torch.tanh(self.t_conv2(x))

    x = self.pool(x)

    x = torch.reshape(x, (x.shape[0], -1))

    x = self.fc(x)
    return x


class CNN(nn.Module):
    def __init__(self, feature_size, seq_len):
        super().__init__()
        self.encoder = Encoder(feature_size)
        self.decoder = Decoder(feature_size, seq_len)

    def training_step(self, batch):
        z = self.encoder(batch)
        w = self.decoder(z)
        x = torch.reshape(batch, (batch.shape[0], -1))
        mse = nn.MSELoss()
        loss = mse(w, x)
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            z = self.encoder(batch)
            w = self.decoder(z)
        x = torch.reshape(batch, (batch.shape[0], -1))
        mse = nn.MSELoss()
        loss = mse(w, x)
        return {'val_loss': loss}
            
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
        
    def epoch_end(self, epoch, result1, result2):
        print("Epoch [{}], trn_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result1['val_loss'], result2['val_loss']))
    
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(to_device(batch,device)) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, trn_loader, val_loader, save_path, opt_func=torch.optim.Adam):
    history = []
    learning_rate = 1e-3
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()),
                        lr=learning_rate,
                        betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=0.)
    min_loss = float('inf')
    epoch_cnt = 0
    for epoch in range(epochs):
        model.train()
        for [batch] in trn_loader:
            batch=to_device(batch,device)
            
            #Train
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result_trn = evaluate(model, trn_loader)
        result_val = evaluate(model, val_loader)
        model.epoch_end(epoch+1, result_trn, result_val)
        history.append({'trn_loss': result_trn['val_loss'], 'val_loss': result_val['val_loss']})

        # Early stopping
        if result_val['val_loss'] < min_loss or epoch == epochs-1:
            torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict()
            }, f'{save_path}')
            min_loss = result_val['val_loss']
            epoch_cnt = 0
        else:
            epoch_cnt += 1

        if epoch_cnt == 10:
            print("Early stopping")
            break

    return history
    
def testing(model, test_loader):
    model.eval()
    results=[]
    y_pred=[]
    y_true=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        with torch.no_grad():
            w=model.decoder(model.encoder(batch))
        w=torch.reshape(w, (w.shape[0], batch.shape[1], batch.shape[2]))
        results.append(torch.median((batch-w)**2,dim=2).values)
        # results.append(torch.sum((torch.median((batch-w)**2,dim=2).values), dim=1))
        # results.append(torch.mean((torch.median((batch-w)**2,dim=2).values), dim=1))
        y_pred.append(w)
        y_true.append(batch)
    return results, y_pred, y_true