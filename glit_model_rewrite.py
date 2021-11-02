import numpy as np
import pytorch_lightning as pl

import pickle
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

class ECFP_DP_MLP(pl.LightningModule):
    '''
    Neural Network with multiple triple layer MLPs

    Initial MLP Inputs: ECFP (2048 units), Diffusion Profile (29959 units)
    Prediction MLP Inputs: ECFP (256 units), Diffusion Profile (500 units)

    #TODO Currently all dimensions are hardcoded, will need to change this eventually to be a variable so it can be programmatically managed 
    '''
    
    def __init__(self):
        super().__init__()

        # Batch Normalization Layer
        self.batch_norm_ecfp = nn.BatchNorm1d(2048)
        self.batch_norm_dp = nn.BatchNorm1d(29959)

        # ECFP MLP
        self.ecfp_mlp_layer_1 = nn.Linear(2048, ((2048*(2/3)) + (256/3)), bias = True)
        self.ecfp_mlp_layer_2 = nn.Linear(((2048*(2/3)) + (256/3)), ((2048/3) + (256*(2/3))), bias = True)
        self.ecfp_mlp_layer_3 = nn.Linear(((2048/3) + (256*(2/3))), 256, bias = True)

        # Diffusion Profile MLP
        self.dp_mlp_layer_1 = nn.Linear(29959, 15000, bias = True)
        self.dp_mlp_layer_2 = nn.Linear(15000, 7500, bias = True)
        self.dp_mlp_layer_3 = nn.Linear(7500, 500, bias = True)

        #Prediction MLP
        self.pred_mlp_layer_1 = nn.Linear(756, (756*(2/3)), bias = True)
        self.pred_mlp_layer_2 = nn.Linear((756*(2/3)), (756/3), bias = True)
        self.pred_mlp_layer_3 = nn.Linear((756/3), 2, bias = True)

        # Dropout Layer
        self.dropout_layer_1 = nn.Dropout() # Better than using F.dropout() since model.eval() auto switches to testing mode with dropout module vs F.dropout()

        self.activ = nn.ReLU()
        self.activ2 = nn.Softmax()
    
    def forward(self, x):
    
        ecfp = x[0].float().to(device)
        diffusion_profile = x[1].float().to(device)

        ecfp = self.batch_norm_ecfp(ecfp)
        diffusion_profile = self.batch_norm_dp(diffusion_profile)

        ecfp = self.dropout_layer_1(ecfp)
        ecfp_embed_1 = self.dropout_layer_1(self.activ(self.ecfp_mlp_layer_1(ecfp)))
        ecfp_embed_2 = self.dropout_layer_1(self.activ(self.ecfp_mlp_layer_2(ecfp_embed_1)))
        ecfp_embed_3 = self.dropout_layer_1(self.activ(self.ecfp_mlp_layer_2(ecfp_embed_2)))

        self.ecfp_embed = ecfp_embed_3

        dp_embed_1 = self.dropout_layer_1(self.activ(self.dp_mlp_layer_1(diffusion_profile)))
        dp_embed_2 = self.dropout_layer_1(self.activ(self.dp_mlp_layer_2(dp_embed_1)))
        dp_embed_3 = self.dropout_layer_1(self.activ(self.dp_mlp_layer_3(dp_embed_2)))

        final_input_embed = torch.cat([ecfp_embed_3, dp_embed_3], dim = -1)

        pred_embed_1 = self.dropout_layer_1(self.activ(self.pred_mlp_layer_1(final_input_embed)))
        pred_embed_2 = self.dropout_layer_1(self.activ(self.pred_mlp_layer_2(pred_embed_1)))
        pred_embed_3 = self.dropout_layer_1(self.activ(self.pred_mlp_layer_3(pred_embed_2)))

        return pred_embed_3

    def cross_entropy_loss(self, probability, label):
        return nn.CrossEntropyLoss(probability, label)
    
    def training_step(self, train_set):
        probability = train_set
        label = train_set[2].to(device)

        loss = self.cross_entropy_loss(probability, label)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, valid_set):
        probability = valid_set
        label = valid_set[2].to(device)

        loss = self.cross_entropy_loss(probability, label)
        self.log('valid_loss', loss)

        


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 5e-4, weight_decay = 1e-5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, 0.95)
            }}

class ECFP_DP_MLP_Data_Module(pl.LightningDataModule):

    def setup(self):

        with open("placeholder_file_name.pkl", "rb") as f:
            combined_train_and_valid = pickle.load(f)
        
        self.train_set, self.valid_set = train_test_split(combined_train_and_valid, test_size = 0.25, random_state = 44)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = 32, shuffle = True)
    
    def valid_dataloader(self):
        return DataLoader(self.valid_set, batch_size = 32, shuffle = True)


data_module = ECFP_DP_MLP_Data_Module()

# train
model = ECFP_DP_MLP()
trainer = pl.Trainer()

trainer.fit(model, data_module)