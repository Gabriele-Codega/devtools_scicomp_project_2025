from .model import AlexNet
from lightning import LightningModule
import torchmetrics
import torch
import torch.nn as nn

class Classifier(LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = AlexNet(num_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def _classifier_step(self, batch):
        features, true_labels = batch
        logit = self(features)

        loss = self.loss_fn(logit, true_labels)

        predicted_labels = logit.argmax(dim=-1)

        return predicted_labels, true_labels, loss

    def training_step(self, batch):
        yhat, y, loss = self._classifier_step(batch)

        self.train_accuracy(yhat,y)
        self.log('train_loss', loss.item())
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch):
        yhat, y, loss = self._classifier_step(batch)
        self.val_accuracy(yhat,y)
        self.log('validation_loss', loss)
        self.log('validation_accuracy', self.val_accuracy, on_step=True, on_epoch=False)

    def test_step(self, batch):
        yhat, y, loss = self._classifier_step(batch)
        self.test_accuracy(yhat,y)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy, on_step=True, on_epoch=False)

    def forward(self, x):
        return self.model(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
