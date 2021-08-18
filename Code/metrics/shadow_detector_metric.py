from torchmetrics import Metric

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BER_metric(Metric):
    def __init__(self,device):
        super().__init__()
        self.device = device

        self.add_state("TP",torch.scalar_tensor(0).to(self.device),"sum")
        self.add_state("TN",torch.scalar_tensor(0).to(self.device),"sum")
        self.add_state("numS",torch.scalar_tensor(1).to(self.device),"sum")
        self.add_state("numNS",torch.scalar_tensor(1).to(self.device),"sum")

    def update(self,output : torch.Tensor,ground_truth : torch.Tensor) -> None:
        """get the output of the model as first input, and the ground truth as second inputs"""
        P = output > 0.5  # positive
        N = ~P  # negatives
        S = ground_truth > 0.5  # shadow pixels
        NS = ~S  # non shadow pixels

        self.TP += torch.sum(P * S)  # number true positive

        self.TN += torch.sum(N * NS)  # true negative

        self.numS += max(torch.sum(S),1)
        self.numNS += max(torch.sum(NS),1)



    def compute(self):
        BER = (1 - 0.5 * (self.TP / self.numS + self.TN / self.numNS)) * 100
        return BER