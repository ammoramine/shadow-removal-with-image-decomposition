from torch import optim
import torch,os
from torch.utils.data import DataLoader
dirFile = os.path.dirname(__file__)
from torch import nn
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
try:
    from .. import shadowRemovelDataset
    from ..preprocess_module import custom_transforms
    from .. import shadowRemovelDataset
    from ..models.BDRARImported import modelBDRAR
    from .. import utils
    from ..metrics import shadow_detector_metric
except: #if the module is launched directly from the Code repertory
    import shadowRemovelDataset
    from preprocess_module import custom_transforms
    import shadowRemovelDataset
    from models.BDRARImported import modelBDRAR
    import utils
    from metrics import shadow_detector_metric



args = {
    'iter_num': 3000,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'scale': 416
}


class Trainer:
    def __init__(self,optimizer,model,dt_loader_train,dt_loader_val,device):
        self.optimizer = optimizer
        self.model = model
        self.dt_loader_train = dt_loader_train
        self.dt_loader_val = dt_loader_val
        self.device = device
        self.bce_logit = nn.BCEWithLogitsLoss().to(self.device)
        self.metric = shadow_detector_metric.BER_metric(device)

        # self.dt_loader_train_iterator = iter(self.dt_loader_train)
        # self.dt_loader_val_iterator = iter(cycle(self.dt_loader_val))
        torch.autograd.set_detect_anomaly(True)
        self.train_losses = []
        self.val_losses = []

    def train(self,nb_epochs=40):
        for epoch in range(nb_epochs):
            loss_train_mean = self.train_over_epoch()
            with torch.no_grad():
                torch.cuda.empty_cache()
            loss_val_mean = self.validate_over_epoch()
            print(f" training loss for epoch {epoch} is {loss_train_mean}")
            print(f" validation loss for epoch {epoch} is {loss_val_mean}")
            print(f" result of evaluation metric for epoch {epoch} is {self.metric.compute()}")
            self.train_losses.append(loss_train_mean)
            self.val_losses.append(loss_val_mean)
    def train_over_epoch(self):
        """iterate over all the shuffled batches of the training dataset for one epoch """
        self.dt_loader_train_iterator = iter(self.dt_loader_train)
        loss_train_sum = 0
        while True:
            try:
                #TODO ,set the optimizer outside the loop
                loss_train = self.iterate_for_training_batch()
                loss_train_sum += loss_train
            except StopIteration:
                # print("Iteration  over epoch")
                break
        loss_train_mean = loss_train_sum/len(self.dt_loader_train)
        return loss_train_mean

    @torch.no_grad()
    def validate_over_epoch(self):
        """iterate over all the shuffled batches of the training dataset for one epoch """
        self.dt_loader_val_iterator = iter(self.dt_loader_val)
        loss_val_sum = 0
        self.metric.reset()
        # the metric is computed during loss computation inside the function for loss computation
        while True:
            try:
                #TODO ,set the optimizer outside the loop
                loss_val = self.iterate_for_validation_batch()
                loss_val_sum += loss_val
            except StopIteration:
                # print("Iteration is over")
                break
        loss_val_mean = loss_val_sum/len(self.dt_loader_val)
        return loss_val_mean

    def reduce_learning_rate(self):
        """folloxing a decay, we just copy the proposed technique for the original model"""
        curr_iter = len(self.train_losses)
        optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
        optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
        return optimizer

    def get_training_loss(self):
        el = next(self.dt_loader_train_iterator)
        inpt = el[0].to(device)
        labels = el[1].to(device)
        self.model.train()
        fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = self.model(inpt)

        loss_fuse = self.bce_logit(fuse_predict, labels)
        loss1_h2l = self.bce_logit(predict1_h2l, labels)
        loss2_h2l = self.bce_logit(predict2_h2l, labels)
        loss3_h2l = self.bce_logit(predict3_h2l, labels)
        loss4_h2l = self.bce_logit(predict4_h2l, labels)
        loss1_l2h = self.bce_logit(predict1_l2h, labels)
        loss2_l2h = self.bce_logit(predict2_l2h, labels)
        loss3_l2h = self.bce_logit(predict3_l2h, labels)
        loss4_l2h = self.bce_logit(predict4_l2h, labels)

        loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + loss2_l2h + loss3_l2h + loss4_l2h
        return loss

    def get_validation_loss(self,with_metric_eval=True):
        el = next(self.dt_loader_val_iterator)
        inpt = el[0].to(device)
        labels = el[1].to(device)
        self.model.eval()
        fuse_predict = self.model(inpt)
        loss_fuse = self.bce_logit(fuse_predict, labels)
        if with_metric_eval:
            self.metric.update(fuse_predict,labels)
        return loss_fuse

    def iterate_for_training_batch(self):
        self.optimizer = self.reduce_learning_rate()

        self.optimizer.zero_grad()
        loss = self.get_training_loss()
        loss.backward()

        self.optimizer.step()
        return loss

    def iterate_for_validation_batch(self):
        loss_val = self.get_validation_loss()
        #add code for validation metric
        return loss_val
def show_output(inpt,model):
    """input image wiht shadow processed"""
    model.eval()
    out = model(inpt).to("cpu").detach().numpy()

    fig,axs = plt.subplots(1,2)

    inpt = inpt.to("cpu").detach()
    inpt_as_np_rgb = np.moveaxis(inpt.numpy().squeeze(), 0, 2)
    inpt_as_np_rgb_rescaled = (inpt_as_np_rgb - inpt_as_np_rgb.min()) / (inpt_as_np_rgb.max() - inpt_as_np_rgb.min())
    axs[0].imshow(inpt_as_np_rgb_rescaled)
    axs[1].imshow(out.squeeze(),cmap='gray')

    plt.show()


def get_results_for_els_witgh_gd_truth(el,model):
    """ get an element from the data_loader, and"""
    model.eval()
    model = model.to("cpu")
    out = model(el[0].to("cpu"))
    gd_truth = el[1]
    return el[0],out,gd_truth
#
#
# def compute_BER_metric(output,ground_truth):
#     P = output>0.5 # positive
#     N = ~P #negatives
#     S = ground_truth>0.5 # shadow pixels
#     NS = ~S # non shadow pixels
#
#     TP = torch.sum(P*S) # number true positive
#
#     TN = torch.sum(N*NS) # true negative
#
#     NumS = torch.sum(S)
#     NumNS = torch.sum(NS)
#
#     NumS = max(NumS,1)
#     NumNS = max(NumNS,1)
#
#     BER = (1 - 0.5 * (TP/NumS + TN/NumNS)) * 100
#     return BER

def show_sample_at_idx(res,idx):
    assert idx < len(res)

    inpt,out,gd_th = res
    inpt, out, gd_th = inpt[idx].cpu(),out[idx].cpu(),gd_th [idx].cpu()

    fig,axs = plt.subplots(1,3)


    axs[0].imshow(utils.rescale_to_float(np.moveaxis(inpt.detach().numpy().squeeze(), 0, 2)))
    axs[1].imshow(out.detach().squeeze().numpy(),cmap='gray')
    axs[2].imshow(gd_th.detach().squeeze().numpy(),cmap='gray')

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--device", help = "choose the device on which to run the code", choices = ["cuda", "cpu"], default = "cuda")
    parser.add_argument("-t","--truncate", help = "choose to limit the number of element on the dataset, for debuggins purpose", type= int, default = None)
    parser.add_argument("-nb","--nb_epochs", help = "the number of epochs for the training process",type=int,default=40)

    args_parser = parser.parse_args()
    nb_epochs = args_parser.nb_epochs
    device = args_parser.device
    truncate = args_parser.truncate

    rel_path_model = "../../Data/BDRAR.pth"
    path_model = os.path.join(dirFile,rel_path_model)

    rel_path_input_data = "../../Data/ISTD_Dataset/train"
    path_input_Data_train = os.path.join(dirFile,rel_path_input_data)

    rel_path_input_data = "../../Data/ISTD_Dataset/test"
    path_input_Data_val = os.path.join(dirFile,rel_path_input_data)

    dtset_train = shadowRemovelDataset.ShadowRemovalDataSet(path_input_Data_train,joint_transform=custom_transforms.joint_transform,inpt_img_transform = custom_transforms.inpt_img_transform,out_img_transform = custom_transforms.out_img_transform,truncate=truncate)
    dtset_val = shadowRemovelDataset.ShadowRemovalDataSet(path_input_Data_val,joint_transform=custom_transforms.joint_transform,inpt_img_transform = custom_transforms.inpt_img_transform,out_img_transform = custom_transforms.out_img_transform,truncate=truncate)
    # dtset_val = dtset_train
    dt_loader_train = DataLoader(dtset_train,batch_size=args["train_batch_size"],collate_fn=dtset_train.collate_fn,shuffle=True)
    dt_loader_val = DataLoader(dtset_val,batch_size=args["train_batch_size"],collate_fn=dtset_val.collate_fn,shuffle=True)

    # dt_loader_val = dt_loader_train
    # must load with cuda anyway, the emodel can't be loaded using cpu, which is a problem

    model = modelBDRAR.BDRAR().to("cuda")
    rel_path_model = "../../Data/3000_old.pth"
    path_model = os.path.join(dirFile, rel_path_model)
    assert os.path.exists(path_model)
    model.load_state_dict(torch.load(path_model, map_location=torch.device("cuda")))

    model = model.to(device)
    model.train()


    # optimizer = optim.Adam(model.parameters(), lr=0.05)
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],'lr': 2 * args['lr']},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],'lr': args['lr'], 'weight_decay': args['weight_decay']}], momentum=args['momentum'])

    alg_trainer = Trainer(optimizer,model,dt_loader_train,dt_loader_val,device)

    alg_trainer.train(nb_epochs)

    # from torch import nn
    # bce_logit = nn.BCEWithLogitsLoss().cuda()

    # mse_loss = nn.MSELoss()

    # torch.autograd.set_detect_anomaly(True)
    # nb_epochs = 30
    # losses = []
    # curr_iter = args['last_iter']
    # for epoch in range(nb_epochs):
    #     loss_sum = 0
    #     for i,el in enumerate(dt_loader_train):
    #         print((epoch+1)*i/len(dt_loader_train))
    #         optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
    #         optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
    #         inpt = el[0].to(device)
    #         labels = el[1].to(device)
    #         model.train()
    #         ###########################################################################################
    #         optimizer.zero_grad()
    #
    #         fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
    #         predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = model(inpt)
    #
    #
    #         loss_fuse = bce_logit(fuse_predict, labels)
    #         loss1_h2l = bce_logit(predict1_h2l, labels)
    #         loss2_h2l = bce_logit(predict2_h2l, labels)
    #         loss3_h2l = bce_logit(predict3_h2l, labels)
    #         loss4_h2l = bce_logit(predict4_h2l, labels)
    #         loss1_l2h = bce_logit(predict1_l2h, labels)
    #         loss2_l2h = bce_logit(predict2_l2h, labels)
    #         loss3_l2h = bce_logit(predict3_l2h, labels)
    #         loss4_l2h = bce_logit(predict4_l2h, labels)
    #
    #         loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + loss2_l2h + loss3_l2h + loss4_l2h
    #         loss_sum += loss
    #         loss.backward()
    #
    #         ###########################################################################################
    #         optimizer.step()
    #
    #     loss_mean = loss_sum/len(dt_loader_train)
    #     print(loss_mean)
    #     losses.append(loss_mean)