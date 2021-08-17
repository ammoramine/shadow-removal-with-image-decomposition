from torch import optim
import torch,os
from torch.utils.data import DataLoader
dirFile = os.path.dirname(__file__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
try:
    from .. import shadowRemovelDataset
except:
    import shadowRemovelDataset


args = {
    'iter_num': 3000,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 416
}


class Trainer:
    def __init__(self,optimizer,model,dt_loader):
        self.optimizer = optimizer
        self.model = model
        self.dt_loader = dt_loader

    def train(self):
        pass


def show_output(inpt):
    """input image wiht shadow processed"""
    import matplotlib.pyplot as plt
    import numpy as np
    model.eval()
    out = model(inpt).to("cpu").detach().numpy()

    fig,axs = plt.subplots(1,2)

    inpt = inpt.to("cpu").detach()
    axs[0].imshow(np.moveaxis(inpt.numpy().squeeze(), 0, 2))
    axs[1].imshow(out.squeeze(),cmap='gray')

    plt.show()

if __name__ == '__main__':
    # optimiezr = optim.Adam()


    rel_path_model = "../../Data/BDRAR.pth"
    path_model = os.path.join(dirFile,rel_path_model)

    rel_path_input_data = "../../Data/ISTD_Dataset/train"
    path_input_Data = os.path.join(dirFile,rel_path_input_data)

    # dtset = shadowRemovelDataset.ShadowRemovalDataSet(path_input_Data)
    try:
        from ..preprocess_module import custom_transforms
        from .. import shadowRemovelDataset
    except:
        from preprocess_module import custom_transforms
        import shadowRemovelDataset
    dtset = shadowRemovelDataset.ShadowRemovalDataSet(
            path_input_Data,
    joint_transform=custom_transforms.joint_transform,
    inpt_img_transform = custom_transforms.inpt_img_transform,
    out_img_transform = custom_transforms.out_img_transform
            )
    dt_loader = DataLoader(dtset,batch_size=8,collate_fn=dtset.collate_fn,shuffle=True)

    # cur_dir = os.getcwd()
    # os.chdir(os.path.join(dirFile,"../models"))
    # model = torch.load(path_model)

    from models.BDRARImported import modelBDRAR
    model = modelBDRAR.BDRAR().to("cuda")
    rel_path_model = "../../Data/3000.pth"
    path_model = os.path.join(dirFile, rel_path_model)
    assert os.path.exists(path_model)
    model.load_state_dict(torch.load(path_model, map_location=torch.device("cuda")))

    # os.chdir(cur_dir)

    model = model.to(device)
    model.train()

    # optimizer = optim.Adam(model.parameters(), lr=0.05)
    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],'lr': 2 * args['lr']},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],'lr': args['lr'], 'weight_decay': args['weight_decay']}], momentum=args['momentum'])

    alg_trainer = Trainer(model,optimizer,dt_loader)

    from torch import nn
    bce_logit = nn.BCEWithLogitsLoss().cuda()

    # mse_loss = nn.MSELoss()

    torch.autograd.set_detect_anomaly(True)
    nb_epochs = 30
    losses = []
    curr_iter = args['last_iter']
    for epoch in range(nb_epochs):
        loss_sum = 0
        for i,el in enumerate(dt_loader):
            print((epoch+1)*i/len(dt_loader))
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
            inpt = el[0].to(device)
            labels = el[1].to(device)
            model.train()
            ###########################################################################################
            optimizer.zero_grad()

            fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
            predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = model(inpt)


            loss_fuse = bce_logit(fuse_predict, labels)
            loss1_h2l = bce_logit(predict1_h2l, labels)
            loss2_h2l = bce_logit(predict2_h2l, labels)
            loss3_h2l = bce_logit(predict3_h2l, labels)
            loss4_h2l = bce_logit(predict4_h2l, labels)
            loss1_l2h = bce_logit(predict1_l2h, labels)
            loss2_l2h = bce_logit(predict2_l2h, labels)
            loss3_l2h = bce_logit(predict3_l2h, labels)
            loss4_l2h = bce_logit(predict4_l2h, labels)

            loss = loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + loss2_l2h + loss3_l2h + loss4_l2h
            loss_sum += loss
            loss.backward()

            ###########################################################################################
            optimizer.step()

        loss_mean = loss_sum/len(dt_loader)
        print(loss_mean)
        losses.append(loss_mean)
        # break
