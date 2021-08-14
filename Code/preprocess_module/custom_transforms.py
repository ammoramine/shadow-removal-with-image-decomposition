import torchvision
# from torchvision import transforms
try:
    from . import joint_tranforms
except:
    import joint_tranforms
# class preprocessor


args = {
    'snapshot': '3000',
    'scale': 416
}

joint_transform = joint_tranforms.Compose([
    # joint_tranforms.RandomHorizontallyFlip(),
    joint_tranforms.Resize((args['scale'], args['scale'])),
])

# val_joint_transform = elementary_tranforms.Compose([
#     elementary_tranforms.Resize((args['scale'], args['scale']))
# ])
#

inpt_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

out_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
# preprocessor_shdw_mask_net = transforms.Compose([
#     transforms.Resize(args['scale']),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# to_pil = transforms.ToPILImage()

# preprocessor_shdw_mask_net_outpt = transforms.Compose(
#     [
#         transforms.Resize((h, w))
#     ]
# )

# prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
# prediction = crf_refine(np.array(img.convert('RGB')), prediction)