# import os.path
# from os import path

# from facenet_pytorch import MTCNN #Face detector
# from facenet_pytorch import InceptionResnetV1 #Feature extractor

# import torch
# import torchvision
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu") # Defines the computation device (cuda:0 => GPU)

# cwd = os.path.dirname(os.path.realpath(__file__))
# _ = os.path.join(cwd, "datasets")
# if path.exists(_) == False:
#     print(cwd)
#     os.mkdir(_)
# else:
#     print("Datasets directory found in %s"%_)

# lfw_train = torchvision.datasets.LFWPairs(root = _, image_set='original', download = True, transform=ToTensor())

# train_dataldr = DataLoader(lfw_train, batch_size = 64)

# train_features, train_labels = next(iter(train_dataldr))

# print(f"Feature batch shape: {train_features.size()}")

# img = train_features[0][0].squeeze()
# plt.imshow(img)
# plt.show()

# detector = MTCNN()

# num_epochs = 10

# model = InceptionResnetV1(pretrained='vggface2') # Facenet model with Inception1 as a backbone architecture and pretrained on VGGFace2 dataset

# for epoch in range(num_epochs):
#     print("Epoch {}/{}".format(epoch+1, epoch+1))

#     for phase in ["train", "eval"]:
#         if phase == "train":
#             model.train().to(device) # Training phase of the model
#         elif phase == "eval":
#             model.eval().to(device) # Evaluating phas eof the model to determine the best one