import torch.nn as nn
from torchvision import models,transforms
import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(device)
        self.feature_extractor = create_feature_extractor(vgg, ["features.34"])
        self.feature_extractor.eval()
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        for param in vgg.parameters():
            param.requires_grad = False
        self.loss = nn.MSELoss()

    def forward(self, gen_im, original_im):
        gen_im = self.normalization(gen_im)
        original_im = self.normalization(original_im)
        gen_features = self.feature_extractor(gen_im)["features.34"]
        original_features = self.feature_extractor(original_im)["features.34"]
        return self.loss(original_features, gen_features)
