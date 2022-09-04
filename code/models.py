import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, dim_feature=8):

        super(Discriminator, self).__init__()

        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.dim_feautre = dim_feature
        threshold_ReLU   = 0.2

        self.network = nn.Sequential(
                                     nn.Conv2d(in_channel, dim_feature * 1, kernel_size=3, stride=2, padding=1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     
                                     nn.Conv2d(dim_feature * 1, dim_feature * 2, kernel_size=3, stride=2, padding=1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     
                                     nn.Conv2d(dim_feature * 2, dim_feature * 4, kernel_size=3, stride=2, padding=1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     
                                     nn.Conv2d(dim_feature * 4, dim_feature * 8, kernel_size=3, stride=2, padding=1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     
                                     nn.Conv2d(dim_feature * 8, dim_feature * 16, kernel_size=3, stride=2, padding=1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     nn.Flatten(),
                                     nn.Linear(dim_feature * 16, dim_feature * 8, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     nn.Linear(dim_feature * 8, dim_feature * 4, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     nn.Linear(dim_feature * 4, dim_feature * 2, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     nn.Linear(dim_feature * 2, dim_feature * 1, bias=True),
                                     nn.LeakyReLU(threshold_ReLU, inplace=True),
                                     nn.Linear(dim_feature * 1, out_channel, bias=True),
                                     nn.Sigmoid()
                                    )
            
        self.initialize_weight()

    def forward(self, x):

        y = self.network.forward(x)

        return y

    def initialize_weight(self):

        print('initialize model parameters :', 'xavier_uniform')

        for m in self.network.modules():
            
            if isinstance(m, nn.Conv2d):
                
                nn.init.xavier_uniform_(m.weight)
                
                if m.bias is not None:

                    nn.init.constant_(m.bias, 1)
                    pass
                    
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
                
            elif isinstance(m, nn.Linear):
                
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    
                    nn.init.constant_(m.bias, 1)
                    pass

class Generator(nn.Module):

    def __init__(self, in_channel=1, out_channel=1, dim_feature=8):

        super(Generator, self).__init__()

        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.dim_feature = dim_feature
        threshold_ReLU   = 0.2

        self.network = nn.Sequential(
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channel, dim_feature * 8, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(dim_feature * 8),
                                    nn.LeakyReLU(threshold_ReLU, inplace=True),

                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(dim_feature * 8, dim_feature * 4, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(dim_feature * 4),
                                    nn.LeakyReLU(threshold_ReLU, inplace=True),

                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(dim_feature * 4, dim_feature * 2, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(dim_feature * 2),
                                    nn.LeakyReLU(threshold_ReLU, inplace=True),

                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(dim_feature * 2, dim_feature * 1, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(dim_feature * 1),
                                    nn.LeakyReLU(threshold_ReLU, inplace=True),
                                    
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    nn.Conv2d(dim_feature * 1, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(out_channel),
                                    nn.Sigmoid() # use this because MNIST values range within 0~1
                                    )

        self.initialize_weight()
    
    def forward(self, x):

        y = self.network.forward(x)

        return y

    def initialize_weight(self):

        print('initialize model parameters :', 'xavier_uniform')

        for m in self.network.modules():
            
            if isinstance(m, nn.Conv2d):
                
                nn.init.xavier_uniform_(m.weight)
                
                if m.bias is not None:

                    nn.init.constant_(m.bias, 1)
                    pass
                    
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
                
            elif isinstance(m, nn.Linear):
                
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    
                    nn.init.constant_(m.bias, 1)
                    pass
