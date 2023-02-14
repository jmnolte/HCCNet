import torch
import monai
import os

class ResNet(torch.nn.Module):
    def __init__(self, num_channels, version: str) -> None:
        super().__init__()

        '''
        Define the model's version and set the number of input channels.

        Args:
            num_channels (int): Number of input channels.
            version (str): Model version.
        '''

        self.version = version

        if self.version == 'resnet10':
            self.model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=num_channels)
        
        elif self.version == 'resnet18':
            self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=num_channels)

        elif self.version == 'resnet34':
            self.model = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=num_channels)

        elif self.version == 'resnet50':
            self.model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=num_channels)

        elif version == 'resnet101':
            self.model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=num_channels)
        
        elif self.version == 'resnet152':
            self.model = monai.networks.nets.resnet152(spatial_dims=3, n_input_channels=num_channels)

        elif self.version == 'resnet200':
            self.model = monai.networks.nets.resnet200(spatial_dims=3, n_input_channels=num_channels)
    
    def define_output_layer(self, num_classes: int) -> None:

        '''
        Define the model's number of output classes.

        Args:
            num_classes (int): Number of output classes.
        '''

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

    def intialize_model(self, weights_path: str, pretrained: bool = True) -> None:

        '''
        Initialize the networks weights. If pretrained weights are used, 
        the weights are loaded from the MedicalNet repository. To access them,
        see: https://github.com/Tencent/MedicalNet

        Args:
            pretrained (bool): If True, pretrained weights are used.
            weights_path (str): Path to the pretrained weights.
        '''

        if pretrained:
            model_dict = self.model.state_dict()
            new_weights_path = os.path.join(weights_path, 'resnet_' + str(self.version.strip('resnet')) + '.pth')
            weights_dict = torch.load(new_weights_path, map_location=torch.device('cpu'))
            weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}
            model_dict.update(weights_dict)
            self.model.load_state_dict(model_dict)

    def extract_features(self, feature_extraction: bool = False) -> None:

        '''
        Freeze the model's weights. If feature_extraction is set to True, only the last layer
        is updated during training. If feature_extraction is set to False, all layers are updated.

        Args:
            feature_extraction (bool): If True, only the last layer is unfrozen. If False,
            all layers are unfrozen.
        '''

        if feature_extraction:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def assert_unfrozen_parameters(self) -> None:

        '''
        Assert which parameters will be updated during the training run.
        '''

        print("Parameters to be updated:")
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(name)

WEIGHTS_PATH = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'

resnet50 = ResNet(num_channels=1, version='resnet50')
resnet50.define_output_layer(num_classes=2)
resnet50.intialize_model(weights_path=WEIGHTS_PATH, pretrained=True)
resnet50.extract_features(feature_extraction=True)
resnet50.assert_unfrozen_parameters()
        