import torch
import monai
import os

class ResNet(torch.nn.Module):
    def __init__(self, version: str, num_out_classes: int, num_in_channels: int, pretrained: bool, feature_extraction: bool, weights_path: str) -> None:
        super().__init__()

        '''
        Define the model's version and set the number of input channels.

        Args:
            version (str): ResNet model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', or 'resnet200'.
            num_out_classes (int): Number of output classes.
            num_in_channels (int): Number of input channels.
            pretrained (bool): If True, pretrained weights are used.
            feature_extraction (bool): If True, only the last layer is updated during training. If False,
                all layers are updated.
            weights_path (str): Path to the pretrained weights.
        '''
        try: 
            assert any(version == version_item for version_item in ['resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200'])
        except AssertionError:
            print('Invalid version. Please choose from: resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200')
            exit(1)

        self.version = version
        self.num_in_channels = num_in_channels
        self.pretrained = pretrained
        if self.version == 'resnet10':
            self.model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet18':
            self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet34':
            self.model = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet50':
            self.model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet101':
            self.model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet152':
            self.model = monai.networks.nets.resnet152(spatial_dims=3, n_input_channels=num_in_channels)
        elif self.version == 'resnet200':
            self.model = monai.networks.nets.resnet200(spatial_dims=3, n_input_channels=num_in_channels)

        self.model.fc = self.define_output_layer(num_out_classes)
        if self.pretrained:
            model_dict = self.intialize_model(weights_path)
            self.model.load_state_dict(model_dict)
            print('Pretrained weights are loaded.')
        self.extract_features(feature_extraction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        Forward pass through the model.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        x = self.model(x)
        return x
    
    def define_output_layer(self, num_classes: int) -> torch.nn.Linear:

        '''
        Define the model's number of output classes.

        Args:
            num_classes (int): Number of output classes.
    
        Returns:
            torch.nn.Linear: Output layer.
        '''

        num_ftrs = self.model.fc.in_features
        return torch.nn.Linear(num_ftrs, num_classes)

    def intialize_model(self, weights_path: str) -> dict:

        '''
        Initialize the networks weights. If pretrained weights are used, 
        the weights are loaded from the MedicalNet repository. To access them,
        see: https://github.com/Tencent/MedicalNet

        Args:
            weights_path (str): Path to the pretrained weights.
        
        Returns:
            dict: Dictionary containing the model's weights.
        '''

        model_dict = self.model.state_dict()
        new_weights_path = os.path.join(weights_path, 'resnet_' + str(self.version.strip('resnet')) + '.pth')
        weights_dict = torch.load(new_weights_path, map_location=torch.device('cuda'))
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}
        model_dict.update(weights_dict)
        conv1_weight = model_dict['conv1.weight']
        channel = 1
        while channel < self.num_in_channels:
            model_dict['conv1.weight'] = torch.cat((model_dict['conv1.weight'], conv1_weight), 1)
            channel += 1
        return model_dict

    def extract_features(self, feature_extraction: bool) -> None:

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

if __name__ == '__main__':
    WEIGHTS_PATH = '/Users/noltinho/MedicalNet/pytorch_files/pretrain'
    resnet50 = ResNet(version='resnet50', num_out_classes=2, num_in_channels=9, pretrained=True, feature_extraction=True, weights_path=WEIGHTS_PATH)