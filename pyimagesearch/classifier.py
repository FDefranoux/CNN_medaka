from torch import nn, hub


class Larval_MLClassifier(nn.Module):
    def __init__(self, numClasses, freeze=True, mode='categorical',
                 model_dict={'repo_or_dir': 'pytorch/vision:v0.10.0',
                             'model': 'resnet18',
                             'pretrained': True,
                             'skip_validation': True}):
        super().__init__()
        self.numClasses = numClasses
        self.freeze = freeze
        self.model_dict = model_dict
        self.baseModel = hub.load(**model_dict)
        self.encoding_size = self.baseModel.fc.out_features
        self.mode = mode

        # Freeze the base model's weights (and omit last dense layer)
        if freeze:
            for i, child in enumerate(list(self.baseModel.children())[:-1]):
                for param in child.parameters():
                    param.requires_grad = False

        #QUESTION : Can we define the regression as a 1 class model ???
        if self.mode == 'categorical':
            self.fc = nn.Linear(self.baseModel.fc.out_features, numClasses)
            self.criterion = nn.CrossEntropyLoss
        elif self.mode == 'regression':
            self.fc = nn.Linear(self.baseModel.fc.out_features, 1)
            self.criterion = nn.MSELoss
        
    def forward(self, x):
        # pass the inputs through the base model to get the features
        # and then pass the features through of fully connected layer
        # to get our output logits
        encoding = self.baseModel(x)
        logits = self.fc(encoding)
        # coefficient = self.fc_regress(encoding)
        # return the classifier outputs
        return {'logits': logits, 'encoding': encoding}