from torch import nn, hub


class Larval_MLPhenotyper(nn.Module):
    def __init__(self, numClasses, freeze=True,
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

        # Freeze the base model's weights (and omit last dense layer)
        if freeze:
            for i, child in enumerate(list(self.baseModel.children())[:-1]):
                for param in child.parameters():
                    param.requires_grad = False

        #QUESTION : Can we define the regression as a 1 class model ???
        self.fc_classif = nn.Linear(self.baseModel.fc.out_features, numClasses)
        self.fc_regress = nn.Linear(self.baseModel.fc.out_features, 1)
        
    def forward(self, x):
        # pass the inputs through the base model to get the features
        # and then pass the features through of fully connected layer
        # to get our output logits
        encoding = self.baseModel(x)
        logits = self.fc_classif(encoding)
        coefficient = self.fc_regress(encoding)
        # return the classifier outputs
        return {'logits': logits, 'coefficient': coefficient, 'encoding': encoding}