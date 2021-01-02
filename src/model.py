import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel,self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size = (3,3), padding = (1,1))
        self.max_pool_1 = nn.MaxPool2d( kernel_size = (2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size = (3,3), padding = (1,1))
        self.max_pool_2 = nn.MaxPool2d( kernel_size = (2, 2))
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)

        # adding a LSTM or GRU model
        self.gru = nn.GRU(64, 32, bidirectional = True, num_layers = 2, dropout = 0.25, batch_first = True)
        self.output = nn.Linear(64, num_chars  + 1)
        # we have 19 clases, but we want one extra class for UNKNOWN_CLASS

    def forward(self, images, targets = None):
        batch_size, channels, height, width = images.size()
        #print(batch_size, channels, height, width)
        x = F.relu(self.conv_1(images))
        #print(x.size())
        x = self.max_pool_1(x)
        #print(x.size())
        x = F.relu(self.conv_2(x))
        #print(x.size())
        x = self.max_pool_2(x) # 1 64 18 75
        #print(x.size())
        # Permute - return a view of the original tensor with its dimensions permuted
        x = x.permute(0, 3, 1, 2) # 1 75 64 18
        #print(x.size())
        # https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        # https://zhang-yang.medium.com/explain-pytorch-tensor-stride-and-tensor-storage-with-code-examples-50e637f1076d
        x = x.view(batch_size, x.size(1), -1)
        #print(x.size())
        x = self.linear_1(x)
        x = self.drop_1(x)
        #print(x.size())
        x, _ = self.gru(x)
        #print(x.size())
        x = self.output(x)
        # SO FOR EACH TIME STAMP (WE HAVE 75 -- middle DIFFERENT TIMESTAMPS) AND EACH TIMESTAMP IT IS RETURNING ME A VECTOR OF SIZE 20 -- final
        #print(x.size())
        x = x.permute(1, 0 , 2) # batch size will go to the middle
        #print(x.size())
        if targets is not None:
            log_softmax_values = F.log_softmax(x,2)
            input_lengths = torch.full(
                size = (batch_size, ) ,
                fill_value = log_softmax_values.size(0),
                dtype = torch.int32
                )
            #print("hello", input_lengths)
            target_lengths = torch.full(
                size = (batch_size, ) ,
                fill_value = targets.size(1),
                dtype = torch.int32
                )
            #print(target_lengths)
            loss =nn.CTCLoss(blank= 0)( log_softmax_values, targets, input_lengths, target_lengths)
            return x, loss
        #return x, None

# remember that the time steps will go in first then batches and then the values

if __name__ == "__main__":
    cm = CaptchaModel(19)
    # img = torch.rand(1,3,75,300)
    # target = torch.randint(1, 20, (1,5))
    # x , loss = cm(img, target)
    img = torch.rand(5,3,75,300)
    target = torch.randint(1, 20, (5,5))
    x , loss = cm(img, target)








