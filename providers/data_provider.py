from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose,ToTensor,Normalize

class DataProvider:
    def getTrainingData(self, batch_size = 128):
        training_data = datasets.MNIST(
            root = "data",
            train = True,
            download = True,
            transform = Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,),(0.3081,))
                ]
            )
        )
        train_dataloader = DataLoader(training_data,batch_size=batch_size,shuffle =True )
        return train_dataloader
    def getTestData(self):
        test_data = datasets.MNIST(
            root = 'data',
            train = False,
            download = True,
            transform = Compose(
                [
                    ToTensor(),
                    Normalize((0.1307,),(0.3081,))
                ]
            )
        )
        test_dataloader = DataLoader(test_data,batch_size=1000,shuffle =True )
        return test_dataloader