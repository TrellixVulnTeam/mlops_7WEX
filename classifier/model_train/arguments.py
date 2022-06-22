import argparse
class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        # Dataset arguments
        self._parser.add_argument('--data_path', type=str, default='/home/jongjin/st/dataset/animals_resize/animals_df.csv', help='Dataset file path.')
        self._parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
        self._parser.add_argument('--image_size', type=int, default=224, help='Integer value for number of points.')
        self._parser.add_argument('--in_channels', type=int, default=3, help='Number of image channels')
        self._parser.add_argument('--num_classes', type=int, default=3, help='Number of image channels')

        # Optimizer arguments
        self._parser.add_argument('--momentum', type=int, default=0.9, help='GPU number to use.')
        self._parser.add_argument('--learning_rate', type=float, default=1e-3, help='Adam : learning rate.')

        # Main arguments
        self._parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from.')
        self._parser.add_argument('--epochs', type=int, default=9, help='Number of epochs of training.')

        # data arguments
        self._parser.add_argument('--hole_size', type=int, default=20, help='Generated results path.')
        self._parser.add_argument('--num_holes', type=int, default=3, help='Generated results path.')

        # Model arguments
        self._parser.add_argument('--backbone', type=str, default='resnet18', help='Generated results path.')
        self._parser.add_argument('--pretrained', type=bool, default=True, help='Generated results path.')
    def parser(self):
        return self._parser

