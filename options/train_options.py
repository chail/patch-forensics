from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def __init__(self, print_opt=True):
        BaseOptions.__init__(self, print_opt)
        parser = self.parser
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results visualization')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=100, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', default='constant', help='lr schedule [constant|plateau]')
        parser.add_argument('--patience', type=int, default=10, help='will stop training if val metric does not improve for this many epochs')
        parser.add_argument('--max_epochs', type=int, help='maximum epochs to train, if not specified, will stop based on patience, or whichever is sooner')

        # augmentations
        parser.add_argument('--random_crop', action='store_true', help='use random crop instead of center crop')
        parser.add_argument('--random_resized_crop', action='store_true', help='use random resized crop instead of center crop')
        parser.add_argument('--no_flip', action='store_true', help='do not flip the images for data augmentation')
        parser.add_argument('--color_augment', action='store_true', help='performs color augmentations, see utils/transforms.py')
        parser.add_argument('--all_augment', action='store_true', help='performs various augmentations, see utils/transforms.py')
        parser.add_argument('--cnn_detection_augment', type=float, help='use augmentations from https://arxiv.org/abs/1912.11035, recommended values are 0.1 or 0.5')

        self.isTrain = True
