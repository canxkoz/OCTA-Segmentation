import os
import errno
from PIL import Image, ImageOps
import torch.utils.data as data
from torchvision import transforms


def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def build_dataset(data_dir, channel=1, isTraining=True, scale_size=(512, 512)):
    database = CRIA(
        data_dir, channel=channel, isTraining=isTraining, scale_size=scale_size
    )
    return database


class CRIA(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(CRIA, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""
        assert (
            self.channel == 1 or self.channel == 3
        ), "the channel must be 1 or 3"  # check the channel is 1 or 3

    def __getitem__(self, index):
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        # gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()
        img = Image.open(imgPath)
        gtPath = imgPath.replace("\\original", "\\gt")

        if not os.path.exists(gtPath):
            print(self.name, imgPath, gtPath)
        gt = Image.open(gtPath)  # .convert("L")
        img = ImageOps.grayscale(img)
        img = ImageOps.equalize(img)
        gt = ImageOps.grayscale(gt)
        # gt = ImageOps.equalize(gt)
        img = simple_transform(img)
        gt = simple_transform(gt)

        return img, gt, self.name

    def __len__(self):
        return len(self.img_lst)

    def get_dataPath(self, root, isTraining):
        if isTraining:
            img_dir = os.path.join(root + "\\train\\original")
            gt_dir = os.path.join(root + "\\train\\gt")
        else:
            img_dir = os.path.join(root + "\\test\\original")
            gt_dir = os.path.join(root + "\\test\\gt")
        img_lst = sorted(
            list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        )
        # gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        # assert len(img_lst) == len(gt_lst)
        # return img_lst, gt_lst
        return img_lst, gt_dir

    def getFileName(self):
        return self.name


def listAllOCTFiles(imageDirPath, imagePostfix):
    fileList = os.listdir(imageDirPath)
    postLen = len(imagePostfix)
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-postLen::] == imagePostfix:
            imageNames.append(fileList[i][:-postLen])
    return imageNames
