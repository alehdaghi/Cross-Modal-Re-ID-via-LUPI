import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None, gray=False):
        
        data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        self.train_color_image = np.load(data_dir + 'train+Val_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train+Val_rgb_resized_label.npy')
        self.train_color_cam = np.load(data_dir + 'train+Val_rgb_resized_camera.npy')

        self.train_ir_image = np.load(data_dir + 'train+Val_ir_resized_img.npy')
        self.train_ir_label = np.load(data_dir + 'train+Val_ir_resized_label.npy')
        self.train_ir_cam = np.load(data_dir + 'train+Val_ir_resized_camera.npy')
        
        # BGR to RGB

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.returnsGray = gray

    def __getitem__(self, index):

        img1,  target1, cam1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]], self.train_color_cam[self.cIndex[index]]
        img2,  target2, cam2 = self.train_ir_image[self.tIndex[index]], self.train_ir_label[self.tIndex[index]], self.train_ir_cam[self.tIndex[index]]
        img3, target3 = -1, -1
        if self.returnsGray:
            img3 = self.rgb2RandomChannel(img1)
            img3 = self.transform(np.stack((img3,)*3, axis=-1))
            target3 = target1

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        #if self.returnsGray:
        return img1, img2, img3, target1, target2, target3, cam1, cam2
        #else:
        #    return img1, img2, img3, target1, target2, None#cam1-1, cam2-1

    def __len__(self):
        return len(self.train_color_label)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(rgb.dtype)

    @staticmethod
    def rgb2RandomChannel(rgb):
        n = np.random.rand(3)
        n /= n.sum()
        return np.dot(rgb[..., :3], n).astype(rgb.dtype)
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, gray=False):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_ir_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_ir_label = train_ir_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.returnsGray = gray

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_ir_label[self.tIndex[index]]
        img3, target3 = -1, -1
        if self.returnsGray:
            img3 = SYSUData.rgb2RandomChannel(img1)
            img3 = self.transform(np.stack((img3,) * 3, axis=-1))
            target3 = target1

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        img3 = self.transform(img3)

        return img1, img2, img3, target1, target2, target3, 1, 2

    def __len__(self):
        return len(self.train_color_label)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])



class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, test_cam, transform=None, img_size = (144,288), colorToGray=False):

        test_image = []
        ret_test_label,  ret_test_cam = [],[]
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            if colorToGray:
                for j in range(9):
                    test_image.append(np.stack((SYSUData.rgb2RandomChannel(pix_array),)*3, axis=-1))
                    ret_test_label.append(test_label[i])
                    ret_test_cam.append(test_cam[i])

            ret_test_cam.append(test_cam[i])
            ret_test_label.append(test_label[i])
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = np.array(ret_test_label)
        self.test_cam = np.array(ret_test_cam)
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1, cam1 = self.test_image[index],  self.test_label[index], self.test_cam[index]
        img1 = self.transform(img1)
        return img1, target1, cam1 - 1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
