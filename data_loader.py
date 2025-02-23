import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
def ACES_profession(x):
	# 定义输入和输出的转换矩阵
	ACESInputMat = np.array([
		[0.59719, 0.35458, 0.04823],
		[0.07600, 0.90834, 0.01566],
		[0.02840, 0.13383, 0.83777]
	])

	ACESOutputMat = np.array([
		[1.60475, -0.53108, -0.07367],
		[-0.10208, 1.10813, -0.00605],
		[-0.00327, -0.07276, 1.07602]
	])

	def RRTAndODTFit(v):
		"""
		模拟 HLSL 的 RRTAndODTFit 函数
		"""
		a = v * (v + 0.0245786) - 0.000090537
		b = v * (0.983729 * v + 0.4329510) + 0.238081
		return a / b

	# 将图像展平为二维矩阵 (N, 3)，其中 N 是像素数
	original_shape = x.shape
	color = x.reshape(-1, 3).T

	# 转换为线性空间
	color = np.dot(ACESInputMat, color)

	# 应用 RRT 和 ODT 映射
	color = RRTAndODTFit(color)

	# 转换为 sRGB 空间
	color = np.dot(ACESOutputMat, color)

	# 恢复为原始图像形状
	color = color.T.reshape(original_shape)

	return color

def ACES_profession_reverse(x):
	# 定义输入和输出的转换矩阵
	ACESInputMat = np.array([
		[0.59719, 0.35458, 0.04823],
		[0.07600, 0.90834, 0.01566],
		[0.02840, 0.13383, 0.83777]
	])

	ACESOutputMat = np.array([
		[1.60475, -0.53108, -0.07367],
		[-0.10208, 1.10813, -0.00605],
		[-0.00327, -0.07276, 1.07602]
	])

	ACESInputMat_inv = np.linalg.inv(ACESInputMat)
	ACESOutputMat_inv = np.linalg.inv(ACESOutputMat)

	def RRTAndODTFitInverse(y):
		"""
		计算 RRTAndODTFit 的逆函数
		"""
		A = 0.983729 * y - 1
		B = 0.4329510 * y - 0.0245786
		C = 0.238081 * y + 0.000090537

		discriminant = B**2 - 4 * A * C
		sqrt_discriminant = np.sqrt(discriminant)

		# 选择符合 v > 0 的解
		v2 = (-B - sqrt_discriminant) / (2 * A)
		return v2

	# 将图像展平为二维矩阵 (N, 3)，其中 N 是像素数
	original_shape = x.shape
	color = x.reshape(-1, 3).T

	# 转换为线性空间
	color = np.dot(ACESOutputMat_inv, color)

	# 应用 RRT 和 ODT 映射的逆函数
	color = RRTAndODTFitInverse(color)

	# 转换为 sRGB 空间
	color = np.dot(ACESInputMat_inv, color)

	# 恢复为原始图像形状
	color = color.T.reshape(original_shape)

	return color

class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)

#is not true remove
def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image

class Flare_Image_Loader(data.Dataset):
	def __init__(self, image_path ,transform_base=None,transform_flare=None,mask_type=None):
		self.ext = ['png','jpeg','jpg','bmp','tif']
		self.data_list=[]
		[self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
		self.flare_dict={}
		self.flare_list=[]
		self.flare_name_list=[]

		self.reflective_flag=False
		self.reflective_dict={}
		self.reflective_list=[]
		self.reflective_name_list=[]

		self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"

		self.transform_base=transform_base
		self.transform_flare=transform_flare

		print("Base Image Loaded with examples:", len(self.data_list))

	def __getitem__(self, index):
		# load base image

		img_path=self.data_list[index]
		base_img= Image.open(img_path)
		
		gamma=np.random.uniform(1.8,2.2)
		to_tensor=transforms.ToTensor()
		adjust_gamma=RandomGammaCorrection(gamma)
		adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
		if self.transform_base is not None:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=self.transform_base(base_img)
		else:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
		sigma_chi=0.01*np.random.chisquare(df=1)
		base_img=Normal(base_img,sigma_chi).sample()
		gain=np.random.uniform(0.5,1.2)
		flare_DC_offset=np.random.uniform(-0.02,0.02)
		base_img=gain*base_img
		base_img=torch.clamp(base_img,min=0,max=1)

		#load flare image
		flare_path=random.choice(self.flare_list)
		flare_img =Image.open(flare_path)
		if self.reflective_flag:
			reflective_path=random.choice(self.reflective_list)
			reflective_img =Image.open(reflective_path)


		flare_img=to_tensor(flare_img)
		flare_img=adjust_gamma(flare_img)
		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img)
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

		flare_img=remove_background(flare_img)

		if self.transform_flare is not None:
			flare_img=self.transform_flare(flare_img)
		
		#change color
		flare_img=color_jitter(flare_img)

		#flare blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset
		flare_img=torch.clamp(flare_img,min=0,max=1)

		#merge image	
		merge_img=flare_img+base_img
		print(merge_img.shape)
		# merge_img=torch.clamp(merge_img,min=0,max=1)
		AC_gain=np.random.uniform(0.5,1.0)
		blur_transform=transforms.GaussianBlur(3,sigma=(1,2))
		merge_img = ACES_profession(ACES_profession_reverse(flare_img)+ AC_gain*ACES_profession_reverse(blur_transform(base_img)))
		print(merge_img.shape)
		merge_img = torch.from_numpy(merge_img).float()
		print(merge_img.shape)
		merge_img=torch.clamp(merge_img,min=0,max=1)

		if self.mask_type==None:
			return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)

		return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),flare_mask,gamma

	def __len__(self):
		return len(self.data_list)
	
	def load_scattering_flare(self,flare_name,flare_path):
		flare_list=[]
		[flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
		self.flare_name_list.append(flare_name)
		self.flare_dict[flare_name]=flare_list
		self.flare_list.extend(flare_list)
		len_flare_list=len(self.flare_dict[flare_name])
		if len_flare_list == 0:
			print("ERROR: scattering flare images are not loaded properly")
		else:
			print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
		print("Now we have",len(self.flare_list),'scattering flare images')

	def load_reflective_flare(self,reflective_name,reflective_path):
		self.reflective_flag=True
		reflective_list=[]
		[reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
		self.reflective_name_list.append(reflective_name)
		self.reflective_dict[reflective_name]=reflective_list
		self.reflective_list.extend(reflective_list)
		len_reflective_list=len(self.reflective_dict[reflective_name])
		if len_reflective_list == 0:
			print("ERROR: reflective flare images are not loaded properly")
		else:
			print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
		print("Now we have",len(self.reflective_list),'refelctive flare images')
