import numpy as np
import os
import random
import math
import pickle as pkl
import sys
import cv2
sys.path.append("..")
from sys import platform as sys_pf
import matplotlib
if sys_pf == 'darwin':
	matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from conceptnet.augmentdata import *

#使用 Modified Hausdorff 距离测试 one-shot 在原论文 的效果如何，你可以查看 one-shot-classification。
#对点集A中的每个点ai到距离此点ai最近的B集合中点bj之间的距离‖ai-bj‖进行排序,然后取该距离中的最大值作为h(A,B)的值.
#H(A,B)=max(h(A,B),h(B,A)),双向Hausdorff距离H(A,B)是单向距离h(A,B)和h(B,A)两者中的较大者,
# 它度量了两个点集间的最大不匹配程度.

# ---
# Demo for how to load image and stroke data for a character
# ---
# Plot the motor trajectory over an image
#
# Input
#  I [105 x 105 nump] grayscale image
#  drawings: [ns list] of strokes (numpy arrays) in motor space
#  lw : line width
def plot_motor_to_image(I,drawing,lw=1):
	drawing = [d[:,0:2] for d in drawing] # strip off the timing data (third column)-坐标
	drawing = [space_motor_to_img(d) for d in drawing] # convert to image space
	plt.imshow(I,cmap='gray')
	ns = len(drawing)
	for sid in range(ns): # for each stroke
		plot_traj(drawing[sid],get_color(sid),lw)
	plt.xticks([])
	plt.yticks([])

# Plot individual stroke
#
# Input
#  stk: [n x 2] individual stroke-坐标
#  color: stroke color
#  lw: line width
def plot_traj(stk,color,lw):
	n = stk.shape[0]
	if n > 1:
		plt.plot(stk[:,0],stk[:,1],color=color,linewidth=lw)
	else:
		plt.plot(stk[0,0],stk[0,1],color=color,linewidth=lw,marker='.')

# Color map for the stroke of index k
def get_color(k):
	scol = ['r','g','b','m','c']
	ncol = len(scol)
	if k < ncol:
		out = scol[k]
	else:
		out = scol[-1]
	return out

# convert to str and add leading zero to single digit numbers
def num2str(idx):
	if idx < 10:
		return '0'+str(idx)
	return str(idx)

# Load binary image for a character
#
# fn : filename
def load_img(fn):
	I = plt.imread(fn)
	I = np.array(I,dtype=np.uint8)
	return I

# Load stroke data for a character from text file
# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)
def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline =='START': # beginning of character
			stk = []
		elif myline =='BREAK': # break between strokes
			stk = np.array(stk)
			motor.append(stk) # add to list of strokes
			stk = [] 
		else:
			arr = np.fromstring(myline,dtype=float,sep=',')
			stk.append(arr)
	return motor

#
# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates
def space_motor_to_img(pt):
	pt[:,1] = -pt[:,1]
	return pt

def space_img_to_motor(pt):
	pt[:,1] = -pt[:,1]
	return pt

#crd=os.path.abspath(os.path.dirname(os.getcwd()))
#增广方法
#在图像的水平和垂直轴上随机执行翻转
#training:1-17;testing:18-20
#针对每个character的实例，共计20个
def rendition_re(rj):
	print('rendition=', rj)
	r=rj#20个字母实例的第”r“个
	#nreps = 1  # number of renditions for each character
	img_dir = './OMNIGLOT/images_background_small1'
	stroke_dir = './OMNIGLOT/strokes_background_small1'
	#training:一个批次一个rendition:1-16,testing:17-20,从validiation中选出一个alpha集中一个进行验证。
	trainimage=[]#图像训练集
	trainstroke=[]#笔画训练集
	trainstrokemg=[]#所有笔画组成一张图像，笔画图像训练集
	#trainstrokenum=[]#笔画数目
	class_num=0#全部训练集的类别数
	part_num=0#全部训练集的笔画数

	nalpha = 3  # number of alphabets to show
	alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.'] # get folder names
	#alphabet_names = random.sample(alphabet_names,nalpha) # choose random alphabets
	for a in range(nalpha,nalpha+1): # for each alphabet：每个字符集
		print('generating figure ' + str(a+1) + ' of ' + str(nalpha))
		alpha_name = alphabet_names[a]
		# choose a random character from the alphabet
		#character_id = random.randint(1,len(os.listdir(os.path.join(img_dir,alpha_name))))

		#cl=len(os.listdir(os.path.join(img_dir, alpha_name)))#character=class数目
		cl=11
		# get image and stroke directories for this character
		# get base file name for this character
		for cid in range(cl,cl+1):#for each character：针对每个字母
			character_id = cid
			img_char_dir = os.path.join(img_dir,alpha_name,'character'+num2str(character_id))
			stroke_char_dir = os.path.join(stroke_dir,alpha_name,'character'+num2str(character_id))
			fn_example = os.listdir(img_char_dir)[r]#character中20个实例的第r个example .._01
			fn_base = fn_example[:fn_example.find('_')]
			#print('fn_base:',fn_base)

			#plt.figure(a,figsize=(10,8))#以下为一个character中的各个image和stroke # for each rendition
			#plt.clf()#不用重新产生窗口
			fn_stk = stroke_char_dir + '/' + fn_base + '_' + num2str(r+1) + '.txt'
			fn_img = img_char_dir + '/' + fn_base + '_' + num2str(r+1) + '.png'
			motor = load_motor(fn_stk)#加载笔画
			if len(motor)>3:
				print('alphabet=',a,'character=',cid,'strokes=',len(motor))

			#开始图像
			I = load_img(fn_img)#np.uint8
			#psi = Perspective()
			#I = psi(I)  # 对图像I进行随机变换

			"""plt.subplot(1,2,1)
			plt.imshow(I)
			plt.title('image:' + str(np.max(I)))
			#plot_motor_to_image(I,motor)
			#plt.title(alpha_name[:15] + '\n character ' + str(character_id)+':'+fn_base + '_' + num2str(r))
			"""
			II = I#.astype(int)#image: each character example
			class_num=class_num+1#全部训练集类别数累加
			trainimage.append([II,class_num])#图像及其对应类别

			#开始每一个字母character的笔画
			dilateker=np.ones((5,5),dtype=np.uint8)
			MTIs = np.zeros((II.shape))#stroke: each character example
			beforestrokenum=part_num
			plt.figure(figsize=(15,3),facecolor='white')
			for mi in range(len(motor)):#each stroke：每个笔画mi:一张图像，与上述图像不同
				part_num = part_num + 1  # 全部训练集的笔画数累加
				print('part_num',part_num)
				MT=(abs(motor[mi])).astype(int)#每个笔画中数据：坐标及其时间
				#gv=(MT[:,2]-np.min(MT[:,2]))/(np.max(MT[:,2])-np.min(MT[:,2]))

				MTI = np.zeros((II.shape))#每个笔画一张图像
				for i in range(MT.shape[0]):#/ len(motor)  # each stroke phase difference
					if MT[i,0]<105 and MT[i,1]<105:
						MTI[MT[i, 0], MT[i, 1]] = 1
						MTIs[MT[i, 0], MT[i, 1]] = 1
				#pss = Perspective()  # 对笔画图像进行变换
				#MTI = pss(MTI)
				MTI=cv2.dilate(MTI,dilateker,iterations=1)
				trainstroke.append([MTI.T,part_num,class_num])#每个笔画一张图像:笔画及其对应部件、类别

				plt.subplot(1,6,mi+2)
				plt.imshow(-MTI.T,cmap='gray')
				plt.axis('off')

			MTIs = cv2.dilate(MTIs, dilateker, iterations=1)
			trainstrokemg.append([MTIs.T,class_num])#所有笔画一张图像
			plt.subplot(1,6,1)
			plt.imshow(-MTIs.T,cmap='gray')
			plt.axis('off')

			plt.show()

			"""plt.subplot(1, 3, 1)
			plt.imshow(II)
			plt.title('image:' + str(class_num))#str(np.max(I)))
			plt.subplot(1,3,2)
			plt.imshow(MTI.T,cmap='gray')
			plt.title('last_stroke:'+str(part_num))
			plt.subplot(1, 3, 3)
			plt.imshow(MTIs.T,cmap='gray')
			plt.title('all_strokes:' + str(beforestrokenum))#str(np.min(MTIs.T)))
			plt.tight_layout()
			plt.show()"""

if __name__ == "__main__":
	rendition_re(1)