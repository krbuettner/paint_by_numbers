# Paint By Numbers Pipeline

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
import time
import cv2
import warnings
import argparse

warnings.filterwarnings('ignore')

IMPORT_PYCUDA = True

if IMPORT_PYCUDA:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule

###########################################################################################################################
# FUNCTIONS
###########################################################################################################################

# Depth-First Search to Identify Connected Componenets in Image

def dfs(img, x_val, y_val, component_num, marked_mask):
	stack = []
	stack.append((x_val,y_val))
	horizontal_change = [-1, 0, 1, 1, 1, 0, -1, 1]
	vertical_change = [1, 1, 1, 0, -1, -1, -1, 0]
	while not len(stack) == 0:
		(x,y) = stack.pop()
		for i in range(8):
			nx = x + horizontal_change[i]
			ny = y + vertical_change[i]
			if img[nx, ny] != 0 and marked_mask[nx, ny] == 0:
				stack.insert(0, (nx, ny))
				marked_mask[nx, ny] = component_num
	return marked_mask

###########################################################################################################################

# BGR to LAB CPU Version

def convert_bgr_to_lab(img):
	assert img.dtype == 'uint8'
	img = img.astype('float32') * (1.0/255.0) 

	# Gamma
	func_sbgr = lambda x : ((x+0.055)/1.055) ** (2.4) if x > 0.04045 else x / 12.92
	vectorized_func_sbgr = np.vectorize(func_sbgr)
	img = vectorized_func_sbgr(img)

	# Convert to XYZ and scale
	transform_matrix = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
	xyz = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			xyz[i, j, :] = np.matmul(transform_matrix, np.flip(img[i, j, :]))
	xyz[:, :, 0] = xyz[:, :, 0] / (0.950456)
	xyz[:, :, 2] = xyz[:, :, 2] / (1.088754)

	# Get L, a, b
	new_img = np.zeros(img.shape)
	func_L = lambda y : 116*(y**(1.0/3)) - 16 if y > 0.008856 else 903.3*y
	vectorized_func_L = np.vectorize(func_L)
	f_t = lambda t : t**(1.0/3) if t > 0.008856 else 7.787*t + (16.0/116)
	vectorized_func_f_t = np.vectorize(f_t)
	delta = 0
	new_img[:, :, 0] = vectorized_func_L(xyz[:, :, 1])
	new_img[:, :, 1] = 500 * (vectorized_func_f_t(xyz[:,:,0]) - vectorized_func_f_t(xyz[:,:,1])) + delta
	new_img[:, :, 2] = 200 * (vectorized_func_f_t(xyz[:,:,1]) - vectorized_func_f_t(xyz[:,:,2])) + delta

	# Scale
	new_img[:, :, 0] = (new_img[:, :, 0] * (255.0/100)).astype('uint8')
	new_img[:, :, 1] = (new_img[:, :, 1] + 128).astype('uint8')
	new_img[:, :, 2] = (new_img[:, :, 2] + 128).astype('uint8')

	return new_img

# BGR to LAB GPU Version

def convert_bgr_to_lab_gpu(img):

	mod = SourceModule("""
	__global__ void convert_bgr_to_lab(float *out_l, float *out_a, float *out_b, float *b, float *g, float *r)
	{
	  int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	  float local_b = 0;
	  float local_g = 0;
	  float local_r = 0;
	  if (b[index] > 0.04045) {
	      local_b = pow((b[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_b = b[index] / 12.92;
	  }
	  if (g[index] > 0.04045) {
	      local_g = pow((g[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_g = g[index] / 12.92;
	  }
	  if (r[index] > 0.04045) {
	      local_r = pow((r[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_r = r[index] / 12.92;
	  }
	  float transform_matrix[3][3] = {0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227};
	  float temp_sum = 0;
	  const int row_size = blockDim.x;
	  const int col_size = gridDim.x;
	  int i = (index / row_size);
	  int j = (index % row_size);
	  float local_x = (0.412453 * local_r + 0.357580 * local_g + 0.180423 * local_b) / 0.950456;
	  float local_y = 0.212671 * local_r + 0.715160 * local_g + 0.072169 * local_b;
	  float local_z = (0.019334 * local_r + 0.119193 * local_g + 0.950227 * local_b) / 1.088754;

	  float f_x = 0;
	  float f_y = 0;
	  float f_z = 0;
	  float delta = 0;
	  if (local_x > 0.008856){
	      f_x = cbrtf(local_x);
	  } else {
          f_x = 7.787*local_x + (16.0/116.0);
	  }
	  if (local_y > 0.008856){
	      f_y = cbrtf(local_y);
	  } else {
          f_y = 7.787*local_y + (16.0/116.0);
	  }
	  if (local_z > 0.008856){
	      f_z = cbrtf(local_z);
	  } else {
          f_z = 7.787*local_z + (16.0/116.0);
	  }
	  if (local_y > 0.008856){
	  	  out_l[index] = 116 * cbrtf(local_y) - 16;
	  }
	  else {
	  	  out_l[index] = 903.3 * local_y;
	  }
	  out_a[index] = 500 * (f_x - f_y) + delta;
	  out_b[index] = 200 * (f_y - f_z) + delta;
	}
	""")
	convert_bgr_to_lab = mod.get_function('convert_bgr_to_lab') 
	img = img.astype('float32') * (1.0/255.0)
	b = img[:, :, 0].flatten().astype('float32')
	g = img[:, :, 1].flatten().astype('float32') 
	r = img[:, :, 2].flatten().astype('float32')
	out_l = np.zeros_like(b)
	out_a = np.zeros_like(g)
	out_b = np.zeros_like(r)
	convert_bgr_to_lab(drv.Out(out_l), drv.Out(out_a), drv.Out(out_b), drv.In(b), drv.In(g), drv.In(r), block=(w,1,1), grid=(h,1,1))
	new_img = np.zeros_like(img)
	new_img[:, :, 0] = (np.reshape(out_l, (h, w)) * 255.0/100).astype('uint8')
	new_img[:, :, 1] = (np.reshape(out_a, (h, w)) + 128).astype('uint8')
	new_img[:, :, 2] = (np.reshape(out_b, (h, w)) + 128).astype('uint8')

	return new_img

###########################################################################################################################

# Outline CPU Version -> Marks Approximate Edges as Black Pixels

def outline_cpu(img):

	(h,w) = (img.shape[0], img.shape[1])
	canvas_image = np.ones((h,w,3), np.uint8) * 255
	outline_image = np.ones((h,w), np.uint8) * 255

	for i in range(w):
		canvas_image[0][i] = [0, 0, 0]
		canvas_image[h-1][i] = [0, 0, 0]

	for i in range(h):
		canvas_image[i][0] = [0, 0, 0]
		canvas_image[i][w-1] = [0, 0, 0]

	for i in range(1, h - 1):
		for j in range(1, w - 1):
			pixel_val = img[i][j]
			if not np.array_equal(pixel_val, img[i-1][j-1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i-1][j]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i-1][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i][j-1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i+1][j-1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i+1][j]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, img[i+1][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			else:
				canvas_image[i, j] = img[i, j]

	return canvas_image, outline_image

# Outline GPU Version

def outline_gpu(img):

	(h,w) = (img.shape[0], img.shape[1])

	mod = SourceModule("""
	__global__ void outline(int *border, int *out_b, int *out_g, int *out_r, int *b, int *g, int *r)
	{
	  int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	  int curr_b = b[index];
	  int curr_g = g[index];
	  int curr_r = r[index];
	  const int row_size = blockDim.x;
	  const int col_size = gridDim.x;
	  int i = (index / row_size);
	  int j = (index % row_size);
	  if (i >= 1 && j >= 1 && i < col_size - 1 && j < row_size - 1){
	  	  if (curr_b != b[(i-1)*row_size+(j-1)] || curr_g != g[(i-1)*row_size+(j-1)] || curr_r != r[(i-1)*row_size+(j-1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i-1)*row_size+(j)] || curr_g != g[(i-1)*row_size+(j)] || curr_r != r[(i-1)*row_size+(j)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i-1)*row_size+(j+1)] || curr_g != g[(i-1)*row_size+(j+1)] || curr_r != r[(i-1)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i)*row_size+(j-1)] || curr_g != g[(i)*row_size+(j-1)] || curr_r != r[(i)*row_size+(j-1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i)*row_size+(j+1)] || curr_g != g[(i)*row_size+(j+1)] || curr_r != r[(i)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j-1)] || curr_g != g[(i+1)*row_size+(j-1)] || curr_r != r[(i+1)*row_size+(j-1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j)] || curr_g != g[(i+1)*row_size+(j)] || curr_r != r[(i+1)*row_size+(j)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j+1)] || curr_g != g[(i+1)*row_size+(j+1)] || curr_r != r[(i+1)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else {
	  	  		out_b[index] = curr_b;
	  	  		out_g[index] = curr_g;
	  	  		out_r[index] = curr_r;
	  	  		border[index] = 255;
	  	  }
	  }
	  else {
	  	  border[index] = 0;
	  }
	}
	""")
	outline = mod.get_function('outline') 
	b = img[:, :, 0].flatten().astype('uint32')
	g = img[:, :, 1].flatten().astype('uint32') 
	r = img[:, :, 2].flatten().astype('uint32')
	out_b = np.zeros_like(b)
	out_g = np.zeros_like(g)
	out_r = np.zeros_like(r)
	border = np.ones_like(b) * 255
	outline(drv.Out(border), drv.Out(out_b), drv.Out(out_g), drv.Out(out_r), drv.In(b), drv.In(g), drv.In(r), block=(w,1,1), grid=(h,1,1))
	canvas = np.zeros_like(img)
	canvas[:, :, 0] = np.reshape(out_b, (h, w))
	canvas[:, :, 1] = np.reshape(out_g, (h, w))
	canvas[:, :, 2] = np.reshape(out_r, (h, w))
	border = np.reshape(border, (h, w))

	return canvas, border

###########################################################################################################################

# Main Function

def main():

	parser = argparse.ArgumentParser(description='Paint By Numbers Outline Script')
	parser.add_argument('--use_gpu', action='store_true', default=False)
	parser.add_argument('--img_folder', type=str, default='test_images/')
	parser.add_argument('--img_name', type=str, default='crosby.jpg')
	parser.add_argument('--resize_width', type=int, default=256)
	parser.add_argument('--resize_height', type=int, default=256)
	parser.add_argument('--num_colors', type=int, default=5)
	parser.add_argument('--blur', type=str, default='median') # or 'gaussian'
	parser.add_argument('--no_plot', action='store_true', default=False)
	parser.add_argument('--use_custom_bgr_to_lab', action='store_true', default=False)
	parser.add_argument('--save_img', action='store_true', default=False)
	args = parser.parse_args()

	use_gpu = args.use_gpu
	img_folder = args.img_folder
	img_name = args.img_name
	full_img_path = img_folder + img_name	
	resize_width = args.resize_width
	resize_height = args.resize_height	
	num_colors = args.num_colors
	blur = args.blur
	show_results = not args.no_plot
	use_custom_bgr_to_lab = args.use_custom_bgr_to_lab
	save_img = args.save_img		

	start_pipeline = time.clock()

	print('Reading Image...')
	original_image = cv2.imread(full_img_path)

	print('Resizing Image...')
	resized_image = cv2.resize(original_image, (resize_width, resize_height))
	(h, w) = (resize_height, resize_width)

	# Quantization Stage
	quant_start = time.clock()

	print('Quantizing Image...')
	print('...Using ' + str(num_colors) + ' Colors')
	print('...Converting BGR to LAB Image...')
	color_cvt_bgr_to_lab_start = time.clock()
	if use_custom_bgr_to_lab:
		temp_image = resized_image.copy()
		if use_gpu:
			print('...Using Custom GPU BGR to LAB...')
			quantized_image = convert_bgr_to_lab_gpu(temp_image)
		else:
			print('...Using Custom CPU BGR to LAB...')
			quantized_image = convert_bgr_to_lab(temp_image)
	else:
		print('...Using OpenCV BGR to LAB')
		quantized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
	color_cvt_bgr_to_lab_end = time.clock()

	print('...K-Means + Assignment of Colors ...')
	kmeans_start = time.clock()
	quantized_image = quantized_image.reshape((h*w, 3))
	clusters = MiniBatchKMeans(n_clusters=num_colors, random_state=3)				
	labels = clusters.fit_predict(quantized_image)
	quantized_image = clusters.cluster_centers_.astype('uint8')[labels]
	quantized_image = quantized_image.reshape((h, w, 3))
	kmeans_end = time.clock()

	print('...Converting LAB to BGR Image...')
	color_cvt_lab_to_bgr_start = time.clock()
	quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)
	color_cvt_lab_to_bgr_end = time.clock()

	quant_end = time.clock()		

	# Smoothing Phase
	print('Smoothing image...')
	blur_start = time.clock()
	if blur == 'gaussian':
		blurred_image = cv2.GaussianBlur(quantized_image, (5,5), 0) 						
	else:
		blurred_image = cv2.medianBlur(quantized_image, 5) 						
	blur_end = time.clock()

	print('Generating outline...')
	outline_start = time.clock()
	if use_gpu:
		canvas_image, outline_image = outline_gpu(blurred_image)
	else:
		canvas_image, outline_image = outline_cpu(blurred_image)
	outline_end = time.clock()

	# Get Connected Components -> This is to demonstrate different regions that can be colored in
	print('Retrieving components...')
	ccl_start = time.clock()
	canvas_image_gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
	marked_mask = np.zeros((h,w))
	component_num = 1
	for i in range(1, h-1):
		for j in range(1, w-1):
			if canvas_image_gray[i, j] != 0  and marked_mask[i, j] == 0:
				marked_mask = dfs(canvas_image_gray, i, j, component_num, marked_mask)
				component_num = component_num + 1
	print('...Num of Components: ' + str(component_num))
	ccl_end = time.clock()

	end_pipeline = time.clock()

	# Final Results
	if show_results:
		# Set up Images for Plotting
		marked_mask = marked_mask.astype('uint8') * 20						# This is dummy line to show the 'marked_mask'; not robust
		marked_mask = cv2.cvtColor(marked_mask, cv2.COLOR_GRAY2BGR)
		outline_image = cv2.cvtColor(outline_image.astype('uint8'), cv2.COLOR_GRAY2BGR)
		row_1 = np.hstack([resized_image, quantized_image, blurred_image])
		row_2 = np.hstack([outline_image, canvas_image, marked_mask])
		images_to_show = np.vstack([row_1, row_2])
		cv2.imshow('Paint By Numbers Pipeline', images_to_show)
		cv2.waitKey(0)
		if save_img:
			cv2.imwrite('sample_results/' + img_name, images_to_show)
		cv2.destroyAllWindows()
	
	# Print Timing
	print('Timing...')
	print('...1) Color Quantization:\t\t ' + str(quant_end - quant_start) + ' s')
	print('...> Convert BGR to LAB:\t\t ' + str(color_cvt_bgr_to_lab_end - color_cvt_bgr_to_lab_start) + ' s')
	print('...> K-Means + Assignment of Colors:\t ' + str(kmeans_end - kmeans_start) + ' s')
	print('...> Convert LAB to BGR:\t\t ' + str(color_cvt_lab_to_bgr_end - color_cvt_lab_to_bgr_start) + ' s')
	print('...2) Blur Filter:\t\t\t ' + str(blur_end - blur_start) + ' s')
	print('...3) Outlining:\t\t\t ' + str(outline_end - outline_start) + ' s')
	print('...4) Connected Components:\t\t ' + str(ccl_end - ccl_start) + ' s')
	print('...Total Pipeline Time:\t\t\t ' + str(end_pipeline - start_pipeline) + ' s')

if __name__ == '__main__':
	main()