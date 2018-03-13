"""
Citation:
1. Professor's helper functions and hints provided on piazza.
2. http://docs.opencv.org/3.2.0/dc/dff/tutorial_py_pyramids.html
3. http://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
4. http://cache.freescale.com/files/dsp/doc/app_note/AN4318.pdf
"""

import os
import sys
import cv2
import numpy

def help_message():
   print("Usage: [Option Number] [Input_Options] [Output_Options]")
   print("[Options]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Option 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   

   red = numpy.zeros((256,), dtype=numpy.int)
   green = numpy.zeros((256,), dtype=numpy.int)
   blue = numpy.zeros((256,), dtype=numpy.int)

   cdfred = numpy.zeros((256,), dtype=numpy.int)
   cdfgreen = numpy.zeros((256,), dtype=numpy.int)
   cdfblue = numpy.zeros((256,), dtype=numpy.int)


   #shape = ()
   #print img_in.shape
   rows, cols, channels = img_in.shape
   for i in range(rows):
      for j in range(cols):
         red[img_in[i,j,2]] = red[img_in[i,j,2]] + 1
         green[img_in[i,j,1]] = green[img_in[i,j,1]] + 1
         blue[img_in[i,j,0]] = blue[img_in[i,j,0]] + 1
   
   #print(red)

   for i in range(256):
      for j in range(i):
         cdfred[i] = cdfred[i] + red[j]
         cdfgreen[i] = cdfgreen[i] + green[j]
         cdfblue[i] = cdfblue[i] + blue[j]
   
   #print(cdfred)
   #print(img_in.size)
   for i in range(256):
      cdfred[i]= (cdfred[i]*255)/(img_in.size/3)
      cdfgreen[i]= (cdfgreen[i]*255)/(img_in.size/3)
      cdfblue[i]= (cdfblue[i]*255)/(img_in.size/3)
  
   #print(cdfred)
   tmpimg = img_in.copy()

   for i in range(rows):
      for j in range(cols):
         tmpimg[i,j,0] = cdfblue[tmpimg[i,j,0]]
         tmpimg[i,j,1] = cdfgreen[tmpimg[i,j,1]]
         tmpimg[i,j,2] = cdfred[tmpimg[i,j,2]]

   
   img_out = tmpimg.copy()


   return True, img_out
   
def Option1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   #if (input_image):
   #   print("Image read")
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Option 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
	
   dft_image = numpy.fft.fft2(numpy.float32(img_in), (img_in.shape[0], img_in.shape[1])) 
   dft_image = numpy.fft.fftshift(dft_image)  
   #print "dft Shape = " + str(dft_image.shape)

   dft_mask = numpy.zeros((dft_image.shape[0],dft_image.shape[1]), numpy.uint8)

   #print "mask Shape = " + str(dft_mask.shape)
   #print "dft Shape = " + str(dft_image.shape)
   rows = dft_mask.shape[0]
   cols = dft_mask.shape[1]
   dft_mask[rows/2 - 10 : rows/2 + 10, cols/2 - 10 : cols/2 + 10] = 1
   dft_image = dft_image * dft_mask
   

   dft_image = numpy.fft.ifftshift(dft_image)
   idft_image = numpy.fft.ifft2(dft_image)

   img_out = numpy.abs(idft_image)
   
	
   return True, img_out

def high_pass_filter(img_in):

   dft_image = numpy.fft.fft2(numpy.float32(img_in), (img_in.shape[0], img_in.shape[1])) 
   dft_image = numpy.fft.fftshift(dft_image)   
   #print "dft Shape = " + str(dft_image.shape)
   rows = dft_image.shape[0]
   cols = dft_image.shape[1]
   dft_image[rows/2 - 10 : rows/2 + 10, cols/2 - 10 : cols/2 + 10] = 0

   dft_image = numpy.fft.ifftshift(dft_image)
   idft_image = numpy.fft.ifft2(dft_image)

   img_out = numpy.abs(idft_image)
   
   return True, img_out
   
def deconvolution(img_in):
   
   #Generate Gaussian Kernel
   gaussian_kernel = cv2.getGaussianKernel(21, 5)
   gaussian_kernel = gaussian_kernel * gaussian_kernel.T

   dft_image = numpy.fft.fft2(numpy.float32(img_in), (img_in.shape[0], img_in.shape[1])) 
   dft_image = numpy.fft.fftshift(dft_image)

   dft_kernel = numpy.fft.fft2(numpy.float32(gaussian_kernel), (img_in.shape[0], img_in.shape[1])) 
   dft_kernel = numpy.fft.fftshift(dft_kernel)

   conv = dft_image / dft_kernel

   conv = numpy.fft.ifftshift(conv)
   iconv = numpy.fft.ifft2(conv)

   img_out = numpy.abs(iconv) * 255
   #print "Output shape = " + str(img_out.shape)
   
   return True, img_out

def Option2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Option 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   l = []
   l.append(img_in1.shape[0])
   l.append(img_in1.shape[1])
   l.append(img_in2.shape[0])
   l.append(img_in2.shape[1])
   minval = 10000
   for i in range(4):
      if l[i]%2 == 0 and minval > l[i]:
         minval = l[i]


   img1 = img_in1.copy()
   img2 = img_in2.copy()

   #make images rectangular
   img1 = img1[:minval, :minval]
   img2 = img2[:minval, :minval]

   #img1 = img1[:200, :200]
   #img2 = img2[:200, :200]

   #print img1.shape
   #print img2.shape

   #make Gaussian pyramids for both images

   gaussian_a = []
   img1t = img1.copy()
   gaussian_a.append(img1t)
   i = 1
   while 1:
      j = i-1
      img1t = cv2.pyrDown(gaussian_a[j])
      rows,cols,channels = img1t.shape
      if rows%2 == 0 and cols%2 == 0:  #the loop will break at odd number of rows or columns as doing pyrUp from them would be difficult
         gaussian_a.append(img1t)
      else:
         gaussian_a.append(img1t)
         break
      i = i + 1

   #print len(gaussian_a)

   gaussian_b = []
   img2t = img2.copy()
   gaussian_b.append(img2t)
   i = 1
   while 1:
      j = i-1
      img2t = cv2.pyrDown(gaussian_b[j])
      rows,cols,channels = img2t.shape
      if rows%2 == 0 and cols%2 == 0:
         gaussian_b.append(img2t)
      else:
         gaussian_b.append(img2t)
         break
      i = i + 1

   #print len(gaussian_b)

   #make laplacian pyramids

   laplacian_a = [gaussian_a[len(gaussian_a) - 1]]   
   for i in range(len(gaussian_a) - 1, 0, -1):
      j = i-1
      #print "i =" + str(i)
      #print "j =" + str(j)
      img1t = cv2.pyrUp(gaussian_a[i])
      laplacian_a.append(cv2.subtract(gaussian_a[j], img1t))

   laplacian_b = [gaussian_b[len(gaussian_b) - 1]]
   for i in range(len(gaussian_b) - 1, 0, -1):
      j = i-1
      #print "i =" + str(i)
      #print "j =" + str(j)
      img2t = cv2.pyrUp(gaussian_b[i])
      laplacian_b.append(cv2.subtract(gaussian_b[j], img2t))

   #print "Length =" + str(len(laplacian_b))

   combined = []
   for i in range(min(len(gaussian_a), len(gaussian_b))):
      a = laplacian_a[i]
      b = laplacian_b[i]
      rows, cols, channels = a.shape
      #print "shape of a = " + str(a.shape)
      #print "shape of b = " + str(b.shape)
      cmbnd = numpy.concatenate((a[:,0:cols/2], b[:,cols/2:]), 1)
      #print "shape of cmbnd = " + str(cmbnd.shape)
      combined.append(cmbnd)

   out = combined[0]
   #print "size of out before loop =" + str(out.shape)
   for i in range(1,len(combined)):
      out = cv2.pyrUp(out)
      #print "size of out =" + str(out.shape)
      out = cv2.add(combined[i], out)

   img_out = out

   return True, img_out

def Option3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   Option_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      Option_number = int(sys.argv[1])
	  
      if (Option_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (Option_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (Option_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (Option_number > 3 or Option_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Option1,
   2 : Option2,
   3 : Option3,
   }

   # Call the function
   function_launch[Option_number]()
