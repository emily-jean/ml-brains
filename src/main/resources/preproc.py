#!/usr/bin/python3

# Imports
import numpy as np
import sys

def centerSlice(img,crop_size=3):
    """Take the center slice of a image."""
    """crop_size defaults to 3"""
    y,x = img.shape
    startx = x//2 - crop_size//2
    starty = y//2 - crop_size//2 
    return img[starty:starty+crop_size, startx:startx+crop_size]


def preProcLine(line, crop_size=7, hist_thres=50):
    """This function pre process the line."""
    """It will yield a pre-processed feature vector"""
    
    l = np.fromstring(line, dtype=int, sep=',')
    if (l.shape == (3087,)):
        # These lines are short in one element
        # due to the '?' label
        # for raw validation set
        #thus we append a dummy zero label here
        l = np.append(l, 0)

    input_data_line=np.array(l[:-1])
    img3d = input_data_line.reshape(21,21,7, order='F')
    label=np.array(l[-1])

    feature_array = np.zeros(11, dtype=float)

    img_center_pixel_value = img3d[10,10,3]

    # --------------------------------------------------------------------------
    # XY center plane
    img=None; hist=None; bins=None; magnitude_spectrum=None; 
    img = img3d[:,:,3]

    # Histogram calc
    hist,bins = np.histogram(img.ravel(),256,[0,256])

    # FFT calculation
    magnitude_spectrum = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))

    # Statistics calculation
    xy_img_center_sclice_mean = np.mean(centerSlice(img,crop_size))
    xy_fft_center_sclice_mean = np.mean(centerSlice(magnitude_spectrum,crop_size))
    xy_hist_count_over_thres = np.sum(hist[hist_thres:])

    # --------------------------------------------------------------------------
    img=None; hist=None; bins=None; magnitude_spectrum=None; 
    # XZ center plane
    img = img3d[:,10,:]

    # Histogram calc
    hist,bins = np.histogram(img.ravel(),256,[0,256])

    # FFT calculation
    magnitude_spectrum = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))

    # Statistics calculation
    xz_img_center_sclice_mean = np.mean(centerSlice(img,crop_size))
    xz_fft_center_sclice_mean = np.mean(centerSlice(magnitude_spectrum,crop_size))
    xz_hist_count_over_thres = np.sum(hist[hist_thres:])

    # --------------------------------------------------------------------------
    # YZ center plane
    img=None; hist=None; bins=None; magnitude_spectrum=None; 

    img = img3d[10,:,:]

    # Histogram calc
    hist,bins = np.histogram(img.ravel(),256,[0,256])

    # FFT calculation
    magnitude_spectrum = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))

    # Statistics calculation
    yz_img_center_sclice_mean = np.mean(centerSlice(img,crop_size))
    yz_fft_center_sclice_mean = np.mean(centerSlice(magnitude_spectrum,crop_size))
    yz_hist_count_over_thres = np.sum(hist[hist_thres:])

    # Storing on the feature vector
    feature_array[0] = img_center_pixel_value

    feature_array[1] = xy_img_center_sclice_mean
    feature_array[2] = xy_fft_center_sclice_mean
    feature_array[3] = xy_hist_count_over_thres

    feature_array[4] = xz_img_center_sclice_mean
    feature_array[5] = xz_fft_center_sclice_mean
    feature_array[6] = xz_hist_count_over_thres

    feature_array[7] = yz_img_center_sclice_mean
    feature_array[8] = yz_fft_center_sclice_mean
    feature_array[9] = yz_hist_count_over_thres

    feature_array[10] = label
    
    #pre_proc_data = []
    #pre_proc_data.append(feature_array)


    return feature_array


def featuresToString(feature_array, separator=','):
    """Return a one line string with the feature array."""
    
    output_string = "{:.3f}".format(feature_array[0])

    for f in feature_array[1:]:
        output_string = output_string+ "{}{:.3f}".format(separator, f)

    return output_string
    
def main():
    for line in sys.stdin:
        print(featuresToString(preProcLine(line)))

if __name__== "__main__":
      main()
