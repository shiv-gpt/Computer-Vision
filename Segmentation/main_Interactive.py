#Citation - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
#Citation - http://answers.opencv.org/question/56988/draw-with-mouse-continuously/

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

np.set_printoptions(threshold=np.inf)
ldrawing = False
rdrawing = False
flag = True

lflag = 0
rflag = 0

def mouse_callback_1(event, x, y, flags, param):
    global lx,ly,ldrawing,rx,ry,rdrawing,lflag,rflag,flag
    if event == cv2.EVENT_LBUTTONDOWN:
        ldrawing = True
        flag = True
        lx = x
        ly = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        rdrawing = True
        flag = False
        rx = x
        ry = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if flag == True and ldrawing == True:
            cv2.line(tpimg, (lx,ly), (x,y), (255,0,0), 5)
            lx = x
            ly = y
        elif flag == False and rdrawing == True:
            cv2.line(tpimg, (rx,ry), (x,y), (0,0,255), 5)
            rx = x
            ry = y
    elif event == cv2.EVENT_LBUTTONUP:
        ldrawing = False        
        cv2.line(tpimg, (lx,ly), (x,y), (255,0,0), 5)
        lflag+=1
    elif event == cv2.EVENT_RBUTTONUP:
        rdrawing = False
        cv2.line(tpimg, (rx,ry), (x,y), (0,0,255), 5)
        rflag+=1
    if rflag >= 2 and lflag >= 2:
        segment(tpimg, superpixels, norm_hists, neighbours)

def segment(img_marking, superpixels, norm_hists, neighbours):
    print "RFLAG = " + str(rflag)
    print "LFLAG = " + str(lflag)
    print "Here"
    img_input1 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    temp = img_marking
    imgones = np.zeros((256,256,3))
    imgones.fill(255)
    for i in range(256):
        for j in range(256):
            if(temp[i,j,0] == 255):
                imgones[i,j,0] = 255
                imgones[i,j,1] = 0
                imgones[i,j,2] = 0
            elif temp[i,j,2] == 255:
                imgones[i,j,0] = 0
                imgones[i,j,1] = 0
                imgones[i,j,2] = 255
    img_marking = imgones

    cv2.namedWindow('marking image')
    cv2.imshow('marking image', img_marking)

    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)
    #print superpixels
    #print fg_segments
    #print bg_segments
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

    
    #norm_fg_hists = normalize_histograms(fg_cumulative_hist)
    #norm_bg_hists = normalize_histograms(bg_cumulative_hist)
    A = [fg_cumulative_hist, bg_cumulative_hist]
    B = [fg_segments, bg_segments]

    graph_cut = do_graph_cut(A, B, norm_hists, neighbours)
    mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
    mask = np.uint8(mask*255)
    cv2.namedWindow('Segmentation Output')
    cv2.imshow('Segmentation Output', mask)


def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.48)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    img_input = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print events
    tpimg = img_input
    centers, color_hists, superpixels, neighbours = superpixels_histograms_neighbors(img_input)
    norm_hists = normalize_histograms(color_hists)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback_1)
    
    while(1):
        cv2.imshow('image', tpimg)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
