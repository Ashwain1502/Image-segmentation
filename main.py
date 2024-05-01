import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import Min_Cut

class SPNode():
	def __init__(self):
		self.label = None
		self.pixels = []
		self.centroid = ()
		self.type = 'na'
		self.mean_lab = None
		self.lab_hist = None
		self.real_lab = None
	
def mark_pixels(event, x, y, flags, param):
	global drawing, mode, marked_bg_pixels, marked_ob_pixels, I_dummy
	h, w, c = I_dummy.shape

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == "ob":
				if x >= 0 and x <= w-1 and y > 0 and y <= h-1:
					marked_ob_pixels.append((y, x))
				cv2.line(I_dummy, (x-3, y), (x+3, y), (0, 0, 255))
			else:
				if x >= 0 and x <= w-1 and y > 0 and y <= h-1:
					marked_bg_pixels.append((y, x))
				cv2.line(I_dummy, (x-3, y), (x+3, y), (255, 0, 0))
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == "ob":
			cv2.line(I_dummy, (x-3, y), (x+3, y), (0, 0, 255))
		else:
			cv2.line(I_dummy, (x-3, y), (x+3, y), (255, 0, 0))

def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def make_graph(I, SP_list):
	G = nx.Graph()
	s = SP
		K = 0
		region_rad = math.sqrt(len(u.pixels)/math.pi)
		for v in SP_list:
			if u != v:
				if distance(u.centroid, v.centroid) <= 2.5*region_rad:
					sim = math.exp(-(cv2.compareHist(u.lab_hist, v.lab_hist, 3)**2/50))*(1/distance(u.centroid, v.centroid))
					K += sim
					G.add_edge(u, v, sim = sim)
		if u.type == 'na':
			G.add_edge(s, u, sim = 1000000)
			G.add_edge(t, u, sim = 1000000)
		if(u.type == 'ob'):
			G.add_edge(s, u, sim = K+1)
			G.add_edge(t, u, sim = 0)
		if(u.type == 'bg'):
			G.add_edge(s, u, sim = 0)
			G.add_edge(t, u, sim = K+1)		
	return G

drawing = False
mode = "ob"
marked_ob_pixels = []
marked_bg_pixels = []
l_range = [0, 256]
a_range = [0, 256]
b_range = [0, 256]
lab_bins = [32, 32, 32]

inputfile = input("Enter File Name: ")
I = cv2.imread(inputfile)
I_dummy = np.zeros(I.shape)
I_dummy = np.copy(I)
	
h, w, c = I.shape
cv2.namedWindow('Window 1')
cv2.setMouseCallback('Window 1', mark_pixels)
while(1):
	cv2.imshow('Window 1', I_dummy)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('o') or k == ord('O'):
		mode = "ob"
	elif k == ord('b') or k == ord('B'):
		mode = "bg"
	elif k == 27:
		break
cv2.destroyAllWindows()

I_lab = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
SP = cv2.ximgproc.createSuperpixelSLIC(I, algorithm = 101, region_size = 20, ruler = 10)
SP.iterate(num_iterations = 4)
SP_labels = SP.getLabels()
SP_list = [None for x in range(SP.getNumberOfSuperpixels())]

for i in range(h):
	for j in range(w):
		if not SP_list[SP_labels[i][j]]:
			tmp_sp = SPNode()
			tmp_sp.label = SP_labels[i][j]
			tmp_sp.pixels.append((i, j))
			SP_list[SP_labels[i][j]] = tmp_sp
		else:
			SP_list[SP_labels[i][j]].pixels.append((i, j))

for sp in SP_list:
	n_pixels = len(sp.pixels)
	i_sum = 0
	j_sum = 0
	lab_sum = [0, 0, 0]
	tmp_mask = np.zeros((h, w), np.uint8)
	for each in sp.pixels:
		i, j = each
		i_sum += i
		j_sum += j
		lab_sum = [x + y for x, y in zip(lab_sum, I_lab[i][j])]
		tmp_mask[i][j] = 255
	sp.lab_hist = cv2.calcHist([I_lab], [0, 1, 2], tmp_mask, lab_bins, l_range + a_range + b_range)
	sp.centroid += (i_sum//n_pixels, j_sum//n_pixels)
	sp.mean_lab = [x/n_pixels for x in lab_sum]
	sp.real_lab = [sp.mean_lab[0]*100/255, sp.mean_lab[1]-128, sp.mean_lab[2]-128]

for x, y in marked_ob_pixels:
	SP_list[SP_labels[x][y]].type = "ob"
for x, y in marked_bg_pixels:
	SP_list[SP_labels[x][y]].type = "bg"

G = make_graph(I_lab, SP_list)
for node in G.nodes():
	if node.label == 's':
		s=node
	if node.label == 't':
		t=node

RG = Min_Cut.boykov_kolmogorov(G, s, t)
source_tree = RG.graph['trees']
partition = set(source_tree)
F = np.zeros((h, w), dtype = np.uint8)
for sp in partition:
	for x, y in sp.pixels:
		F[x][y] = 1
Final = cv2.bitwise_and(I, I, mask = F)

plt.subplot(1, 3, 1)
plt.imshow(I[..., ::-1])
plt.xlabel("Original Image")

sp_lab = np.zeros(I.shape, dtype = np.uint8)
for sp in SP_list:
	for pixels in sp.pixels:
		i, j = pixels
		sp_lab[i][j] = sp.mean_lab
sp_lab = cv2.cvtColor(sp_lab, cv2.COLOR_Lab2RGB)

plt.subplot(1, 3, 2)
plt.imshow(sp_lab)
plt.xlabel("After Pixels Grouping")

plt.subplot(1, 3, 3)
plt.imshow(Final[..., ::-1])
plt.xlabel("Segmented Image")
plt.show()