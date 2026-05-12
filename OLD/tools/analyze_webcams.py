#This script is for analyzing the FAA webcams dataset. It outputs total image counts for each class, each site, and counts of those per the other.

import torch
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.realpath('..'))

from dsets import Webcams

CLASS_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
DSET_DIR = '/home/feet/Documents/LAWN/datasets/quality-labeled-webcams/by-network/good'
limits = dict()

info_list = []
dset = Webcams.Webcams_cls_10(DSET_DIR, lambda x: x, limits)

for i, file in enumerate(dset.files):
    dir_name = os.path.basename(os.path.dirname(file))
    file_name = os.path.basename(file)
    
    try:
        timestamp = datetime.strptime(dir_name, '%Y-%m-%d-%H-%M-%S')
    except:
        timestamp = 'N/A'
    
    name = file_name[0:-4]
    site, ornt, vis = tuple(name.split('_'))
    
    vis = CLASS_VALUES[torch.argmax(dset.labels[i]).item()]
    
    info_list += [(site, ornt, vis, timestamp)]

#Get counts of visibilites
vis_counts = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:0}
for info in info_list:
    vis_counts[info[2]] += 1

#Get counts of sites
site_counts = dict()
for info in info_list:
    if info[0] not in site_counts:
        site_counts[info[0]] = 0
    site_counts[info[0]] += 1

#Get visibility counts for each site
site_vis_counts = dict()
for info in info_list:
    if info[0] not in site_vis_counts:
        site_vis_counts[info[0]] = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10.0:0}
    site_vis_counts[info[0]][info[2]] += 1

#Get site counts for each visibility
vis_site_counts = dict()
for info in info_list:
    if info[2] not in vis_site_counts:
        vis_site_counts[info[2]] = dict()
    if info[0] not in vis_site_counts[info[2]]:
        vis_site_counts[info[2]][info[0]] = 0
    vis_site_counts[info[2]][info[0]] += 1

# print('site,count')
# for site in site_counts:
#     print(site, site_counts[site], sep=',')

# print('site,1,2,3,4,5,6,7,8,9,10')
# for site in site_vis_counts:
#     print(site, end='')
#     for i in range(10):
#         print(',' + str(site_vis_counts[site][float(i+1)] / vis_counts[float(i+1)]), end='')
#     print('')


#Y: VIS X: SITES
# for i in range(1,11):
#     total = vis_counts[float(i)]
#     vis_site_counts[float(i)] = {k: v / total for k, v in vis_site_counts[float(i)].items()}

# data = np.zeros((10, len(site_counts)), dtype=float)

# for i in range(10):
#     for j, site in enumerate(site_counts.keys()):
#         if site in vis_site_counts[float(i+1)]:
#             data[i][j] = vis_site_counts[float(i+1)][site]
#         else:
#             data[i][j] = 0.0

# data_frame = pd.DataFrame(data, index=class_values)
# # plt.figure(dpi=300)
# plot = sns.heatmap(data_frame, annot=False, vmin=0.0, vmax=1.0)
# plot.set_xlabel("Site")
# plot.set_ylabel("Visibility")
# plt.show()


#Y: SITES X: VISfor i, site in enumerate(site_counts):
#     total = site_counts[site]
#     site_vis_counts[site] = {k: v / total for k, v in site_vis_counts[site].items()}

# data = np.zeros((len(site_counts), 10), dtype=float)

# for i, site in enumerate(site_counts.keys()):
#     for j in range(10):
#         data[i][j] = site_vis_counts[site][float(j+1)]

# data = data.swapaxes(1, 0)
# data_frame = pd.DataFrame(data, index=class_values)
# # plt.figure(dpi=300)
# plot = sns.heatmap(data_frame, annot=False, vmin=0.0, vmax=1.0)
# plot.set_xlabel("Site")
# plot.set_ylabel("Visibility")
# plt.show()

# data = np.zeros((len(site_counts), 10), dtype=float)

# for i, site in enumerate(site_counts.keys()):
#     for j in range(10):
#         data[i][j] = site_vis_counts[site][float(j+1)]

# data = data.swapaxes(1, 0)
# data_frame = pd.DataFrame(data, index=class_values)
# # plt.figure(dpi=300)
# plot = sns.heatmap(data_frame, annot=False, vmin=0.0, vmax=1.0)
# plot.set_xlabel("Site")
# plot.set_ylabel("Visibility")
# plt.show()

# for site in site_counts.keys():
#     print(site, site_counts[site], sep=',')

for site in site_vis_counts.keys():
    print(site, '"'+str(site_vis_counts[site])+'"', site_counts[site], sep=',')