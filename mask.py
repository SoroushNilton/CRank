
import torch
import numpy as np

from utils import Sor_dictionary
import pickle5 as pickle

pickle_file = 'POST_CLEAN_JUST_R1_MEDOIDandBELOWAVG.pickle'


with open(pickle_file, 'rb') as handle:
    dict_of_masks = pickle.load(handle)
    

############## Added One #############################
#print("dict_of_masks",dict_of_masks)

# CP Rate
list_ = []
for i in range(1,13):
    if i < 3:
#         print('64')
        list_.append(( 64 - len(dict_of_masks[i])) / 64)
    elif 2 < i <5:
#         print('128')
        list_.append(( 128 - len(dict_of_masks[i])) / 128)
    elif 4 < i < 8:
#         print('256')
        list_.append(( 256 - len(dict_of_masks[i])) / 256)
    elif i > 7:
#         print('512')
        list_.append(( 512 - len(dict_of_masks[i])) / 512)
list_.append(( 512 - len(dict_of_masks[12])) / 512)

size_of_layer = list_ #[64,64,128,128,256,256, 256, 512, 512, 512, 512,512]
# print(size_of_layer)
# Commneted Section to test ======= Un-comment =====================================


# list_ = []
# for i, item_size in enumerate(size_of_layer):
#     #     print(i+1, ' ========= ',item_size)
#     list_.append((item_size - len(dict_of_masks[i + 1])) / item_size)

# Commneted Section to test ======= Un-comment ======================================
# list_ = [0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10
compress_rate = size_of_layer


# Mask
class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=4,  arch="vgg_16_bn"):
        params = self.model.parameters()
#         path_to_ranks  = "./VGG16/"

#         prefix = "rank_conv/old/"+arch+"/rank_conv"
#         subfix = ".npy"
        
#         print('cov_id is ===>', cov_id)
#         print('staying filters are: ')
#         print(dict_of_masks[cov_id])
        
        
        
        
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
#                 print('item.size is ===> ', item.size())
#                 print('f is ===> ', f)
#                 rank = np.load(path_to_ranks + prefix + str(cov_id) + subfix)
#                 rank = np.load(prefix + str(cov_id) + subfix)
#                 pruned_num = int(self.compress_rate[cov_id - 1] * f)
#                 ind = np.argsort(rank)[pruned_num:]  # preserved filter id


                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(int(f)):
                    if i in dict_of_masks[cov_id]:
                        zeros[i, 0, 0, 0] = 1.
                        
                    else:
                        pass
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]
#                 print('Zeros is: ')
#                 print(zeros)



#                 zeros = torch.zeros(f, 1, 1, 1).to(self.device)
#                 for i in range(len(ind)):
#                     zeros[ind[i], 0, 0, 0] = 1.
#                 self.mask[index] = zeros  # covolutional weight
#                 item.data = item.data * self.mask[index]

            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight



class mask_resnet_56:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_56"):
        params = self.model.parameters()
#         prefix = "rank_conv/ours/"+ arch +"/rank_conv"
#         subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
#                 rank = np.load(prefix + str(cov_id) + subfix)
#                 pruned_num = int(self.compress_rate[cov_id - 1] * f)
#                 ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                
        
                for i in range(int(f)):
                    if i in dict_of_masks[cov_id]:
                        zeros[i, 0, 0, 0] = 1.
                        
                    else:
                        pass
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]
        
#                 for i in range(len(ind)):
#                     zeros[ind[i], 0, 0, 0] = 1.
#                 self.mask[index] = zeros  # covolutional weight
#                 item.data = item.data * self.mask[index]    


            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index].to(self.device)

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight




class mask_densenet_40:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.job_dir=job_dir
        self.device=device
        self.mask = {}

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="densenet_40"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            # prune BN's parameter
            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
                # if this BN not belong to 1st conv or transition conv --> add pre-BN mask to this mask
                if cov_id>=2 and cov_id!=14 and cov_id!=27:
                    self.mask[index] = torch.cat([self.mask[index-param_per_cov], torch.squeeze(zeros)], 0).to(self.device)
                else:
                    self.mask[index] = torch.squeeze(zeros).to(self.device)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)


class mask_googlenet:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=28,  arch="googlenet"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == (cov_id-1) * param_per_cov + 4:
                break
            if (cov_id==1 and index==0)\
                    or index == (cov_id - 1) * param_per_cov - 24 \
                    or index == (cov_id - 1) * param_per_cov - 16 \
                    or index == (cov_id - 1) * param_per_cov - 8 \
                    or index == (cov_id - 1) * param_per_cov - 4 \
                    or index == (cov_id - 1) * param_per_cov:

                if index == (cov_id - 1) * param_per_cov - 24:
                    rank = np.load(prefix + str(cov_id)+'_'+'n1x1' + subfix)
                elif index == (cov_id - 1) * param_per_cov - 16:
                    rank = np.load(prefix + str(cov_id)+'_'+'n3x3' + subfix)
                elif index == (cov_id - 1) * param_per_cov - 8 \
                        or index == (cov_id - 1) * param_per_cov - 4:
                    rank = np.load(prefix + str(cov_id)+'_'+'n5x5' + subfix)
                elif cov_id==1 and index==0:
                    rank = np.load(prefix + str(cov_id) + subfix)
                else:
                    rank = np.load(prefix + str(cov_id) + '_' + 'pool_planes' + subfix)

                f, c, w, h = item.size()
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif cov_id==1 and index > 0 and index <= 3:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

            elif (index>=(cov_id - 1) * param_per_cov - 20 and index< (cov_id - 1) * param_per_cov - 16) \
                    or (index>=(cov_id - 1) * param_per_cov - 12 and index< (cov_id - 1) * param_per_cov - 8):
                continue

            elif index > (cov_id-1)*param_per_cov-24 and index < (cov_id-1)*param_per_cov+4:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == (cov_id-1) * self.param_per_cov + 4:
                break
            if index not in self.mask:
                continue
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet_110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_110_convwise"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id*param_per_cov:
                break

            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.

                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            elif index > (cov_id-1)*param_per_cov and index < cov_id*param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            item.data = item.data * self.mask[index].to(self.device)#prune certain weight


class mask_resnet_50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device

    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_50"):
        params = self.model.parameters()
#         prefix = "rank_conv/old/"+ "resnet_50" +"/rank_conv"
#         subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break

            if index == (cov_id-1) * param_per_cov:
                f, c, w, h = item.size()
#                 rank = np.load(prefix + str(cov_id) + subfix)
#                 pruned_num = int(self.compress_rate[cov_id - 1] * f)
#                 ind = np.argsort(rank)[pruned_num:]  # preserved filter id
                
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)#.cuda(self.device[0])#.to(self.device)
                
                for i in range(int(f)):
                    if i in dict_of_masks[cov_id]:
                        zeros[i, 0, 0, 0] = 1.
                        
                    else:
                        pass
        
#                 for i in range(len(ind)):
#                     zeros[ind[i], 0, 0, 0] = 1.
                    
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]
                    

            elif index > (cov_id-1) * param_per_cov and index < cov_id * param_per_cov:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight