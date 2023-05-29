go to the root folder of project, then
> chmod +x run.sh
Then run
> ./run.sh

This should download the pretrained models, our ranks, and create required repositories for replicating the results. In some scenarios we have seen that
gdown is not able to download the files correctly. 

if models were not downloaded using gdown, cd into the pretrained_models directory, then create 3 following folders:
<pre>
VGG_16
ResNet56
ResNet50
</pre>
then cd into VGG_16, download the pretrained model from: 

VGG_16 : https://drive.google.com/file/d/1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE/view

then get back and cd into ResNet56 folder and download the pretrained model from:
ResNet_56 : https://drive.google.com/file/d/1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T/view

then get back and cd into ResNet50 folder and download the pretrained model from:
ResNet_50 : https://drive.google.com/file/d/1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB/view

Note that these models are the same model that HRANK has used. We then rename the models as follows:
VGG_16 : vgg_16_bn.pt 
ResNet_56 : resnet_56.pt
ResNet_50 : resnet50-19c8e357.pth

Your pretrained_models folder should now look like:
<pre>
pretrained_models
        ├── ResNet50
        │     └── resnet50-19c8e357.pth
        ├── ResNet56
        │     └── resnet_56.pt
        ├── VGG_16
              └── vgg_16_bn.pt
</pre>

In order to generate ranks from scratch please use the following codes:
> rank_generation_cifar.py
> rank_generation_imagenet.py

Note that both VGG16 and ResNet56 ranks can be generated using rank_generation_cifar.py . Please change the --arch and --resume
parameters based on your inetrest.

In order to generate imagenet ranks, please download the dataset and set the "data_root" parameter.
If you don't want to generate ranks from scratch, the bash script will download the ranks
too, if gdown didn't work, download the zip file using the following link and extract it in root folder.

rank_conv: https://drive.google.com/file/d/1pwFcyVCGtFvcZh0XWlynFzdzuuLV_i5j/view


When ranks are generated, for run:
<pre>
kval_cifar_vgg.py
kval_cifar_resnet.py
kval_imagenet.py
</pre>

to perform clustering and find best K values based on the architecture and dataset you want to test.

At the end, it will save a pickle file which is going to be the input for generate_filter_lists.py

these files are called:

<pre>
VGG_10D_Best_K_in_layers.pickle
ResNet_10D_Best_K_in_layers_.pickle
NEW_COMPLETE_ResNet_1000D_Best_K_in_layers.pickle

</pre>

output of generate_filter_lists.py is a filename called POST_CLEAN_JUST_R1_MEDOIDandBELOWAVG.pickle which is input of fine-tuning step which is done in fine_tuning.py


In generate_filter_lists.py, change the parameters in this file based on your interest:

<pre>
num_class = 10 #1000
pretrained_model_path = './pretrained_models/VGG_16/vgg_16_bn.pt'
complete_pickle_filepath = './VGG_10D_Best_K_in_layers.pickle'
</pre>




finally run fine_tuning.py to perform fine-tuning as the final step of our pruning method. Note that size_of_layer parameter must be changed based on the chosen architecture.