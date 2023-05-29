rm -rf ./result
mkdir -p ./result
rm -rf ./rank_conv
mkdir -p ./rank_conv
rm -rf ./pretrained_models
mkdir -p ./pretrained_models
pip3 install -r requirements.txt
cd ./pretrained_models
mkdir ./VGG_16
cd ./VGG_16
gdown 1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE
mv ./* ./vgg_16_bn.pt
cd ..
mkdir ./ResNet56
cd ./ResNet56
gdown 1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T
mv ./* ./resnet_56.pt
cd ..
mkdir ./ResNet50
cd ./ResNet50
gdown 1OYpVB84BMU0y-KU7PdEPhbHwODmFvPbB
cd ../..
gdown 1pwFcyVCGtFvcZh0XWlynFzdzuuLV_i5j
unzip rank_conv.zip


