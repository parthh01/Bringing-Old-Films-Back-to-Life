mkdir pretrained_models
mkdir OUTPUT
gcloud alpha storage cp gs://old-films/raft-sintel.pth ./pretrained_models/
gcloud alpha storage cp gs://old-films/RNN_Swin_4.zip ./OUTPUT/
cd OUTPUT 
unzip RNN_Swin_4.zip
cd ../
pip install -r requirements.txt
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
gcloud alpha storage cp gs://old-films/yb_full.zip
unzip yb_full.zip 
mv yb_full test_data/data_1/
rm -rf test_data/data_1/002/ 
mv test_data/data_1/yb_full test_data/data_1/002
