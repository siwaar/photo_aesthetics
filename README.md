python -m venv wenv
./wenv/Scripts/activate

pip install -r .\requirements.txt 

or 

pip install pyyaml 
python .\requirements.py

pip install matplotlib
pip install pandas
pip install torch
pip install torchvision

python -W ignore test.py --model model_files/epoch-82.pth --test_images images --predictions predictions

pip install numpy==1.23.4

python main.py --img_path images/ --train --train_csv_file dataset/train_labels.csv --val_csv_file dataset/val_labels.csv --conv_base_lr 5e-4 --dense_lr 5e-3 --decay --ckpt_path model_files/epoch-82.pth --epochs 1 --early_stoppping_patience 10
