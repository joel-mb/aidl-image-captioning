# [aidl] Image Captioning

## Installation

```sh
git clone https://github.com/joel-mb/aidl-image-captioning
cd aidl-image-captioning
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Training
All the training logic is inside the `train.py` file. The following arguments may be selected:

| Option | Description |
| --- | --- |
| --session-name `<STRING>` | name of the session |
| --data-root `<PATH>` | path to Flickr8k data folder (default: `data/flickr8k`) |
| --num-epochs `<INT>` | number of epochs (default: 10) |
| --batch-size `<INT>` | batch size (default: 32) |
| --learning-rate `<FLOAT>` | learning rate (default: 1e-3) |
| --num-workers `<INT>` | number of workers used in the data loader (default: 4) |
| --encoder-type `<CHOICE>` | encoder type (choices: [`resnet101`, `senet154`], default: `resnet101`) |
| --attention-type `<CHOICE>` | attention type of the decoder (choices: [`none`, `additive`], default: `additive`) |
| --vocab-size `<INT>` | minimum frequency of a word to be added in the vocab (default: 1) |
| --max-seq-length `<INT>` | maximum sequence length (default: 25) |
| --encoder-size `<INT>` | encoder size (default: 64) |
| --hidden-size `<INT>` | hidden size (default: 256) |
| --embedding-size `<INT>` | embedding size (default: 128) |
| --attention-size `<INT>` | attention size (default: 64) |
| --no-save-model | do not save trained model. The model is saved in the `models` folder |
| --log-interval `<INT>` | logging step with tensorboard (per batch) (default: 25) |
| --save-checkpoints | save checkpoints. Checkpoints are saved in the `checkpoints` folder |
| --overfitting | use the overfitting dataset (`data/flickr8k_overfitting`) |

Example of usage:
```sh
python train.py --session-name test1 --num-epochs 50 --attention-type none
```

### Inference
All the inference logic is inside the `inference.py` file. The following arguments may be selected:

* Positional arguments

| Option | Description |
| --- | --- |
| model_name `<STRING>` | name of the model to be used (session_name option of the training phase) |
| image_path `<PATH>` | path to the image to predict the caption |

* Optional arguments

| Option | Description |
| --- | --- |
| --data-root `<PATH>` | path to Flickr8k data folder (default: `data/flickr8k`) |
| --max-seq-length `<INT>` | maximum sequence length (default: 25) |

Example of usage:
```
python inference.py test1 data/flickr8k_overfitting/Flickr8k_Dataset/12830823_87d2654e31.jpg
```