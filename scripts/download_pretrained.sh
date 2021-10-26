set -ex

mkdir -p checkpoints
cd checkpoints
gdown https://drive.google.com/uc?id=1-dVl8U58EqgE_zp-dM3qbkVl9m6jyLgp
tar -xzf pix2pix.tar.gz
rm -rf pix2pix.tar.gz