mkdir data

cd data
wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

python3 -m venv env; source env/bin/activate
pip install -r requirements.txt