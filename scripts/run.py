from pyclassify import kNN
from pyclassify.utils import read_config, read_file
import argparse
import random

#for reproducibility
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str , help="Config file to read from")

parser = parser.parse_args()

kwargs = read_config(parser.config)

X,y = read_file(kwargs["dataset"])
kwargs.pop("dataset")

# classifier = kNN(kwargs["k"],kwargs["backend"])
classifier = kNN(**kwargs)

ntrain = int(0.2*len(y))
ntest = len(y)-ntrain

idx = list(range(len(y)))
random.shuffle(idx)

idx_train = idx[:ntrain]
idx_test = idx[ntrain:]

xtrain = [X[i] for i in idx_train]
ytrain = [y[i] for i in idx_train]
xtest = [X[i] for i in idx_test]
ytest = [y[i] for i in idx_test]

yhat = classifier((xtrain,ytrain),xtest)

acc = 0
for yy, yyh in zip(ytest,yhat):
    acc += abs(yy == yyh)
acc /= ntest

print(f"Accuracy = {acc*100:.2f} %")
