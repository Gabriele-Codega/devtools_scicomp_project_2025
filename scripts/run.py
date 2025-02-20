from pyclassify import kNN
from pyclassify.utils import read_config, read_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str , help="Config file to read from")

parser = parser.parse_args()

kwargs = read_config(parser.config)

X,y = read_file(kwargs["dataset"])

classifier = kNN(kwargs["k"])

ntrain = int(0.2*len(y))
ntest = len(y)-ntrain

xtrain = X[:ntrain]
ytrain = y[:ntrain]
xtest = X[ntrain:]
ytest = y[ntrain:]

yhat = classifier((xtrain,ytrain),xtest)

acc = 0
for yy, yyh in zip(ytest,yhat):
    acc += abs(yy == yyh)
acc /= ntest

print(f"Accuracy = {acc*100:.2f} %")
