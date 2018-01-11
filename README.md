# Attribute-transfer
Authors: Juncen Li, Robin Jia, He He, Percy Liang

NOTES:

* The instructions below are still a work in progress.
	* If you encounter any problems, please open a GitHub issue, or submit a pull request if you know the fix!
* This code requires data directories that we have not uploaded yet.
* This is research code meant to serve as a reference implementation. We do not recommend heavily extending or modifying this codebase for other purposes.
If you have questions, please email Juncen Li at mlijuncenm at gmail.com.
This work is based on Theano
Python requirement: theano numpy scipy nltk

## Data Format
Please name the corpora of two styles by "x.0" and "x.1" respectively, and use "x" to refer to them in options. Each file should consist of one sentence per line with tokens separated by a space.
The <code>data/yelp/</code> directory contains Yelp review dataset ("x.0" is negative data and "x.1" is positive data).
The <code>data/amazon/</code> directory contains Amazon review dataset ("x.0" is negative data and "x.1" is positive data).
The <code>data/imagecaption/</code> directory contains dataset of image captions ("x.0" is humorous data and "x.1" is romantic data).

## Quick Start
To train a model, run the following command:
sh run.sh train model_name data_name

To test a model, run the following command:
sh run.sh test model_name data_name

Where model_name can be "DeleteOnly" or "DeleteAndRetrieve", data_name can be "yelp", "amazon" or "imagecaption". For example you want to reproduce the training process of DeleteOnly on Yelp review dataset, you can run the following command:

sh run.sh train DeleteOnly yelp
 




