# Run this script from the parent Canonical directory, as
# PYTHONPATH=. python scripts/splicepoint_model_demo.py

import torch
from modular_splicing.models.modules.lssi_in_model import load_individual_lssi_model

which = "acceptor"

# load the model
m = load_individual_lssi_model(f"splicepoint-models/{which}.m", trainable=False)
# put the model into evaluation mode
m = m.eval()
# sequence of random bases, 10 batches each with 1000 elements
x = torch.randint(4, size=(10, 1000))
# convert to one-hot encoding
x = torch.eye(4, dtype=torch.float)[x]
# you can comment out the following two lines to run on cpu
x = x.cuda()
m = m.cuda()
with torch.no_grad():
    yp = m(x).softmax(-1)
yp = yp.cpu().numpy()
# yp : (10, 1000, 3). Channels are P(null), P(acceptor), P(donor). Pull the channel we care about
yp = yp[:, :, {"acceptor": 1, "donor": 2}[which]]
# this produces a prediction at all sites, but it is not reliable on sites towards the ends of the sequence
# which are within the footprint
