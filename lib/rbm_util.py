from RBM import RBM
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import joblib
import matplotlib.pyplot as plt


def train_rbm(fp_path="data/fp.bin",
              out_path="data/rbm_J.bin",
              batch_size=16,
              all_units=2048,
              use_gpu=True,
              epochs=100,
              k=1,
              check_reconstruction=3,
              plot=True,
              qubo_mode=False
              ):
    """
    utility function for the RBM class

    Parameters
    ----------
    fp_path : string
        path to fingerprint file (dumped by joblib. contains list of fingerprint to be learned)
    out_path : string
        saving path. learned parameters will be as a qubo matrix (dumped by joblib)
    batch_size: int
        batch size
    all_units: int
        total units (= visible dim + hidden dim)
    use_gpu: bool
        use gpu or not
    epochs: int
        epochs
    k: int
        contrastive-k method
    check_reconstruction: int
        after learning, original reconstruction images will be shown with random (check_reconstruction) cases
    plot: bool
        plot images after learning
    qubo_mode: bool
        if true, np array of qubo will be returned. otherwise, reconstructed images will be returned

    Returns
    -------
    rbm_qubo : np array
        qubo array
    rc_list : list
        list of reconstructed data  
    """

    fp_list = np.array(joblib.load(fp_path))

    # init dataloader
    tensor_x = torch.tensor(fp_list, dtype=torch.float32)
    _dataset = torch.utils.data.TensorDataset(tensor_x)
    train_loader = torch.utils.data.DataLoader(_dataset,
                                               batch_size=batch_size, shuffle=True, drop_last=True)

    # init RBM
    visible_units = len(fp_list[0])
    hidden_units = all_units-visible_units

    rbm_mnist = RBM(visible_units, hidden_units,
                    use_gpu=use_gpu, k=k).cuda()

    # train
    rbm_mnist.train(train_loader, epochs, batch_size)
    rbm_qubo = rbm_mnist.get_J()
    # save
    joblib.dump(rbm_qubo, out_path)

    rc_list = None

    if check_reconstruction:
        rc_list = []
        for i in range(check_reconstruction):
            idx = np.random.randint(0, len(tensor_x))
            img = tensor_x[idx]
            reconstructed_img = img.view(-1).type(torch.FloatTensor)

            #_,reconstructed_img = rbm_mnist.reconstruct(reconstructed_img,1)
            reconstructed_img, _ = rbm_mnist.reconstruct(reconstructed_img, 1)
            reconstructed_img = reconstructed_img.view(
                (16, -1)).detach().cpu().numpy()

            if plot:
                print("Original and reconstructed images")
                plt.imshow(img.view(16, -1), cmap='gray')
                plt.show()
                plt.imshow(reconstructed_img, cmap='gray')
                plt.show()

            rc_list.append(
                [img.detach().cpu().numpy().reshape(-1), reconstructed_img.reshape(-1)])

    del rbm_mnist
    torch.cuda.empty_cache()

    """
    print("cleaning memory...")
    try:
        del rbm_mnist
        torch.cuda.empty_cache()
        print("done!")
    except:
        print("error!")    
    """
    if qubo_mode:
        return rbm_qubo
    else:
        return rc_list
