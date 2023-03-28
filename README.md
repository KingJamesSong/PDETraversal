# PDETraversal

ICML23 paper "[Latent Traversals in Generative Models as Potential Flows](https://arxiv.org/abs/)"
[Yue Song](https://kingjamessong.github.io/)<sup>1,2</sup>, [Andy Keller](https://scholar.google.com/citations?hl=en&user=Tb86kC0AAAAJ)<sup>1</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>2</sup>, [Max Welling](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en)<sup>1</sup>
<sup>1</sup>University of Amsterdam, the Netherlands <br>
<sup>2</sup>University of Trento, Italy <br> 

## Pre-trained GAN

Please first run [checkpoint2model.py]() for downloading pre-trained GANs, and run [anime.sh]() and [anime_eval.sh]() for the training potential functions and evaluation.

## Pre-trained VAE

Please first run [train_vae.py]() to train VAEs and then run [mnist.sh]() for training potentials.

## Training VAE from scratch

Please run [mnist_scratch.sh]() for training VAEs and potentials simultaneously.

## Citation

If you think the code is helpful to your research, please consider citing our paper:

```
@inproceedings{song2023latent,
  title={Latent Traversals in Generative Models as Potential Flows},
  author={Song, Yue and Keller, Andy and Sebe, Nicu and Welling, Max},
  booktitle={ICML},
  year={2023},
  organization={PMLR}
}
```

The code is built based on [WarpedGANSpace](https://github.com/chi0tzp/WarpedGANSpace) and we sincerely thank their contributions. If you have any questions or suggestions, please feel free to contact me via `yue.song@unitn.it`.
