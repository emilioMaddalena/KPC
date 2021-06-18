# KPC 

This code is the supplementary material for the paper

```
@inproceedings{maddalena2021kpc,
  title={KPC: Learning-based model predictive control with deterministic guarantees},
  author={Maddalena, Emilio T and Scharnhorst, Paul and Jiang, Yuning and Jones, Colin N},
  booktitle={Learning for Dynamics and Control},
  pages={1015--1026},
  year={2021},
  organization={PMLR}
}
```

## Description :books:

We make use of deterministic, finite-sample, error bounds for kernel models to design robust MPC controllers. Non-parametric kernel machines are used to learn the dynamics `f(x,u)` of discrete-time dynamical systems. Thanks to the bounds, we can build hyper-rectangles aroung the nominal predictions that are guaranteed to contain the ground-truth states. 

## Dependencies  :building_construction:

- YALMIP (https://yalmip.github.io/)
- casADi (https://web.casadi.org/)
- Multi-parametric Toolbox 3.0 (https://www.mpt3.org/)

![alt text](https://github.com/emilioMaddalena/KPC/blob/dev/fig/pred.png)
