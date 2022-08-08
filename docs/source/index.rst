.. image:: ./_static/logo.png
   :scale: 3 %
   :alt: DeepH-pack
   :align: center

DeepH-pack's documentation
======================================

DeepH-pack is a package for the application of deep neural 
networks to the prediction of density functional theory (DFT) 
Hamiltonian matrices based on local coordinates and basis 
transformation [#deeph]_. DeepH-pack supports DFT results made
by `ABACUS <https://abacus.ustc.edu.cn/>`_,
`OpenMX <http://www.openmx-square.org/>`_ or
`FHI-aims <https://fhi-aims.org/>`_
and will support `SIESTA <https://departments.icmab.es/leem/siesta/>`_
and `HONPAS <http://honpas.ustc.edu.cn/>`_ soon.

.. toctree::
   :glob:
   :caption: Getting Started
   :maxdepth: 2

   installation/installation
   dataset/dataset
   preprocess/preprocess
   train/train
   inference/inference

.. toctree::
   :glob:
   :caption: Demo
   :maxdepth: 1

   demo/demo1
   demo/demo2

.. toctree::
   :glob:
   :caption: Input Keywords
   :maxdepth: 2
   
   keyword/keyword

References
^^^^^^^^^^^^^^^^^
.. [#deeph] H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu.
   `Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation <https://doi.org/10.1038/s43588-022-00265-6>`_.
   *Nat. Comput. Sci.* **2** (1), 367–377 2022.

