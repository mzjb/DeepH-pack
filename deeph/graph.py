import collections
import itertools
import os
import json
import warnings
import math

import torch
from torch_geometric.data import Data, Batch
import numpy as np
import h5py

from .model import get_spherical_from_cartesian, SphericalHarmonics
from .from_pymatgen import find_neighbors, _one_to_three, _compute_cube_index, _three_to_one


"""
The function _spherical_harmonics below is come from "https://github.com/e3nn/e3nn", which has the MIT License below

---------------------------------------------------------------------------
MIT License

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy), 
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
and Kostiantyn Lapchevskyi. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([
            sh_0_0,
        ], dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
        ], dim=-1)

    sh_4_0 = 0.935414346693485*sh_3_0*z + 0.935414346693485*sh_3_6*x
    sh_4_1 = 0.661437827766148*sh_3_0*y + 0.810092587300982*sh_3_1*z + 0.810092587300983*sh_3_5*x
    sh_4_2 = -0.176776695296637*sh_3_0*z + 0.866025403784439*sh_3_1*y + 0.684653196881458*sh_3_2*z + 0.684653196881457*sh_3_4*x + 0.176776695296637*sh_3_6*x
    sh_4_3 = -0.306186217847897*sh_3_1*z + 0.968245836551855*sh_3_2*y + 0.790569415042095*sh_3_3*x + 0.306186217847897*sh_3_5*x
    sh_4_4 = -0.612372435695795*sh_3_2*x + sh_3_3*y - 0.612372435695795*sh_3_4*z
    sh_4_5 = -0.306186217847897*sh_3_1*x + 0.790569415042096*sh_3_3*z + 0.968245836551854*sh_3_4*y - 0.306186217847897*sh_3_5*z
    sh_4_6 = -0.176776695296637*sh_3_0*x - 0.684653196881457*sh_3_2*x + 0.684653196881457*sh_3_4*z + 0.866025403784439*sh_3_5*y - 0.176776695296637*sh_3_6*z
    sh_4_7 = -0.810092587300982*sh_3_1*x + 0.810092587300982*sh_3_5*z + 0.661437827766148*sh_3_6*y
    sh_4_8 = -0.935414346693485*sh_3_0*x + 0.935414346693486*sh_3_6*z
    if lmax == 4:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
        ], dim=-1)

    sh_5_0 = 0.948683298050513*sh_4_0*z + 0.948683298050513*sh_4_8*x
    sh_5_1 = 0.6*sh_4_0*y + 0.848528137423857*sh_4_1*z + 0.848528137423858*sh_4_7*x
    sh_5_2 = -0.14142135623731*sh_4_0*z + 0.8*sh_4_1*y + 0.748331477354788*sh_4_2*z + 0.748331477354788*sh_4_6*x + 0.14142135623731*sh_4_8*x
    sh_5_3 = -0.244948974278318*sh_4_1*z + 0.916515138991168*sh_4_2*y + 0.648074069840786*sh_4_3*z + 0.648074069840787*sh_4_5*x + 0.244948974278318*sh_4_7*x
    sh_5_4 = -0.346410161513776*sh_4_2*z + 0.979795897113272*sh_4_3*y + 0.774596669241484*sh_4_4*x + 0.346410161513776*sh_4_6*x
    sh_5_5 = -0.632455532033676*sh_4_3*x + sh_4_4*y - 0.632455532033676*sh_4_5*z
    sh_5_6 = -0.346410161513776*sh_4_2*x + 0.774596669241483*sh_4_4*z + 0.979795897113273*sh_4_5*y - 0.346410161513776*sh_4_6*z
    sh_5_7 = -0.244948974278318*sh_4_1*x - 0.648074069840787*sh_4_3*x + 0.648074069840786*sh_4_5*z + 0.916515138991169*sh_4_6*y - 0.244948974278318*sh_4_7*z
    sh_5_8 = -0.141421356237309*sh_4_0*x - 0.748331477354788*sh_4_2*x + 0.748331477354788*sh_4_6*z + 0.8*sh_4_7*y - 0.141421356237309*sh_4_8*z
    sh_5_9 = -0.848528137423857*sh_4_1*x + 0.848528137423857*sh_4_7*z + 0.6*sh_4_8*y
    sh_5_10 = -0.948683298050513*sh_4_0*x + 0.948683298050513*sh_4_8*z
    if lmax == 5:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
        ], dim=-1)

    sh_6_0 = 0.957427107756337*sh_5_0*z + 0.957427107756338*sh_5_10*x
    sh_6_1 = 0.552770798392565*sh_5_0*y + 0.874007373475125*sh_5_1*z + 0.874007373475125*sh_5_9*x
    sh_6_2 = -0.117851130197757*sh_5_0*z + 0.745355992499929*sh_5_1*y + 0.117851130197758*sh_5_10*x + 0.790569415042094*sh_5_2*z + 0.790569415042093*sh_5_8*x
    sh_6_3 = -0.204124145231931*sh_5_1*z + 0.866025403784437*sh_5_2*y + 0.707106781186546*sh_5_3*z + 0.707106781186547*sh_5_7*x + 0.204124145231931*sh_5_9*x
    sh_6_4 = -0.288675134594813*sh_5_2*z + 0.942809041582062*sh_5_3*y + 0.623609564462323*sh_5_4*z + 0.623609564462322*sh_5_6*x + 0.288675134594812*sh_5_8*x
    sh_6_5 = -0.372677996249965*sh_5_3*z + 0.986013297183268*sh_5_4*y + 0.763762615825972*sh_5_5*x + 0.372677996249964*sh_5_7*x
    sh_6_6 = -0.645497224367901*sh_5_4*x + sh_5_5*y - 0.645497224367902*sh_5_6*z
    sh_6_7 = -0.372677996249964*sh_5_3*x + 0.763762615825972*sh_5_5*z + 0.986013297183269*sh_5_6*y - 0.372677996249965*sh_5_7*z
    sh_6_8 = -0.288675134594813*sh_5_2*x - 0.623609564462323*sh_5_4*x + 0.623609564462323*sh_5_6*z + 0.942809041582062*sh_5_7*y - 0.288675134594812*sh_5_8*z
    sh_6_9 = -0.20412414523193*sh_5_1*x - 0.707106781186546*sh_5_3*x + 0.707106781186547*sh_5_7*z + 0.866025403784438*sh_5_8*y - 0.204124145231931*sh_5_9*z
    sh_6_10 = -0.117851130197757*sh_5_0*x - 0.117851130197757*sh_5_10*z - 0.790569415042094*sh_5_2*x + 0.790569415042093*sh_5_8*z + 0.745355992499929*sh_5_9*y
    sh_6_11 = -0.874007373475124*sh_5_1*x + 0.552770798392566*sh_5_10*y + 0.874007373475125*sh_5_9*z
    sh_6_12 = -0.957427107756337*sh_5_0*x + 0.957427107756336*sh_5_10*z
    if lmax == 6:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
        ], dim=-1)

    sh_7_0 = 0.963624111659433*sh_6_0*z + 0.963624111659432*sh_6_12*x
    sh_7_1 = 0.515078753637713*sh_6_0*y + 0.892142571199771*sh_6_1*z + 0.892142571199771*sh_6_11*x
    sh_7_2 = -0.101015254455221*sh_6_0*z + 0.699854212223765*sh_6_1*y + 0.82065180664829*sh_6_10*x + 0.101015254455222*sh_6_12*x + 0.82065180664829*sh_6_2*z
    sh_7_3 = -0.174963553055942*sh_6_1*z + 0.174963553055941*sh_6_11*x + 0.82065180664829*sh_6_2*y + 0.749149177264394*sh_6_3*z + 0.749149177264394*sh_6_9*x
    sh_7_4 = 0.247435829652697*sh_6_10*x - 0.247435829652697*sh_6_2*z + 0.903507902905251*sh_6_3*y + 0.677630927178938*sh_6_4*z + 0.677630927178938*sh_6_8*x
    sh_7_5 = -0.31943828249997*sh_6_3*z + 0.95831484749991*sh_6_4*y + 0.606091526731326*sh_6_5*z + 0.606091526731326*sh_6_7*x + 0.31943828249997*sh_6_9*x
    sh_7_6 = -0.391230398217976*sh_6_4*z + 0.989743318610787*sh_6_5*y + 0.755928946018454*sh_6_6*x + 0.391230398217975*sh_6_8*x
    sh_7_7 = -0.654653670707977*sh_6_5*x + sh_6_6*y - 0.654653670707978*sh_6_7*z
    sh_7_8 = -0.391230398217976*sh_6_4*x + 0.755928946018455*sh_6_6*z + 0.989743318610787*sh_6_7*y - 0.391230398217975*sh_6_8*z
    sh_7_9 = -0.31943828249997*sh_6_3*x - 0.606091526731327*sh_6_5*x + 0.606091526731326*sh_6_7*z + 0.95831484749991*sh_6_8*y - 0.31943828249997*sh_6_9*z
    sh_7_10 = -0.247435829652697*sh_6_10*z - 0.247435829652697*sh_6_2*x - 0.677630927178938*sh_6_4*x + 0.677630927178938*sh_6_8*z + 0.903507902905251*sh_6_9*y
    sh_7_11 = -0.174963553055942*sh_6_1*x + 0.820651806648289*sh_6_10*y - 0.174963553055941*sh_6_11*z - 0.749149177264394*sh_6_3*x + 0.749149177264394*sh_6_9*z
    sh_7_12 = -0.101015254455221*sh_6_0*x + 0.82065180664829*sh_6_10*z + 0.699854212223766*sh_6_11*y - 0.101015254455221*sh_6_12*z - 0.82065180664829*sh_6_2*x
    sh_7_13 = -0.892142571199772*sh_6_1*x + 0.892142571199772*sh_6_11*z + 0.515078753637713*sh_6_12*y
    sh_7_14 = -0.963624111659431*sh_6_0*x + 0.963624111659433*sh_6_12*z
    if lmax == 7:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
        ], dim=-1)

    sh_8_0 = 0.968245836551854*sh_7_0*z + 0.968245836551853*sh_7_14*x
    sh_8_1 = 0.484122918275928*sh_7_0*y + 0.90571104663684*sh_7_1*z + 0.90571104663684*sh_7_13*x
    sh_8_2 = -0.0883883476483189*sh_7_0*z + 0.661437827766148*sh_7_1*y + 0.843171097702002*sh_7_12*x + 0.088388347648318*sh_7_14*x + 0.843171097702003*sh_7_2*z
    sh_8_3 = -0.153093108923948*sh_7_1*z + 0.7806247497998*sh_7_11*x + 0.153093108923949*sh_7_13*x + 0.7806247497998*sh_7_2*y + 0.780624749799799*sh_7_3*z
    sh_8_4 = 0.718070330817253*sh_7_10*x + 0.21650635094611*sh_7_12*x - 0.21650635094611*sh_7_2*z + 0.866025403784439*sh_7_3*y + 0.718070330817254*sh_7_4*z
    sh_8_5 = 0.279508497187474*sh_7_11*x - 0.279508497187474*sh_7_3*z + 0.927024810886958*sh_7_4*y + 0.655505530106345*sh_7_5*z + 0.655505530106344*sh_7_9*x
    sh_8_6 = 0.342326598440729*sh_7_10*x - 0.342326598440729*sh_7_4*z + 0.968245836551854*sh_7_5*y + 0.592927061281572*sh_7_6*z + 0.592927061281571*sh_7_8*x
    sh_8_7 = -0.405046293650492*sh_7_5*z + 0.992156741649221*sh_7_6*y + 0.75*sh_7_7*x + 0.405046293650492*sh_7_9*x
    sh_8_8 = -0.661437827766148*sh_7_6*x + sh_7_7*y - 0.661437827766148*sh_7_8*z
    sh_8_9 = -0.405046293650492*sh_7_5*x + 0.75*sh_7_7*z + 0.992156741649221*sh_7_8*y - 0.405046293650491*sh_7_9*z
    sh_8_10 = -0.342326598440728*sh_7_10*z - 0.342326598440729*sh_7_4*x - 0.592927061281571*sh_7_6*x + 0.592927061281571*sh_7_8*z + 0.968245836551855*sh_7_9*y
    sh_8_11 = 0.927024810886958*sh_7_10*y - 0.279508497187474*sh_7_11*z - 0.279508497187474*sh_7_3*x - 0.655505530106345*sh_7_5*x + 0.655505530106345*sh_7_9*z
    sh_8_12 = 0.718070330817253*sh_7_10*z + 0.866025403784439*sh_7_11*y - 0.216506350946109*sh_7_12*z - 0.216506350946109*sh_7_2*x - 0.718070330817254*sh_7_4*x
    sh_8_13 = -0.153093108923948*sh_7_1*x + 0.7806247497998*sh_7_11*z + 0.7806247497998*sh_7_12*y - 0.153093108923948*sh_7_13*z - 0.780624749799799*sh_7_3*x
    sh_8_14 = -0.0883883476483179*sh_7_0*x + 0.843171097702002*sh_7_12*z + 0.661437827766147*sh_7_13*y - 0.088388347648319*sh_7_14*z - 0.843171097702002*sh_7_2*x
    sh_8_15 = -0.90571104663684*sh_7_1*x + 0.90571104663684*sh_7_13*z + 0.484122918275927*sh_7_14*y
    sh_8_16 = -0.968245836551853*sh_7_0*x + 0.968245836551855*sh_7_14*z
    if lmax == 8:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
        ], dim=-1)

    sh_9_0 = 0.97182531580755*sh_8_0*z + 0.971825315807551*sh_8_16*x
    sh_9_1 = 0.458122847290851*sh_8_0*y + 0.916245694581702*sh_8_1*z + 0.916245694581702*sh_8_15*x
    sh_9_2 = -0.078567420131839*sh_8_0*z + 0.62853936105471*sh_8_1*y + 0.86066296582387*sh_8_14*x + 0.0785674201318385*sh_8_16*x + 0.860662965823871*sh_8_2*z
    sh_9_3 = -0.136082763487955*sh_8_1*z + 0.805076485899413*sh_8_13*x + 0.136082763487954*sh_8_15*x + 0.74535599249993*sh_8_2*y + 0.805076485899413*sh_8_3*z
    sh_9_4 = 0.749485420179558*sh_8_12*x + 0.192450089729875*sh_8_14*x - 0.192450089729876*sh_8_2*z + 0.831479419283099*sh_8_3*y + 0.749485420179558*sh_8_4*z
    sh_9_5 = 0.693888666488711*sh_8_11*x + 0.248451997499977*sh_8_13*x - 0.248451997499976*sh_8_3*z + 0.895806416477617*sh_8_4*y + 0.69388866648871*sh_8_5*z
    sh_9_6 = 0.638284738504225*sh_8_10*x + 0.304290309725092*sh_8_12*x - 0.304290309725092*sh_8_4*z + 0.942809041582063*sh_8_5*y + 0.638284738504225*sh_8_6*z
    sh_9_7 = 0.360041149911548*sh_8_11*x - 0.360041149911548*sh_8_5*z + 0.974996043043569*sh_8_6*y + 0.582671582316751*sh_8_7*z + 0.582671582316751*sh_8_9*x
    sh_9_8 = 0.415739709641549*sh_8_10*x - 0.415739709641549*sh_8_6*z + 0.993807989999906*sh_8_7*y + 0.74535599249993*sh_8_8*x
    sh_9_9 = -0.66666666666666666667*sh_8_7*x + sh_8_8*y - 0.66666666666666666667*sh_8_9*z
    sh_9_10 = -0.415739709641549*sh_8_10*z - 0.415739709641549*sh_8_6*x + 0.74535599249993*sh_8_8*z + 0.993807989999906*sh_8_9*y
    sh_9_11 = 0.974996043043568*sh_8_10*y - 0.360041149911547*sh_8_11*z - 0.360041149911548*sh_8_5*x - 0.582671582316751*sh_8_7*x + 0.582671582316751*sh_8_9*z
    sh_9_12 = 0.638284738504225*sh_8_10*z + 0.942809041582063*sh_8_11*y - 0.304290309725092*sh_8_12*z - 0.304290309725092*sh_8_4*x - 0.638284738504225*sh_8_6*x
    sh_9_13 = 0.693888666488711*sh_8_11*z + 0.895806416477617*sh_8_12*y - 0.248451997499977*sh_8_13*z - 0.248451997499977*sh_8_3*x - 0.693888666488711*sh_8_5*x
    sh_9_14 = 0.749485420179558*sh_8_12*z + 0.831479419283098*sh_8_13*y - 0.192450089729875*sh_8_14*z - 0.192450089729875*sh_8_2*x - 0.749485420179558*sh_8_4*x
    sh_9_15 = -0.136082763487954*sh_8_1*x + 0.805076485899413*sh_8_13*z + 0.745355992499929*sh_8_14*y - 0.136082763487955*sh_8_15*z - 0.805076485899413*sh_8_3*x
    sh_9_16 = -0.0785674201318389*sh_8_0*x + 0.86066296582387*sh_8_14*z + 0.628539361054709*sh_8_15*y - 0.0785674201318387*sh_8_16*z - 0.860662965823871*sh_8_2*x
    sh_9_17 = -0.9162456945817*sh_8_1*x + 0.916245694581702*sh_8_15*z + 0.458122847290851*sh_8_16*y
    sh_9_18 = -0.97182531580755*sh_8_0*x + 0.97182531580755*sh_8_16*z
    if lmax == 9:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
        ], dim=-1)

    sh_10_0 = 0.974679434480897*sh_9_0*z + 0.974679434480897*sh_9_18*x
    sh_10_1 = 0.435889894354067*sh_9_0*y + 0.924662100445347*sh_9_1*z + 0.924662100445347*sh_9_17*x
    sh_10_2 = -0.0707106781186546*sh_9_0*z + 0.6*sh_9_1*y + 0.874642784226796*sh_9_16*x + 0.070710678118655*sh_9_18*x + 0.874642784226795*sh_9_2*z
    sh_10_3 = -0.122474487139159*sh_9_1*z + 0.824621125123533*sh_9_15*x + 0.122474487139159*sh_9_17*x + 0.714142842854285*sh_9_2*y + 0.824621125123533*sh_9_3*z
    sh_10_4 = 0.774596669241484*sh_9_14*x + 0.173205080756887*sh_9_16*x - 0.173205080756888*sh_9_2*z + 0.8*sh_9_3*y + 0.774596669241483*sh_9_4*z
    sh_10_5 = 0.724568837309472*sh_9_13*x + 0.223606797749979*sh_9_15*x - 0.223606797749979*sh_9_3*z + 0.866025403784438*sh_9_4*y + 0.724568837309472*sh_9_5*z
    sh_10_6 = 0.674536878161602*sh_9_12*x + 0.273861278752583*sh_9_14*x - 0.273861278752583*sh_9_4*z + 0.916515138991168*sh_9_5*y + 0.674536878161602*sh_9_6*z
    sh_10_7 = 0.62449979983984*sh_9_11*x + 0.324037034920393*sh_9_13*x - 0.324037034920393*sh_9_5*z + 0.953939201416946*sh_9_6*y + 0.62449979983984*sh_9_7*z
    sh_10_8 = 0.574456264653803*sh_9_10*x + 0.374165738677394*sh_9_12*x - 0.374165738677394*sh_9_6*z + 0.979795897113272*sh_9_7*y + 0.574456264653803*sh_9_8*z
    sh_10_9 = 0.424264068711928*sh_9_11*x - 0.424264068711929*sh_9_7*z + 0.99498743710662*sh_9_8*y + 0.741619848709567*sh_9_9*x
    sh_10_10 = -0.670820393249937*sh_9_10*z - 0.670820393249937*sh_9_8*x + sh_9_9*y
    sh_10_11 = 0.99498743710662*sh_9_10*y - 0.424264068711929*sh_9_11*z - 0.424264068711929*sh_9_7*x + 0.741619848709567*sh_9_9*z
    sh_10_12 = 0.574456264653803*sh_9_10*z + 0.979795897113272*sh_9_11*y - 0.374165738677395*sh_9_12*z - 0.374165738677394*sh_9_6*x - 0.574456264653803*sh_9_8*x
    sh_10_13 = 0.62449979983984*sh_9_11*z + 0.953939201416946*sh_9_12*y - 0.324037034920393*sh_9_13*z - 0.324037034920393*sh_9_5*x - 0.62449979983984*sh_9_7*x
    sh_10_14 = 0.674536878161602*sh_9_12*z + 0.916515138991168*sh_9_13*y - 0.273861278752583*sh_9_14*z - 0.273861278752583*sh_9_4*x - 0.674536878161603*sh_9_6*x
    sh_10_15 = 0.724568837309472*sh_9_13*z + 0.866025403784439*sh_9_14*y - 0.223606797749979*sh_9_15*z - 0.223606797749979*sh_9_3*x - 0.724568837309472*sh_9_5*x
    sh_10_16 = 0.774596669241484*sh_9_14*z + 0.8*sh_9_15*y - 0.173205080756888*sh_9_16*z - 0.173205080756887*sh_9_2*x - 0.774596669241484*sh_9_4*x
    sh_10_17 = -0.12247448713916*sh_9_1*x + 0.824621125123532*sh_9_15*z + 0.714142842854285*sh_9_16*y - 0.122474487139158*sh_9_17*z - 0.824621125123533*sh_9_3*x
    sh_10_18 = -0.0707106781186548*sh_9_0*x + 0.874642784226796*sh_9_16*z + 0.6*sh_9_17*y - 0.0707106781186546*sh_9_18*z - 0.874642784226796*sh_9_2*x
    sh_10_19 = -0.924662100445348*sh_9_1*x + 0.924662100445347*sh_9_17*z + 0.435889894354068*sh_9_18*y
    sh_10_20 = -0.974679434480898*sh_9_0*x + 0.974679434480896*sh_9_18*z
    if lmax == 10:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
            sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
            sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
            sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
            sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
            sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
            sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
        ], dim=-1)

    sh_11_0 = 0.977008420918394*sh_10_0*z + 0.977008420918394*sh_10_20*x
    sh_11_1 = 0.416597790450531*sh_10_0*y + 0.9315409787236*sh_10_1*z + 0.931540978723599*sh_10_19*x
    sh_11_2 = -0.0642824346533223*sh_10_0*z + 0.574959574576069*sh_10_1*y + 0.88607221316445*sh_10_18*x + 0.886072213164452*sh_10_2*z + 0.0642824346533226*sh_10_20*x
    sh_11_3 = -0.111340442853781*sh_10_1*z + 0.84060190949577*sh_10_17*x + 0.111340442853781*sh_10_19*x + 0.686348585024614*sh_10_2*y + 0.840601909495769*sh_10_3*z
    sh_11_4 = 0.795129803842541*sh_10_16*x + 0.157459164324444*sh_10_18*x - 0.157459164324443*sh_10_2*z + 0.771389215839871*sh_10_3*y + 0.795129803842541*sh_10_4*z
    sh_11_5 = 0.74965556829412*sh_10_15*x + 0.203278907045435*sh_10_17*x - 0.203278907045436*sh_10_3*z + 0.838140405208444*sh_10_4*y + 0.74965556829412*sh_10_5*z
    sh_11_6 = 0.70417879021953*sh_10_14*x + 0.248964798865985*sh_10_16*x - 0.248964798865985*sh_10_4*z + 0.890723542830247*sh_10_5*y + 0.704178790219531*sh_10_6*z
    sh_11_7 = 0.658698943008611*sh_10_13*x + 0.294579122654903*sh_10_15*x - 0.294579122654903*sh_10_5*z + 0.9315409787236*sh_10_6*y + 0.658698943008611*sh_10_7*z
    sh_11_8 = 0.613215343783275*sh_10_12*x + 0.340150671524904*sh_10_14*x - 0.340150671524904*sh_10_6*z + 0.962091385841669*sh_10_7*y + 0.613215343783274*sh_10_8*z
    sh_11_9 = 0.567727090763491*sh_10_11*x + 0.385694607919935*sh_10_13*x - 0.385694607919935*sh_10_7*z + 0.983332166035633*sh_10_8*y + 0.56772709076349*sh_10_9*z
    sh_11_10 = 0.738548945875997*sh_10_10*x + 0.431219680932052*sh_10_12*x - 0.431219680932052*sh_10_8*z + 0.995859195463938*sh_10_9*y
    sh_11_11 = sh_10_10*y - 0.674199862463242*sh_10_11*z - 0.674199862463243*sh_10_9*x
    sh_11_12 = 0.738548945875996*sh_10_10*z + 0.995859195463939*sh_10_11*y - 0.431219680932052*sh_10_12*z - 0.431219680932053*sh_10_8*x
    sh_11_13 = 0.567727090763491*sh_10_11*z + 0.983332166035634*sh_10_12*y - 0.385694607919935*sh_10_13*z - 0.385694607919935*sh_10_7*x - 0.567727090763491*sh_10_9*x
    sh_11_14 = 0.613215343783275*sh_10_12*z + 0.96209138584167*sh_10_13*y - 0.340150671524904*sh_10_14*z - 0.340150671524904*sh_10_6*x - 0.613215343783274*sh_10_8*x
    sh_11_15 = 0.658698943008611*sh_10_13*z + 0.9315409787236*sh_10_14*y - 0.294579122654903*sh_10_15*z - 0.294579122654903*sh_10_5*x - 0.65869894300861*sh_10_7*x
    sh_11_16 = 0.70417879021953*sh_10_14*z + 0.890723542830246*sh_10_15*y - 0.248964798865985*sh_10_16*z - 0.248964798865985*sh_10_4*x - 0.70417879021953*sh_10_6*x
    sh_11_17 = 0.749655568294121*sh_10_15*z + 0.838140405208444*sh_10_16*y - 0.203278907045436*sh_10_17*z - 0.203278907045435*sh_10_3*x - 0.749655568294119*sh_10_5*x
    sh_11_18 = 0.79512980384254*sh_10_16*z + 0.77138921583987*sh_10_17*y - 0.157459164324443*sh_10_18*z - 0.157459164324444*sh_10_2*x - 0.795129803842541*sh_10_4*x
    sh_11_19 = -0.111340442853782*sh_10_1*x + 0.84060190949577*sh_10_17*z + 0.686348585024614*sh_10_18*y - 0.111340442853781*sh_10_19*z - 0.840601909495769*sh_10_3*x
    sh_11_20 = -0.0642824346533226*sh_10_0*x + 0.886072213164451*sh_10_18*z + 0.57495957457607*sh_10_19*y - 0.886072213164451*sh_10_2*x - 0.0642824346533228*sh_10_20*z
    sh_11_21 = -0.9315409787236*sh_10_1*x + 0.931540978723599*sh_10_19*z + 0.416597790450531*sh_10_20*y
    sh_11_22 = -0.977008420918393*sh_10_0*x + 0.977008420918393*sh_10_20*z
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
        sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
    ], dim=-1)


def collate_fn(graph_list):
    return Collater(if_lcmp=True)(graph_list)


class Collater:
    def __init__(self, if_lcmp):
        self.if_lcmp = if_lcmp

    def __call__(self, graph_list):
        if self.if_lcmp:
            # 对于 1 个晶体图有 n 个节点 m 个边, 每个边两端有 2 个节点, 一共有 2m 个边两侧节点, 2m 个节点中每个节点又有各自的
            # 近邻, 总得算起来 2m 个节点在 1 个晶体图中带来 M 个近邻. 1 个晶体图的 subgraph_atom_idx shape 是 (M, 2), 取值是
            # 晶体图的节点 index 也就是 0 ~ n-1; subgraph_edge_idx shape 是 (M), 取值是晶体图的边 index 也就是 0 ~ m-1;
            # subgraph_edge_ang_batch shape 是 (M, num_l^2); subgraph_index 是用于对卷积后的节点更新做 scatter_add() 的index,
            # shape 是 (M), 取值是 0 ~ 2m-1

            batch = Batch.from_data_list(graph_list)

            subgraph_atom_idx_batch = []
            subgraph_edge_idx_batch = []
            subgraph_edge_ang_batch = []
            subgraph_index_batch = []
            # batch_edge = [] # edge 的 batch.batch
            for index_batch, (subgraph_atom_idx, subgraph_edge_idx, subgraph_edge_ang, subgraph_index) in enumerate(
                    batch.subgraph):
                subgraph_atom_idx_batch.append(subgraph_atom_idx + batch.__slices__['x'][index_batch])
                subgraph_edge_idx_batch.append(subgraph_edge_idx + batch.__slices__['edge_attr'][index_batch])
                subgraph_edge_ang_batch.append(subgraph_edge_ang)
                subgraph_index_batch.append(subgraph_index + batch.__slices__['edge_attr'][index_batch] * 2)

            subgraph_atom_idx_batch = torch.cat(subgraph_atom_idx_batch, dim=0)
            subgraph_edge_idx_batch = torch.cat(subgraph_edge_idx_batch, dim=0)
            subgraph_edge_ang_batch = torch.cat(subgraph_edge_ang_batch, dim=0)
            subgraph_index_batch = torch.cat(subgraph_index_batch, dim=0)

            subgraph = (subgraph_atom_idx_batch, subgraph_edge_idx_batch, subgraph_edge_ang_batch, subgraph_index_batch)

            return batch, subgraph
        else:
            return Batch.from_data_list(graph_list)


def load_orbital_types(path, return_orbital_types=False):
    orbital_types = []
    with open(path) as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in orbital_types]
    if return_orbital_types:
        return atom_num_orbital, orbital_types
    else:
        return atom_num_orbital


"""
The function get_graph below is extended from "https://github.com/materialsproject/pymatgen", which has the MIT License below

---------------------------------------------------------------------------
The MIT License (MIT)
Copyright (c) 2011-2012 MIT & The Regents of the University of California, through Lawrence Berkeley National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
def get_graph(cart_coords, frac_coords, numbers, stru_id, r, max_num_nbr, numerical_tol, lattice,
              default_dtype_torch, tb_folder, interface, num_l, create_from_DFT, if_lcmp_graph,
              separate_onsite, target='hamiltonian', huge_structure=False, only_get_R_list=False, if_new_sp=False,
              if_require_grad=False, fid_rc=None, **kwargs):
    assert target in ['hamiltonian', 'phiVdphi', 'density_matrix', 'O_ij', 'E_ij', 'E_i']
    if target == 'density_matrix' or target == 'O_ij':
        assert interface == 'h5' or interface == 'h5_rc_only'
    if target == 'E_ij':
        assert interface == 'h5'
        assert create_from_DFT is True
        assert separate_onsite is True
    if target == 'E_i':
        assert interface == 'h5'
        assert if_lcmp_graph is False
        assert separate_onsite is True
    if create_from_DFT:
        assert tb_folder is not None
        assert max_num_nbr == 0
        if interface == 'h5_rc_only' and target == 'E_ij':
            raise NotImplementedError
        elif interface == 'h5' or (interface == 'h5_rc_only' and target != 'E_ij'):
            key_atom_list = [[] for _ in range(len(numbers))]
            edge_idx, edge_fea, edge_idx_first = [], [], []
            if if_lcmp_graph:
                atom_idx_connect, edge_idx_connect = [], []
                edge_idx_connect_cursor = 0
            if target == 'E_ij':
                fid = h5py.File(os.path.join(tb_folder, 'E_delta_ee_ij.h5'), 'r')
            else:
                if if_require_grad:
                    fid = fid_rc
                else:
                    fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
            for k in fid.keys():
                key = json.loads(k)
                key_tensor = torch.tensor([key[0], key[1], key[2], key[3] - 1, key[4] - 1]) # (R, i, j) i and j is 0-based index
                if separate_onsite:
                    if key[0] == 0 and key[1] == 0 and key[2] == 0 and key[3] == key[4]:
                        continue
                key_atom_list[key[3] - 1].append(key_tensor)
            if target != 'E_ij' and not if_require_grad:
                fid.close()

            for index_first, (cart_coord, keys_tensor) in enumerate(zip(cart_coords, key_atom_list)):
                keys_tensor = torch.stack(keys_tensor)
                cart_coords_j = cart_coords[keys_tensor[:, 4]] + keys_tensor[:, :3].type(default_dtype_torch).to(cart_coords.device) @ lattice.to(cart_coords.device)
                dist = torch.norm(cart_coords_j - cart_coord[None, :], dim=1)
                len_nn = keys_tensor.shape[0]
                edge_idx_first.extend([index_first] * len_nn)
                edge_idx.extend(keys_tensor[:, 4].tolist())

                edge_fea_single = torch.cat([dist.view(-1, 1), cart_coord.view(1, 3).expand(len_nn, 3)], dim=-1)
                edge_fea_single = torch.cat([edge_fea_single, cart_coords_j, cart_coords[keys_tensor[:, 4]]], dim=-1)
                edge_fea.append(edge_fea_single)

                if if_lcmp_graph:
                    atom_idx_connect.append(keys_tensor[:, 4])
                    edge_idx_connect.append(range(edge_idx_connect_cursor, edge_idx_connect_cursor + len_nn))
                    edge_idx_connect_cursor += len_nn

            edge_fea = torch.cat(edge_fea).type(default_dtype_torch)
            edge_idx = torch.stack([torch.LongTensor(edge_idx_first), torch.LongTensor(edge_idx)])
        else:
            raise NotImplemented
    else:
        cart_coords_np = cart_coords.detach().numpy()
        frac_coords_np = frac_coords.detach().numpy()
        lattice_np = lattice.detach().numpy()
        num_atom = cart_coords.shape[0]

        center_coords_min = np.min(cart_coords_np, axis=0)
        center_coords_max = np.max(cart_coords_np, axis=0)
        global_min = center_coords_min - r - numerical_tol
        global_max = center_coords_max + r + numerical_tol
        global_min_torch = torch.tensor(global_min)
        global_max_torch = torch.tensor(global_max)

        reciprocal_lattice = np.linalg.inv(lattice_np).T * 2 * np.pi
        recp_len = np.sqrt(np.sum(reciprocal_lattice ** 2, axis=1))
        maxr = np.ceil((r + 0.15) * recp_len / (2 * np.pi))
        nmin = np.floor(np.min(frac_coords_np, axis=0)) - maxr
        nmax = np.ceil(np.max(frac_coords_np, axis=0)) + maxr
        all_ranges = [np.arange(x, y, dtype='int64') for x, y in zip(nmin, nmax)]
        images = torch.tensor(list(itertools.product(*all_ranges))).type_as(lattice)

        if only_get_R_list:
            return images

        coords = (images @ lattice)[:, None, :] + cart_coords[None, :, :]
        indices = torch.arange(num_atom).unsqueeze(0).expand(images.shape[0], num_atom)
        valid_index_bool = coords.gt(global_min_torch) * coords.lt(global_max_torch)
        valid_index_bool = valid_index_bool.all(dim=-1)
        valid_coords = coords[valid_index_bool]
        valid_indices = indices[valid_index_bool]


        valid_coords_np = valid_coords.detach().numpy()
        all_cube_index = _compute_cube_index(valid_coords_np, global_min, r)
        nx, ny, nz = _compute_cube_index(global_max, global_min, r) + 1
        all_cube_index = _three_to_one(all_cube_index, ny, nz)
        site_cube_index = _three_to_one(_compute_cube_index(cart_coords_np, global_min, r), ny, nz)
        cube_to_coords_index = collections.defaultdict(list)  # type: Dict[int, List]

        for index, cart_coord in enumerate(all_cube_index.ravel()):
            cube_to_coords_index[cart_coord].append(index)

        site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)

        edge_idx, edge_fea, edge_idx_first = [], [], []
        if if_lcmp_graph:
            atom_idx_connect, edge_idx_connect = [], []
            edge_idx_connect_cursor = 0
        for index_first, (cart_coord, j) in enumerate(zip(cart_coords, site_neighbors)):
            l1 = np.array(_three_to_one(j, ny, nz), dtype=int).ravel()
            ks = [k for k in l1 if k in cube_to_coords_index]
            nn_coords_index = np.concatenate([cube_to_coords_index[k] for k in ks], axis=0)
            nn_coords = valid_coords[nn_coords_index]
            nn_indices = valid_indices[nn_coords_index]
            dist = torch.norm(nn_coords - cart_coord[None, :], dim=1)

            if separate_onsite is False:
                nn_coords = nn_coords.squeeze()
                nn_indices = nn_indices.squeeze()
                dist = dist.squeeze()
            else:
                nonzero_index = dist.nonzero(as_tuple=False)
                nn_coords = nn_coords[nonzero_index]
                nn_coords = nn_coords.squeeze(1)
                nn_indices = nn_indices[nonzero_index].view(-1)
                dist = dist[nonzero_index].view(-1)

            if max_num_nbr > 0:
                if len(dist) >= max_num_nbr:
                    dist_top, index_top = dist.topk(max_num_nbr, largest=False, sorted=True)
                    edge_idx.extend(nn_indices[index_top])
                    if if_lcmp_graph:
                        atom_idx_connect.append(nn_indices[index_top])
                    edge_idx_first.extend([index_first] * len(index_top))
                    edge_fea_single = torch.cat([dist_top.view(-1, 1), cart_coord.view(1, 3).expand(len(index_top), 3)], dim=-1)
                    edge_fea_single = torch.cat([edge_fea_single, nn_coords[index_top], cart_coords[nn_indices[index_top]]], dim=-1)
                    edge_fea.append(edge_fea_single)
                else:
                    warnings.warn("Can not find a number of max_num_nbr atoms within radius")
                    edge_idx.extend(nn_indices)
                    if if_lcmp_graph:
                        atom_idx_connect.append(nn_indices)
                    edge_idx_first.extend([index_first] * len(nn_indices))
                    edge_fea_single = torch.cat([dist.view(-1, 1), cart_coord.view(1, 3).expand(len(nn_indices), 3)], dim=-1)
                    edge_fea_single = torch.cat([edge_fea_single, nn_coords, cart_coords[nn_indices]], dim=-1)
                    edge_fea.append(edge_fea_single)
            else:
                index_top = dist.lt(r + numerical_tol)
                edge_idx.extend(nn_indices[index_top])
                if if_lcmp_graph:
                    atom_idx_connect.append(nn_indices[index_top])
                edge_idx_first.extend([index_first] * len(nn_indices[index_top]))
                edge_fea_single = torch.cat([dist[index_top].view(-1, 1), cart_coord.view(1, 3).expand(len(nn_indices[index_top]), 3)], dim=-1)
                edge_fea_single = torch.cat([edge_fea_single, nn_coords[index_top], cart_coords[nn_indices[index_top]]], dim=-1)
                edge_fea.append(edge_fea_single)
            if if_lcmp_graph:
                edge_idx_connect.append(range(edge_idx_connect_cursor, edge_idx_connect_cursor + len(atom_idx_connect[-1])))
                edge_idx_connect_cursor += len(atom_idx_connect[-1])


        edge_fea = torch.cat(edge_fea).type(default_dtype_torch)
        edge_idx_first = torch.LongTensor(edge_idx_first)
        edge_idx = torch.stack([edge_idx_first, torch.LongTensor(edge_idx)])


    if tb_folder is not None:
        if target == 'E_ij':
            read_file_list = ['E_ij.h5', 'E_delta_ee_ij.h5', 'E_xc_ij.h5']
            graph_key_list = ['E_ij', 'E_delta_ee_ij', 'E_xc_ij']
            read_terms_dict = {}
            for read_file, graph_key in zip(read_file_list, graph_key_list):
                read_terms = {}
                fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                for k, v in fid.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                    read_terms[key] = torch.tensor(v[...], dtype=default_dtype_torch)
                read_terms_dict[graph_key] = read_terms
                fid.close()

            local_rotation_dict = {}
            if if_require_grad:
                fid = fid_rc
            else:
                fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
            for k, v in fid.items():
                key = json.loads(k)
                key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)  # (R, i, j) i and j is 0-based index
                if if_require_grad:
                    local_rotation_dict[key] = v
                else:
                    local_rotation_dict[key] = torch.tensor(v, dtype=default_dtype_torch)
            if not if_require_grad:
                fid.close()
        elif target == 'E_i':
            read_file_list = ['E_i.h5']
            graph_key_list = ['E_i']
            read_terms_dict = {}
            for read_file, graph_key in zip(read_file_list, graph_key_list):
                read_terms = {}
                fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                for k, v in fid.items():
                    index_i = int(k)  # index_i is 0-based index
                    read_terms[index_i] = torch.tensor(v[...], dtype=default_dtype_torch)
                fid.close()
                read_terms_dict[graph_key] = read_terms
        else:
            if interface == 'h5' or interface == 'h5_rc_only':
                atom_num_orbital = load_orbital_types(os.path.join(tb_folder, 'orbital_types.dat'))

                if interface == 'h5':
                    with open(os.path.join(tb_folder, 'info.json'), 'r') as info_f:
                        info_dict = json.load(info_f)
                        spinful = info_dict["isspinful"]

                if interface == 'h5':
                    if target == 'hamiltonian':
                        read_file_list = ['rh.h5']
                        graph_key_list = ['term_real']
                    elif target == 'phiVdphi':
                        read_file_list = ['rphiVdphi.h5']
                        graph_key_list = ['term_real']
                    elif target == 'density_matrix':
                        read_file_list = ['rdm.h5']
                        graph_key_list = ['term_real']
                    elif target == 'O_ij':
                        read_file_list = ['rh.h5', 'rdm.h5', 'rvna.h5', 'rvdee.h5', 'rvxc.h5']
                        graph_key_list = ['rh', 'rdm', 'rvna', 'rvdee', 'rvxc']
                    else:
                        raise ValueError('Unknown prediction target: {}'.format(target))
                    read_terms_dict = {}
                    for read_file, graph_key in zip(read_file_list, graph_key_list):
                        read_terms = {}
                        fid = h5py.File(os.path.join(tb_folder, read_file), 'r')
                        for k, v in fid.items():
                            key = json.loads(k)
                            key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                            if spinful:
                                num_orbital_row = atom_num_orbital[key[3]]
                                num_orbital_column = atom_num_orbital[key[4]]
                                # soc block order:
                                # 1 3
                                # 4 2
                                if target == 'phiVdphi':
                                    raise NotImplementedError
                                else:
                                    read_value = torch.stack([
                                        torch.tensor(v[:num_orbital_row, :num_orbital_column].real, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, :num_orbital_column].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, num_orbital_column:].real, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, num_orbital_column:].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, num_orbital_column:].real, dtype=default_dtype_torch),
                                        torch.tensor(v[:num_orbital_row, num_orbital_column:].imag, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, :num_orbital_column].real, dtype=default_dtype_torch),
                                        torch.tensor(v[num_orbital_row:, :num_orbital_column].imag, dtype=default_dtype_torch)
                                    ], dim=-1)
                                read_terms[key] = read_value
                            else:
                                read_terms[key] = torch.tensor(v, dtype=default_dtype_torch)
                        read_terms_dict[graph_key] = read_terms
                        fid.close()

                local_rotation_dict = {}
                if if_require_grad:
                    fid = fid_rc
                else:
                    fid = h5py.File(os.path.join(tb_folder, 'rc.h5'), 'r')
                for k, v in fid.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)  # (R, i, j) i and j is 0-based index
                    if if_require_grad:
                        local_rotation_dict[key] = v
                    else:
                        local_rotation_dict[key] = torch.tensor(v, dtype=default_dtype_torch)
                if not if_require_grad:
                    fid.close()

                max_num_orbital = max(atom_num_orbital)

            elif interface == 'npz' or interface == 'npz_rc_only':
                spinful = False
                atom_num_orbital = load_orbital_types(os.path.join(tb_folder, 'orbital_types.dat'))

                if interface == 'npz':
                    graph_key_list = ['term_real']
                    read_terms_dict = {'term_real': {}}
                    hopping_dict_read = np.load(os.path.join(tb_folder, 'rh.npz'))
                    for k, v in hopping_dict_read.items():
                        key = json.loads(k)
                        key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)  # (R, i, j) i and j is 0-based index
                        read_terms_dict['term_real'][key] = torch.tensor(v, dtype=default_dtype_torch)

                local_rotation_dict = {}
                local_rotation_dict_read = np.load(os.path.join(tb_folder, 'rc.npz'))
                for k, v in local_rotation_dict_read.items():
                    key = json.loads(k)
                    key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                    local_rotation_dict[key] = torch.tensor(v, dtype=default_dtype_torch)

                max_num_orbital = max(atom_num_orbital)
            else:
                raise ValueError(f'Unknown interface: {interface}')


        # process data
        if target == 'E_i':
            term_dict = {}
            onsite_term_dict = {}
            for graph_key in graph_key_list:
                term_dict[graph_key] = torch.full([numbers.shape[0], 1], np.nan, dtype=default_dtype_torch)
            for index_atom in range(numbers.shape[0]):
                assert index_atom in read_terms_dict[graph_key_list[0]]
                for graph_key in graph_key_list:
                    term_dict[graph_key][index_atom] = read_terms_dict[graph_key][index_atom]
            subgraph = None
        else:
            if interface == 'h5_rc_only' or interface == 'npz_rc_only':
                local_rotation = []
            else:
                term_dict = {}
                onsite_term_dict = {}
                if target == 'E_ij':
                    for graph_key in graph_key_list:
                        term_dict[graph_key] = torch.full([edge_fea.shape[0], 1], np.nan, dtype=default_dtype_torch)
                    local_rotation = []
                    if separate_onsite is True:
                        for graph_key in graph_key_list:
                            onsite_term_dict['onsite_' + graph_key] = torch.full([numbers.shape[0], 1], np.nan, dtype=default_dtype_torch)
                else:
                    term_mask = torch.zeros(edge_fea.shape[0], dtype=torch.bool)
                    for graph_key in graph_key_list:
                        if spinful:
                            term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital, 8],
                                                              np.nan, dtype=default_dtype_torch)
                        else:
                            if target == 'phiVdphi':
                                term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital, 3],
                                                                  np.nan, dtype=default_dtype_torch)
                            else:
                                term_dict[graph_key] = torch.full([edge_fea.shape[0], max_num_orbital, max_num_orbital],
                                                                  np.nan, dtype=default_dtype_torch)
                    local_rotation = []
                    if separate_onsite is True:
                        for graph_key in graph_key_list:
                            if spinful:
                                onsite_term_dict['onsite_' + graph_key] = torch.full(
                                    [numbers.shape[0], max_num_orbital, max_num_orbital, 8],
                                    np.nan, dtype=default_dtype_torch)
                            else:
                                if target == 'phiVdphi':
                                    onsite_term_dict['onsite_' + graph_key] = torch.full(
                                        [numbers.shape[0], max_num_orbital, max_num_orbital, 3],
                                        np.nan, dtype=default_dtype_torch)
                                else:
                                    onsite_term_dict['onsite_' + graph_key] = torch.full(
                                        [numbers.shape[0], max_num_orbital, max_num_orbital],
                                        np.nan, dtype=default_dtype_torch)

            inv_lattice = torch.inverse(lattice).type(default_dtype_torch)
            for index_edge in range(edge_fea.shape[0]):
                # h_{i0, jR} i and j is 0-based index
                R = torch.round(edge_fea[index_edge, 4:7].cpu() @ inv_lattice - edge_fea[index_edge, 7:10].cpu() @ inv_lattice).int().tolist()
                i, j = edge_idx[:, index_edge]

                key_term = (*R, i.item(), j.item())
                if interface == 'h5_rc_only' or interface == 'npz_rc_only':
                    local_rotation.append(local_rotation_dict[key_term])
                else:
                    if key_term in read_terms_dict[graph_key_list[0]]:
                        for graph_key in graph_key_list:
                            if target == 'E_ij':
                                term_dict[graph_key][index_edge] = read_terms_dict[graph_key][key_term]
                            else:
                                term_mask[index_edge] = True
                                if spinful:
                                    term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j], :] = read_terms_dict[graph_key][key_term]
                                else:
                                    term_dict[graph_key][index_edge, :atom_num_orbital[i], :atom_num_orbital[j]] = read_terms_dict[graph_key][key_term]
                        local_rotation.append(local_rotation_dict[key_term])
                    else:
                        raise NotImplementedError(
                            "Not yet have support for graph radius including hopping without calculation")

            if separate_onsite is True and interface != 'h5_rc_only' and interface != 'npz_rc_only':
                for index_atom in range(numbers.shape[0]):
                    key_term = (0, 0, 0, index_atom, index_atom)
                    assert key_term in read_terms_dict[graph_key_list[0]]
                    for graph_key in graph_key_list:
                        if target == 'E_ij':
                            onsite_term_dict['onsite_' + graph_key][index_atom] = read_terms_dict[graph_key][key_term]
                        else:
                            if spinful:
                                onsite_term_dict['onsite_' + graph_key][index_atom, :atom_num_orbital[i], :atom_num_orbital[j], :] = \
                                read_terms_dict[graph_key][key_term]
                            else:
                                onsite_term_dict['onsite_' + graph_key][index_atom, :atom_num_orbital[i], :atom_num_orbital[j]] = \
                                read_terms_dict[graph_key][key_term]

            if if_lcmp_graph:
                local_rotation = torch.stack(local_rotation, dim=0)
                assert local_rotation.shape[0] == edge_fea.shape[0]
                r_vec = edge_fea[:, 1:4] - edge_fea[:, 4:7]
                r_vec = r_vec.unsqueeze(1)
                if huge_structure is False:
                    r_vec = torch.matmul(r_vec[:, None, :, :], local_rotation[None, :, :, :].to(r_vec.device)).reshape(-1, 3)
                    if if_new_sp:
                        r_vec = torch.nn.functional.normalize(r_vec, dim=-1)
                        angular_expansion = _spherical_harmonics(num_l - 1, -r_vec[..., 2], r_vec[..., 0],
                                                                 r_vec[..., 1])
                        angular_expansion.mul_(torch.cat([
                            (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                         dtype=angular_expansion.dtype,
                                                                                         device=angular_expansion.device)
                            for l in range(num_l)
                        ]))
                        angular_expansion = angular_expansion.reshape(edge_fea.shape[0], edge_fea.shape[0], -1)
                    else:
                        r_vec_sp = get_spherical_from_cartesian(r_vec)
                        sph_harm_func = SphericalHarmonics()
                        angular_expansion = []
                        for l in range(num_l):
                            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                        angular_expansion = torch.cat(angular_expansion, dim=-1).reshape(edge_fea.shape[0], edge_fea.shape[0], -1)
                    # shape (不同的边, 不同的local坐标, 边特征)

                subgraph_atom_idx_list = []
                subgraph_edge_idx_list = []
                subgraph_edge_ang_list = []
                subgraph_index = []
                index_cursor = 0

                for index in range(edge_fea.shape[0]):
                    # h_{i0, jR}
                    i, j = edge_idx[:, index]
                    #subgraph
                    subgraph_atom_idx = torch.stack([i.repeat(len(atom_idx_connect[i])), atom_idx_connect[i]]).T
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[i])
                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3)
                        if if_new_sp:
                            r_vec_tmp = torch.nn.functional.normalize(r_vec_tmp, dim=-1)
                            subgraph_edge_ang = _spherical_harmonics(num_l - 1, -r_vec_tmp[..., 2], r_vec_tmp[..., 0], r_vec_tmp[..., 1])
                            subgraph_edge_ang.mul_(torch.cat([
                                (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                             dtype=subgraph_edge_ang.dtype,
                                                                                             device=subgraph_edge_ang.device)
                                for l in range(num_l)
                            ]))
                        else:
                            r_vec_sp = get_spherical_from_cartesian(r_vec_tmp)
                            sph_harm_func = SphericalHarmonics()
                            angular_expansion = []
                            for l in range(num_l):
                                angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                            subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2)
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :]

                    subgraph_atom_idx_list.append(subgraph_atom_idx)
                    subgraph_edge_idx_list.append(subgraph_edge_idx)
                    subgraph_edge_ang_list.append(subgraph_edge_ang)
                    subgraph_index += [index_cursor] * len(atom_idx_connect[i])
                    index_cursor += 1

                    subgraph_atom_idx = torch.stack([j.repeat(len(atom_idx_connect[j])), atom_idx_connect[j]]).T
                    subgraph_edge_idx = torch.LongTensor(edge_idx_connect[j])
                    if huge_structure:
                        r_vec_tmp = torch.matmul(r_vec[subgraph_edge_idx, :, :], local_rotation[index, :, :].to(r_vec.device)).reshape(-1, 3)
                        if if_new_sp:
                            r_vec_tmp = torch.nn.functional.normalize(r_vec_tmp, dim=-1)
                            subgraph_edge_ang = _spherical_harmonics(num_l - 1, -r_vec_tmp[..., 2], r_vec_tmp[..., 0], r_vec_tmp[..., 1])
                            subgraph_edge_ang.mul_(torch.cat([
                                (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1,
                                                                                             dtype=subgraph_edge_ang.dtype,
                                                                                             device=subgraph_edge_ang.device)
                                for l in range(num_l)
                            ]))
                        else:
                            r_vec_sp = get_spherical_from_cartesian(r_vec_tmp)
                            sph_harm_func = SphericalHarmonics()
                            angular_expansion = []
                            for l in range(num_l):
                                angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
                            subgraph_edge_ang = torch.cat(angular_expansion, dim=-1).reshape(-1, num_l ** 2)
                    else:
                        subgraph_edge_ang = angular_expansion[subgraph_edge_idx, index, :]
                    subgraph_atom_idx_list.append(subgraph_atom_idx)
                    subgraph_edge_idx_list.append(subgraph_edge_idx)
                    subgraph_edge_ang_list.append(subgraph_edge_ang)
                    subgraph_index += [index_cursor] * len(atom_idx_connect[j])
                    index_cursor += 1
                subgraph = (torch.cat(subgraph_atom_idx_list, dim=0),
                            torch.cat(subgraph_edge_idx_list, dim=0),
                            torch.cat(subgraph_edge_ang_list, dim=0),
                            torch.LongTensor(subgraph_index))
            else:
                subgraph = None

        if interface == 'h5_rc_only' or interface == 'npz_rc_only':
            data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, term_mask=None,
                        term_real=None, onsite_term_real=None,
                        atom_num_orbital=torch.tensor(atom_num_orbital),
                        subgraph=subgraph,
                        **kwargs)
        else:
            if target == 'E_ij' or target == 'E_i':
                data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id,
                            **term_dict, **onsite_term_dict,
                            subgraph=subgraph,
                            spinful=False,
                            **kwargs)
            else:
                data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, term_mask=term_mask,
                            **term_dict, **onsite_term_dict,
                            atom_num_orbital=torch.tensor(atom_num_orbital),
                            subgraph=subgraph,
                            spinful=spinful,
                            **kwargs)
    else:
        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=stru_id, **kwargs)
    return data
