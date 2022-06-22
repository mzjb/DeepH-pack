# Definition
- 所有的结构建议在任何计算之前都先经过 to_unit_cell 操作, 因为 openmx 似乎会对输入的结构做 to_unit_cell 操作
- 1 Hartree = 27.2113845 eV
- *site_positions.dat*: 晶体的笛卡尔坐标, 矩阵文本格式, 一共有 3 行对应于 xyz 方向, 每一列对应于一个原子. python 可以使用
    ```python
    import numpy as np
    np.loadtxt('site_positions.dat').T
    ```
    读取, julia 可以使用
    ```julia
    using DelimitedFiles
    readdlm("site_positions.dat")
    ```
    读取.
- *lat.dat*: 晶体的正格矢, 矩阵文本格式, 第一列是 a, 第二列是 b, 第三列是 c. python 可以使用
    ```python
    import numpy as np
    np.loadtxt('lat.dat').T
    ```
    读取为行矢量, julia 可以使用
    ```julia
    using DelimitedFiles
    readdlm("lat.dat")
    ```
    读取为列矢量.
- *rlat.dat*: 晶体的倒格矢, 矩阵文本格式, 第一列是 a', 第二列是 b', 第三列是 c'. python 可以使用
    ```python
    import numpy as np
    np.loadtxt('rlat.dat').T
    ```
    读取为行矢量, julia 可以使用
    ```julia
    using DelimitedFiles
    readdlm("rlat.dat")
    ```
    读取为列矢量.
- *element.dat*: 结构的原子序数, 文本格式. python 可以使用
    ```python
    import numpy as np
    np.loadtxt('element.dat')
    ```
    读取, julia 可以使用
    ```julia
    using DelimitedFiles
    readdlm("element.dat")
    ```
    读取.
- *rc.h5*: 结构每个 bond local coordinate 相对结构文件坐标系的转动矩阵, 使用 HDF5 格式存储, 此 HDF5 格式下只有一级目录, 每个 dataset 的 key 是形如 
    ```python
    f"[{Rx}, {Ry}, {Rz}, {i}, {j}]"
    ```
    这代表原子 i 和原子 j 之间的键, 其中 i 在 (0, 0, 0) 这个原胞中, j 在 (Rx, Ry, Rz) 这个原胞中, 其中 i 和 j 是从 1 开始数的. python 可以使用
    ```python
    import h5py
    import json
    import numpy as np
    fid = h5py.File("rc.h5", "r")

    for key_str, v in fid:
        key = json.loads(key_str)
        R = (key[0], key[1], key[2])
        atom_i = key[3] - 1
        atom_j = key[4] - 1
        rc = np.array(v)
    
    fid.close()
    ```
    读取并遍历. julia 可以使用
    ```julia
    using HDF5
    fid = h5open("rc.h5", "r")
    
    T = eltype(fid[keys(fid)[1]])
    d_out = Dict{Array{Int64,1},T}()
    for key in keys(fid)
        data = read(fid[key])
        nk = map(x -> parse(Int64, convert(String, x)), split(key[2 : length(key) - 1], ','))
        d_out[nk] = permutedims(data)
    end
    
    close(fid)
    ```
    读取并遍历.
