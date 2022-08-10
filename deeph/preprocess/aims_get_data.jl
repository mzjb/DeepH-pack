using JSON
using HDF5
using LinearAlgebra
using DelimitedFiles
using StaticArrays
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "NoTB.dat, basis-indices.out, geometry.in"
            arg_type = String
            default = "./"
        "--output_dir", "-o"
            help = ""
            arg_type = String
            default = "./output"
        "--save_overlap", "-s"
            help = ""
            arg_type = Bool
            default = false
        "--save_position", "-p"
            help = ""
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end
parsed_args = parse_commandline()

input_dir = abspath(parsed_args["input_dir"])
output_dir = abspath(parsed_args["output_dir"])

@assert isfile(joinpath(input_dir, "NoTB.dat"))
@assert isfile(joinpath(input_dir, "basis-indices.out"))
@assert isfile(joinpath(input_dir, "geometry.in"))

# @info string("get data from: ", input_dir)
periodic_table = JSON.parsefile(joinpath(@__DIR__, "periodic_table.json"))
mkpath(output_dir)

# The function parse_openmx below is come from "https://github.com/HopTB/HopTB.jl"
f = open(joinpath(input_dir, "NoTB.dat"))
# number of basis
@assert occursin("n_basis", readline(f)) # start
norbits = parse(Int64, readline(f))
@assert occursin("end", readline(f)) # end
@assert occursin("n_ham", readline(f)) # start
nhams = parse(Int64, readline(f))
@assert occursin("end", readline(f)) # end
@assert occursin("n_cell", readline(f)) # start
ncells = parse(Int64, readline(f))
@assert occursin("end", readline(f)) # end
# lattice vector
@assert occursin("lattice_vector", readline(f)) # start
lat = Matrix{Float64}(I, 3, 3)
for i in 1:3
    lat[:, i] = map(x->parse(Float64, x), split(readline(f)))
end
@assert occursin("end", readline(f)) # end
# hamiltonian
@assert occursin("hamiltonian", readline(f)) # start
hamiltonian = zeros(nhams)
i = 1
while true
    global i
    @assert !eof(f)
    ln = split(readline(f))
    if occursin("end", ln[1]) break end
    hamiltonian[i:i + length(ln) - 1] = map(x->parse(Float64, x), ln)
    i += length(ln)
end
# overlaps
@assert occursin("overlap", readline(f)) # start
overlaps = zeros(nhams)
i = 1
while true
    global i
    @assert !eof(f)
    ln = split(readline(f))
    if occursin("end", ln[1]) break end
    overlaps[i:i + length(ln) - 1] = map(x->parse(Float64, x), ln)
    i += length(ln)
end
# index hamiltonian
@assert occursin("index_hamiltonian", readline(f)) # start
indexhamiltonian = zeros(Int64, ncells * norbits, 4)
i = 1
while true
    global i
    @assert !eof(f)
    ln = split(readline(f))
    if occursin("end", ln[1]) break end
    indexhamiltonian[i, :] = map(x->parse(Int64, x), ln)
    i += 1
end
# cell index
@assert occursin("cell_index", readline(f)) # start
cellindex = zeros(Int64, ncells, 3)
i = 1
while true
    global i
    @assert !eof(f)
    ln = split(readline(f))
    if occursin("end", ln[1]) break end
    if i <= ncells
        cellindex[i, :] = map(x->parse(Int64, x), ln)
    end
    i += 1
end
# column index hamiltonian
@assert occursin("column_index_hamiltonian", readline(f)) # start
columnindexhamiltonian = zeros(Int64, nhams)
i = 1
while true
    global i
    @assert !eof(f)
    ln = split(readline(f))
    if occursin("end", ln[1]) break end
    columnindexhamiltonian[i:i + length(ln) - 1] = map(x->parse(Int64, x), ln)
    i += length(ln)
end
# positions
positions = zeros(nhams, 3)
for dir in 1:3
    positionsdir = zeros(nhams)
    @assert occursin("position", readline(f)) # start
    readline(f) # skip direction
    i = 1
    while true
        @assert !eof(f)
        ln = split(readline(f))
        if occursin("end", ln[1]) break end
        positionsdir[i:i + length(ln) - 1] = map(x->parse(Float64, x), ln)
        i += length(ln)
    end
    positions[:, dir] = positionsdir
end
if !eof(f)
    spinful = true
    soc_matrix = zeros(nhams, 3)
    for dir in 1:3
        socdir = zeros(nhams)
        @assert occursin("soc_matrix", readline(f)) # start
        readline(f) # skip direction
        i = 1
        while true
            @assert !eof(f)
            ln = split(readline(f))
            if occursin("end", ln[1]) break end
            socdir[i:i + length(ln) - 1] = map(x->parse(Float64, x), ln)
            i += length(ln)
        end
        soc_matrix[:, dir] = socdir
    end
else
    spinful = false
end
close(f)

orbital_types = Array{Array{Int64,1},1}(undef, 0)
basis_dir = joinpath(input_dir, "basis-indices.out")
@assert ispath(basis_dir)
f = open(basis_dir)
readline(f)
@assert split(readline(f))[1] == "fn."
basis_indices = zeros(Int64, norbits, 4)
for index_orbit in 1:norbits
    line = map(x->parse(Int64, x), split(readline(f))[[1, 3, 4, 5, 6]])
    @assert line[1] == index_orbit
    basis_indices[index_orbit, :] = line[2:5]
    # basis_indices: 1 ia, 2 n, 3 l, 4 m
    if size(orbital_types, 1) < line[2]
        orbital_type = Array{Int64,1}(undef, 0)
        push!(orbital_types, orbital_type)
    end
    if line[4] == line[5]
        push!(orbital_types[line[2]], line[4])
    end
end
nsites = size(orbital_types, 1)
site_norbits = (x->sum(x .* 2 .+ 1)).(orbital_types) * (1 + spinful)
@assert norbits == sum(site_norbits)
site_norbits_cumsum = cumsum(site_norbits)
site_indices = zeros(Int64, norbits)
for index_site in 1:nsites
    if index_site == 1
        site_indices[1:site_norbits_cumsum[index_site]] .= index_site
    else
        site_indices[site_norbits_cumsum[index_site - 1] + 1:site_norbits_cumsum[index_site]] .= index_site
    end
end
close(f)

f = open(joinpath(input_dir, "geometry.in"))
# atom_frac_pos = zeros(Float64, 3, nsites)
element = Array{Int64,1}(undef, 0)
index_atom = 0
while !eof(f)
    line = split(readline(f))
    if size(line, 1) > 0 && line[1] == "atom_frac"
        global index_atom
        index_atom += 1
        # atom_frac_pos[:, index_atom] = map(x->parse(Float64, x), line[[2, 3, 4]])
        push!(element, periodic_table[line[5]]["Atomic no"])
    end
end
@assert index_atom == nsites
# site_positions = lat * atom_frac_pos
close(f)

@info string("spinful: ", spinful)
# write to file
site_positions = fill(NaN, (3, nsites))
overlaps_dict = Dict{Array{Int64, 1}, Array{Float64, 2}}()
positions_dict = Dict{Array{Int64, 1}, Array{Float64, 2}}()
R_list = Set{Vector{Int64}}()
if spinful
    hamiltonians_dict = Dict{Array{Int64, 1}, Array{Complex{Float64}, 2}}()
    @error "spinful not implemented yet"
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    σ0 = [1 0; 0 1]
    nm = TBModel{ComplexF64}(2*norbits, lat, isorthogonal=false)
    # convention here is first half up (spin=0); second half down (spin=1).
    for i in 1:size(indexhamiltonian, 1)
        for j in indexhamiltonian[i, 3]:indexhamiltonian[i, 4]
            for nspin in 0:1
                for mspin in 0:1
                    sethopping!(nm,
                        cellindex[indexhamiltonian[i, 1], :],
                        columnindexhamiltonian[j] + norbits * nspin,
                        indexhamiltonian[i, 2] + norbits * mspin,
                        σ0[nspin + 1, mspin + 1] * hamiltonian[j] -
                        (σx[nspin + 1, mspin + 1] * soc_matrix[j, 1] +
                        σy[nspin + 1, mspin + 1] * soc_matrix[j, 2] +
                        σz[nspin + 1, mspin + 1] * soc_matrix[j, 3]) * im)
                    setoverlap!(nm,
                        cellindex[indexhamiltonian[i, 1], :],
                        columnindexhamiltonian[j] + norbits * nspin,
                        indexhamiltonian[i, 2] + norbits * mspin,
                        σ0[nspin + 1, mspin + 1] * overlaps[j])
                end
            end
        end
    end
    for i in 1:size(indexhamiltonian, 1)
        for j in indexhamiltonian[i, 3]:indexhamiltonian[i, 4]
            for nspin in 0:1
                for mspin in 0:1
                    for dir in 1:3
                        setposition!(nm,
                            cellindex[indexhamiltonian[i, 1], :],
                            columnindexhamiltonian[j] + norbits * nspin,
                            indexhamiltonian[i, 2] + norbits * mspin,
                            dir,
                            σ0[nspin + 1, mspin + 1] * positions[j, dir])
                    end
                end
            end
        end
    end
    return nm
else
    hamiltonians_dict = Dict{Array{Int64, 1}, Array{Float64, 2}}()

    for i in 1:size(indexhamiltonian, 1)
        for j in indexhamiltonian[i, 3]:indexhamiltonian[i, 4]
            R = cellindex[indexhamiltonian[i, 1], :]
            push!(R_list, SVector{3, Int64}(R))
            orbital_i_whole = columnindexhamiltonian[j]
            orbital_j_whole = indexhamiltonian[i, 2]
            site_i = site_indices[orbital_i_whole]
            site_j = site_indices[orbital_j_whole]
            block_matrix_i = orbital_i_whole - site_norbits_cumsum[site_i] + site_norbits[site_i]
            block_matrix_j = orbital_j_whole - site_norbits_cumsum[site_j] + site_norbits[site_j]
            key = cat(dims=1, R, site_i, site_j)
            key_inv = cat(dims=1, -R, site_j, site_i)
            
            mi = 0
            mj = 0
            # p-orbital
            if basis_indices[orbital_i_whole, 3] == 1
                if basis_indices[orbital_i_whole, 4] == -1
                    block_matrix_i += 1
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 0
                    block_matrix_i += 1
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 1
                    block_matrix_i += -2
                    mi = 1
                end
            end
            if basis_indices[orbital_j_whole, 3] == 1
                if basis_indices[orbital_j_whole, 4] == -1
                    block_matrix_j += 1
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 0
                    block_matrix_j += 1
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 1
                    block_matrix_j += -2
                    mj = 1
                end
            end
            # d-orbital
            if basis_indices[orbital_i_whole, 3] == 2
                if basis_indices[orbital_i_whole, 4] == -2
                    block_matrix_i += 2
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == -1
                    block_matrix_i += 3
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 0
                    block_matrix_i += -2
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 1
                    block_matrix_i += 0
                    mi = 1
                elseif basis_indices[orbital_i_whole, 4] == 2
                    block_matrix_i += -3
                    mi = 0
                end
            end
            if basis_indices[orbital_j_whole, 3] == 2
                if basis_indices[orbital_j_whole, 4] == -2
                    block_matrix_j += 2
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == -1
                    block_matrix_j += 3
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 0
                    block_matrix_j += -2
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 1
                    block_matrix_j += 0
                    mj = 1
                elseif basis_indices[orbital_j_whole, 4] == 2
                    block_matrix_j += -3
                    mj = 0
                end
            end
            # f-orbital
            if basis_indices[orbital_i_whole, 3] == 3
                if basis_indices[orbital_i_whole, 4] == -3
                    block_matrix_i += 6
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == -2
                    block_matrix_i += 3
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == -1
                    block_matrix_i += 0
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 0
                    block_matrix_i += -3
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 1
                    block_matrix_i += -3
                    mi = 1
                elseif basis_indices[orbital_i_whole, 4] == 2
                    block_matrix_i += -2
                    mi = 0
                elseif basis_indices[orbital_i_whole, 4] == 3
                    block_matrix_i += -1
                    mi = 1
                end
            end
            if basis_indices[orbital_j_whole, 3] == 3
                if basis_indices[orbital_j_whole, 4] == -3
                    block_matrix_j += 6
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == -2
                    block_matrix_j += 3
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == -1
                    block_matrix_j += 0
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 0
                    block_matrix_j += -3
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 1
                    block_matrix_j += -3
                    mj = 1
                elseif basis_indices[orbital_j_whole, 4] == 2
                    block_matrix_j += -2
                    mj = 0
                elseif basis_indices[orbital_j_whole, 4] == 3
                    block_matrix_j += -1
                    mj = 1
                end
            end
            if (basis_indices[orbital_i_whole, 3] > 3) || (basis_indices[orbital_j_whole, 3] > 3)
                @error("The case of l>3 is not implemented")
            end

            if !(key ∈ keys(hamiltonians_dict))
                # overlaps_dict[key] = fill(convert(Float64, NaN), (site_norbits[site_i], site_norbits[site_j]))
                overlaps_dict[key] = zeros(Float64, site_norbits[site_i], site_norbits[site_j])
                hamiltonians_dict[key] = zeros(Float64, site_norbits[site_i], site_norbits[site_j])
                for direction in 1:3
                    positions_dict[cat(dims=1, key, direction)] = zeros(Float64, site_norbits[site_i], site_norbits[site_j])
                end
            end
            if !(key_inv ∈ keys(hamiltonians_dict))
                overlaps_dict[key_inv] = zeros(Float64, site_norbits[site_j], site_norbits[site_i])
                hamiltonians_dict[key_inv] = zeros(Float64, site_norbits[site_j], site_norbits[site_i])
                for direction in 1:3
                    positions_dict[cat(dims=1, key_inv, direction)] = zeros(Float64, site_norbits[site_j], site_norbits[site_i])
                end
            end
            overlaps_dict[key][block_matrix_i, block_matrix_j] = overlaps[j] * (-1) ^ (mi + mj)
            hamiltonians_dict[key][block_matrix_i, block_matrix_j] = hamiltonian[j] * (-1) ^ (mi + mj)
            for direction in 1:3
                positions_dict[cat(dims=1, key, direction)][block_matrix_i, block_matrix_j] = positions[j, direction] * (-1) ^ (mi + mj)
            end

            overlaps_dict[key_inv][block_matrix_j, block_matrix_i] = overlaps[j] * (-1) ^ (mi + mj)
            hamiltonians_dict[key_inv][block_matrix_j, block_matrix_i] = hamiltonian[j] * (-1) ^ (mi + mj)
            for direction in 1:3
                positions_dict[cat(dims=1, key_inv, direction)][block_matrix_j, block_matrix_i] = positions[j, direction] * (-1) ^ (mi + mj)
                if (R == [0, 0, 0]) && (block_matrix_i == block_matrix_j) && isnan(site_positions[direction, site_i])
                    site_positions[direction, site_i] = positions[j, direction]
                end
            end
        end
    end
end

if parsed_args["save_overlap"]
    h5open(joinpath(output_dir, "overlaps.h5"), "w") do fid
        for (key, overlap) in overlaps_dict
            write(fid, string(key), permutedims(overlap))
        end
    end
end
h5open(joinpath(output_dir, "hamiltonians.h5"), "w") do fid
    for (key, hamiltonian) in hamiltonians_dict
        write(fid, string(key), permutedims(hamiltonian)) # npz似乎为julia专门做了个转置而h5没有做
    end
end
if parsed_args["save_position"]
    h5open(joinpath(output_dir, "positions.h5"), "w") do fid
        for (key, position) in positions_dict
            write(fid, string(key), permutedims(position)) # npz似乎为julia专门做了个转置而h5没有做
        end
    end
end

open(joinpath(output_dir, "orbital_types.dat"), "w") do f
    writedlm(f, orbital_types)
end
open(joinpath(output_dir, "lat.dat"), "w") do f
    writedlm(f, lat)
end
rlat = 2pi * inv(lat)'
open(joinpath(output_dir, "rlat.dat"), "w") do f
    writedlm(f, rlat)
end
open(joinpath(output_dir, "site_positions.dat"), "w") do f
    writedlm(f, site_positions)
end
R_list = collect(R_list)
open(joinpath(output_dir, "R_list.dat"), "w") do f
    writedlm(f, R_list)
end
info_dict = Dict(
    "isspinful" => spinful
    )
open(joinpath(output_dir, "info.json"), "w") do f
    write(f, json(info_dict, 4))
end
open(joinpath(output_dir, "element.dat"), "w") do f
    writedlm(f, element)
end
