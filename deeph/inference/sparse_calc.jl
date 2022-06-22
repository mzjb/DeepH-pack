using DelimitedFiles, LinearAlgebra, JSON
using HDF5
using ArgParse
using SparseArrays
using Arpack
using JLD


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "path of rlat.dat, orbital_types.dat, site_positions.dat, hamiltonians_pred.h5, and overlaps.h5"
            arg_type = String
            default = "/home/lihe/DeepH/TBG/workflow/6-5/"
        "--output_dir", "-o"
            help = "path of output openmx.Band"
            arg_type = String
            default = "/home/lihe/DeepH/TBG/workflow/6-5/"
        "--config"
            help = "config file in the format of JSON"
            arg_type = String
            default = ""
    end
    return parse_args(s)
end
parsed_args = parse_commandline()


function _create_dict_h5(filename::String)
    fid = h5open(filename, "r")
    T = eltype(fid[keys(fid)[1]])
    d_out = Dict{Array{Int64,1}, Array{T, 2}}()
    for key in keys(fid)
        data = read(fid[key])
        nk = map(x -> parse(Int64, convert(String, x)), split(key[2 : length(key) - 1], ','))
        d_out[nk] = permutedims(data)
    end
    close(fid)
    return d_out
end

const ev2Hartree = 0.036749324533634074
const Bohr2Ang = 0.529177249

function genlist(x)
    return collect(range(x[1], stop = x[2], length = Int64(x[3])))
end

function k_data2num_ks(kdata::AbstractString)
    return parse(Int64,split(kdata)[1])
end

function k_data2kpath(kdata::AbstractString)
    return map(x->parse(Float64,x), split(kdata)[2:7])
end

function std_out_array(a::AbstractArray)
    return string(map(x->string(x," "),a)...)
end

default_dtype = Complex{Float64}

println(parsed_args["config"])
config = JSON.parsefile(parsed_args["config"])
calc_job = config["calc_job"]

if isfile(joinpath(parsed_args["input_dir"],"info.json"))
    spinful = JSON.parsefile(joinpath(parsed_args["input_dir"],"info.json"))["isspinful"]
else
    spinful = false
end

site_positions = readdlm(joinpath(parsed_args["input_dir"], "site_positions.dat"))
nsites = size(site_positions, 2)

orbital_types_f = open(joinpath(parsed_args["input_dir"], "orbital_types.dat"), "r")
site_norbits = zeros(nsites)
orbital_types = Vector{Vector{Int64}}()
for index_site = 1:nsites
    orbital_type = parse.(Int64, split(readline(orbital_types_f)))
    push!(orbital_types, orbital_type)
end
site_norbits = (x->sum(x .* 2 .+ 1)).(orbital_types) * (1 + spinful)
norbits = sum(site_norbits)
site_norbits_cumsum = cumsum(site_norbits)

rlat = readdlm(joinpath(parsed_args["input_dir"], "rlat.dat"))


if isfile(joinpath(parsed_args["input_dir"], "sparse_matrix.jld"))
    @info string("read sparse matrix from ", parsed_args["input_dir"], "/sparse_matrix.jld")
    H_R = load(joinpath(parsed_args["input_dir"], "sparse_matrix.jld"), "H_R")
    S_R = load(joinpath(parsed_args["input_dir"], "sparse_matrix.jld"), "S_R")
else
    if true
        @info "read h5"
        begin_time = time()
        hamiltonians_pred = _create_dict_h5(joinpath(parsed_args["input_dir"], "hamiltonians_pred.h5"))
        overlaps = _create_dict_h5(joinpath(parsed_args["input_dir"], "overlaps.h5"))
        println("Time for reading h5: ", time() - begin_time, "s")

        I_R = Dict{Vector{Int64}, Vector{Int64}}()
        J_R = Dict{Vector{Int64}, Vector{Int64}}()
        H_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
        S_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()

        @info "construct sparse matrix in the format of COO"
        begin_time = time()
        for key in collect(keys(hamiltonians_pred))
            hamiltonian_pred = hamiltonians_pred[key]
            overlap = overlaps[key]
            if spinful
                overlap = vcat(hcat(overlap,zeros(size(overlap))),hcat(zeros(size(overlap)),overlap)) # the readout overlap matrix only contains the upper-left block # TODO maybe drop the zeros?
            end
            R = key[1:3]; atom_i=key[4]; atom_j=key[5]

            @assert (site_norbits[atom_i], site_norbits[atom_j]) == size(hamiltonian_pred)
            @assert (site_norbits[atom_i], site_norbits[atom_j]) == size(overlap)
            if !(R ∈ keys(I_R))
                I_R[R] = Vector{Int64}()
                J_R[R] = Vector{Int64}()
                H_V_R[R] = Vector{default_dtype}()
                S_V_R[R] = Vector{default_dtype}()
            end
            for block_matrix_i in 1:site_norbits[atom_i]
                for block_matrix_j in 1:site_norbits[atom_j]
                    coo_i = site_norbits_cumsum[atom_i] - site_norbits[atom_i] + block_matrix_i
                    coo_j = site_norbits_cumsum[atom_j] - site_norbits[atom_j] + block_matrix_j
                    push!(I_R[R], coo_i)
                    push!(J_R[R], coo_j)
                    push!(H_V_R[R], hamiltonian_pred[block_matrix_i, block_matrix_j])
                    push!(S_V_R[R], overlap[block_matrix_i, block_matrix_j])
                end
            end
        end
        println("Time for constructing sparse matrix in the format of COO: ", time() - begin_time, "s")
    else
        @info "construct sparse matrix in the format of COO from h5"
        begin_time = time()
        I_R = Dict{Vector{Int64}, Vector{Int64}}()
        J_R = Dict{Vector{Int64}, Vector{Int64}}()
        H_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
        S_V_R = Dict{Vector{Int64}, Vector{default_dtype}}()
        fid_H = h5open(joinpath(parsed_args["input_dir"], "hamiltonians_pred.h5"), "r")
        fid_S = h5open(joinpath(parsed_args["input_dir"], "overlaps.h5"), "r")

        for key in keys(fid_H)
            hamiltonian_pred = permutedims(read(fid_H[key]))
            overlap = permutedims(read(fid_S[key]))
            Rij = map(x -> parse(Int64, convert(String, x)), split(key[2 : length(key) - 1], ','))
            R = Rij[1:3]; atom_i=Rij[4]; atom_j=Rij[5]

            @assert (site_norbits[atom_i], site_norbits[atom_j]) == size(hamiltonian_pred)
            @assert (site_norbits[atom_i], site_norbits[atom_j]) == size(overlap)
            if !(R ∈ keys(I_R))
                I_R[R] = Vector{Int64}()
                J_R[R] = Vector{Int64}()
                H_V_R[R] = Vector{default_dtype}()
                S_V_R[R] = Vector{default_dtype}()
            end
            for block_matrix_i in 1:site_norbits[atom_i]
                for block_matrix_j in 1:site_norbits[atom_j]
                    coo_i = site_norbits_cumsum[atom_i] - site_norbits[atom_i] + block_matrix_i
                    coo_j = site_norbits_cumsum[atom_j] - site_norbits[atom_j] + block_matrix_j
                    push!(I_R[R], coo_i)
                    push!(J_R[R], coo_j)
                    push!(H_V_R[R], hamiltonian_pred[block_matrix_i, block_matrix_j])
                    push!(S_V_R[R], overlap[block_matrix_i, block_matrix_j])
                end
            end
        end
        println("Time for reading h5 in the format of COO: ", time() - begin_time, "s")
    end

    @info "convert sparse matrix to the format of CSC"
    begin_time = time()
    H_R = Dict{Vector{Int64}, SparseMatrixCSC{default_dtype, Int64}}()
    S_R = Dict{Vector{Int64}, SparseMatrixCSC{default_dtype, Int64}}()

    for R in keys(I_R)
        H_R[R] = sparse(I_R[R], J_R[R], H_V_R[R], norbits, norbits)
        S_R[R] = sparse(I_R[R], J_R[R], S_V_R[R], norbits, norbits)
    end
    println("Time for converting to the format of CSC: ", time() - begin_time, "s")

    save(joinpath(parsed_args["input_dir"], "sparse_matrix.jld"), "H_R", H_R, "S_R", S_R)
end

if calc_job == "band"
    which_k = config["which_k"] # which k point to calculate, start counting from 1, 0 for all k points
    fermi_level = config["fermi_level"]
    lowest_band = config["lowest_band"]
    max_iter = config["max_iter"]
    num_band = config["num_band"]
    k_data = config["k_data"]

    @info "calculate bands"
    num_ks = k_data2num_ks.(k_data)
    kpaths = k_data2kpath.(k_data)

    egvals = zeros(Float64, num_band, sum(num_ks)[1])

    begin_time = time()
    idx_k = 1
    for i = 1:size(kpaths, 1)
        kpath = kpaths[i]
        pnkpts = num_ks[i]
        kxs = LinRange(kpath[1], kpath[4], pnkpts)
        kys = LinRange(kpath[2], kpath[5], pnkpts)
        kzs = LinRange(kpath[3], kpath[6], pnkpts)
        for (kx, ky, kz) in zip(kxs, kys, kzs)
            global idx_k
            if which_k == 0 || which_k == idx_k
                H_k = spzeros(default_dtype, norbits, norbits)
                S_k = spzeros(default_dtype, norbits, norbits)
                for R in keys(H_R)
                    H_k += H_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
                    S_k += S_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
                end
                flush(stdout)
                egval = real(eigs(H_k, S_k, nev=num_band, sigma=(fermi_level + lowest_band), which=:LR, ritzvec=false, maxiter=max_iter)[1])
                if which_k == 0
                    println(egval .- fermi_level)
                else
                    open(joinpath(parsed_args["output_dir"], "kpoint.dat"), "w") do f
                        writedlm(f, [kx, ky, kz])
                    end
                    open(joinpath(parsed_args["output_dir"], "egval.dat"), "w") do f
                        writedlm(f, egval)
                    end
                end
                egvals[:, idx_k] = egval
                println("Time for solving No.$idx_k eigenvalues at k = ", [kx, ky, kz], ": ", time() - begin_time, "s")
            end
            idx_k += 1
        end
    end

    # output in openmx band format
    f = open(joinpath(parsed_args["output_dir"], "openmx.Band"),"w")
    println(f, num_band, " ", 0, " ", ev2Hartree * fermi_level)
    openmx_rlat = reshape((rlat .* Bohr2Ang), 1, :)
    println(f, std_out_array(openmx_rlat))
    println(f, length(k_data))
    for line in k_data
        println(f,line)
    end
    idx_k = 1
    for i = 1:size(kpaths, 1)
        pnkpts = num_ks[i]
        kstart = kpaths[i][1:3]
        kend = kpaths[i][4:6]
        k_list = zeros(Float64,pnkpts,3)
        for alpha = 1:3
            k_list[:,alpha] = genlist([kstart[alpha],kend[alpha],pnkpts])
        end
        for j = 1:pnkpts
            global idx_k
            kvec = k_list[j,:]
            println(f, num_band, " ", std_out_array(kvec))
            println(f, std_out_array(ev2Hartree * egvals[:, idx_k]))
            idx_k += 1
        end
    end
    close(f)
end
