using DelimitedFiles, LinearAlgebra, JSON
using HDF5
using ArgParse
using SparseArrays
using Pardiso, Arpack, LinearMaps
using JLD
# BLAS.set_num_threads(1)

const ev2Hartree = 0.036749324533634074
const Bohr2Ang = 0.529177249
const default_dtype = Complex{Float64}


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "path of rlat.dat, orbital_types.dat, site_positions.dat, hamiltonians_pred.h5, and overlaps.h5"
            arg_type = String
            default = "./"
        "--output_dir", "-o"
            help = "path of output openmx.Band"
            arg_type = String
            default = "./"
        "--config"
            help = "config file in the format of JSON"
            arg_type = String
        "--ill_project"
            help = "projects out the eigenvectors of the overlap matrix that correspond to eigenvalues smaller than ill_threshold"
            arg_type = Bool
            default = true
        "--ill_threshold"
            help = "threshold for ill_project"
            arg_type = Float64
            default = 5e-4
    end
    return parse_args(s)
end


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


# The function construct_linear_map below is come from https://discourse.julialang.org/t/smallest-magnitude-eigenvalues-of-the-generalized-eigenvalue-equation-for-a-large-sparse-matrix/75485/11
function construct_linear_map(H, S)
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    H_pardiso = get_matrix(ps, H, :N)
    b = rand(ComplexF64, size(H, 1))
    set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)
    return (
        LinearMap{ComplexF64}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating=true
        ),
        ps
    )
end


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


function constructmeshkpts(nkmesh::Vector{Int64}; offset::Vector{Float64}=[0.0, 0.0, 0.0],
    k1::Vector{Float64}=[0.0, 0.0, 0.0], k2::Vector{Float64}=[1.0, 1.0, 1.0])
    length(nkmesh) == 3 || throw(ArgumentError("nkmesh in wrong size."))
    nkpts = prod(nkmesh)
    kpts = zeros(3, nkpts)
    ik = 1
    for ikx in 1:nkmesh[1], iky in 1:nkmesh[2], ikz in 1:nkmesh[3]
        kpts[:, ik] = [
            (ikx-1)/nkmesh[1]*(k2[1]-k1[1])+k1[1],
            (iky-1)/nkmesh[2]*(k2[2]-k1[2])+k1[2],
            (ikz-1)/nkmesh[3]*(k2[3]-k1[3])+k1[3]
        ]
        ik += 1
    end
    return kpts.+offset
end


function main()
    parsed_args = parse_commandline()

    println(parsed_args["config"])
    config = JSON.parsefile(parsed_args["config"])
    calc_job = config["calc_job"]
    ill_project = parsed_args["ill_project"]
    ill_threshold = parsed_args["ill_threshold"]

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
            if (key ∈ keys(overlaps))
                overlap = overlaps[key]
            else
                # continue
                overlap = zero(hamiltonian_pred)
            end
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
                if which_k == 0 || which_k == idx_k
                    H_k = spzeros(default_dtype, norbits, norbits)
                    S_k = spzeros(default_dtype, norbits, norbits)
                    for R in keys(H_R)
                        H_k += H_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
                        S_k += S_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
                    end
                    S_k = (S_k + S_k') / 2
                    H_k = (H_k + H_k') / 2
                    if ill_project
                        lm, ps = construct_linear_map(Hermitian(H_k) - (fermi_level) * Hermitian(S_k), Hermitian(S_k))
                        println("Time for No.$idx_k matrix factorization: ", time() - begin_time, "s")
                        egval_sub_inv, egvec_sub = eigs(lm, nev=num_band, which=:LM, ritzvec=true, maxiter=max_iter)
                        set_phase!(ps, Pardiso.RELEASE_ALL)
                        pardiso(ps)
                        egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)

                        # orthogonalize the eigenvectors
                        egvec_sub_qr = qr(egvec_sub)
                        egvec_sub = convert(Matrix{default_dtype}, egvec_sub_qr.Q)

                        S_k_sub = egvec_sub' * S_k * egvec_sub
                        (egval_S, egvec_S) = eigen(Hermitian(S_k_sub))
                        # egvec_S: shape (num_basis, num_bands)
                        project_index = abs.(egval_S) .> ill_threshold
                        if sum(project_index) != length(project_index)
                            H_k_sub = egvec_sub' * H_k * egvec_sub
                            egvec_S = egvec_S[:, project_index]
                            @warn "ill-conditioned eigenvalues detected, projected out $(length(project_index) - sum(project_index)) eigenvalues"
                            H_k_sub = egvec_S' * H_k_sub * egvec_S
                            S_k_sub = egvec_S' * S_k_sub * egvec_S
                            (egval, egvec) = eigen(Hermitian(H_k_sub), Hermitian(S_k_sub))
                            egval = vcat(egval, fill(1e4, length(project_index) - sum(project_index)))
                            egvec = egvec_S * egvec
                            egvec = egvec_sub * egvec
                        else
                            egval = egval_sub
                        end
                    else
                        lm, ps = construct_linear_map(Hermitian(H_k) - (fermi_level) * Hermitian(S_k), Hermitian(S_k))
                        println("Time for No.$idx_k matrix factorization: ", time() - begin_time, "s")
                        egval_inv, egvec = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=max_iter)
                        set_phase!(ps, Pardiso.RELEASE_ALL)
                        pardiso(ps)
                        egval = real(1 ./ egval_inv) .+ (fermi_level)
                        # egval = real(eigs(H_k, S_k, nev=num_band, sigma=(fermi_level + lowest_band), which=:LR, ritzvec=false, maxiter=max_iter)[1])
                    end
                    egvals[:, idx_k] = egval
                    if which_k == 0
                        # println(egval .- fermi_level)
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
                kvec = k_list[j,:]
                println(f, num_band, " ", std_out_array(kvec))
                println(f, std_out_array(ev2Hartree * egvals[:, idx_k]))
                idx_k += 1
            end
        end
        close(f)
    elseif calc_job == "dos"
        fermi_level = config["fermi_level"]
        max_iter = config["max_iter"]
        num_band = config["num_band"]
        nkmesh = convert(Array{Int64,1}, config["kmesh"])
        ks = constructmeshkpts(nkmesh)
        nks = size(ks, 2)

        egvals = zeros(Float64, num_band, nks)
        begin_time = time()
        for idx_k in 1:nks
            kx, ky, kz = ks[:, idx_k]

            H_k = spzeros(default_dtype, norbits, norbits)
            S_k = spzeros(default_dtype, norbits, norbits)
            for R in keys(H_R)
                H_k += H_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
                S_k += S_R[R] * exp(im*2π*([kx, ky, kz]⋅R))
            end
            S_k = (S_k + S_k') / 2
            H_k = (H_k + H_k') / 2
            if ill_project
                lm, ps = construct_linear_map(Hermitian(H_k) - (fermi_level) * Hermitian(S_k), Hermitian(S_k))
                println("Time for No.$idx_k matrix factorization: ", time() - begin_time, "s")
                egval_sub_inv, egvec_sub = eigs(lm, nev=num_band, which=:LM, ritzvec=true, maxiter=max_iter)
                set_phase!(ps, Pardiso.RELEASE_ALL)
                pardiso(ps)
                egval_sub = real(1 ./ egval_sub_inv) .+ (fermi_level)

                # orthogonalize the eigenvectors
                egvec_sub_qr = qr(egvec_sub)
                egvec_sub = convert(Matrix{default_dtype}, egvec_sub_qr.Q)

                S_k_sub = egvec_sub' * S_k * egvec_sub
                (egval_S, egvec_S) = eigen(Hermitian(S_k_sub))
                # egvec_S: shape (num_basis, num_bands)
                project_index = abs.(egval_S) .> ill_threshold
                if sum(project_index) != length(project_index)
                    H_k_sub = egvec_sub' * H_k * egvec_sub
                    egvec_S = egvec_S[:, project_index]
                    @warn "ill-conditioned eigenvalues detected, projected out $(length(project_index) - sum(project_index)) eigenvalues"
                    H_k_sub = egvec_S' * H_k_sub * egvec_S
                    S_k_sub = egvec_S' * S_k_sub * egvec_S
                    (egval, egvec) = eigen(Hermitian(H_k_sub), Hermitian(S_k_sub))
                    egval = vcat(egval, fill(1e4, length(project_index) - sum(project_index)))
                    egvec = egvec_S * egvec
                    egvec = egvec_sub * egvec
                else
                    egval = egval_sub
                end
            else
                lm, ps = construct_linear_map(Hermitian(H_k) - (fermi_level) * Hermitian(S_k), Hermitian(S_k))
                println("Time for No.$idx_k matrix factorization: ", time() - begin_time, "s")
                egval_inv, egvec = eigs(lm, nev=num_band, which=:LM, ritzvec=false, maxiter=max_iter)
                set_phase!(ps, Pardiso.RELEASE_ALL)
                pardiso(ps)
                egval = real(1 ./ egval_inv) .+ (fermi_level)
                # egval = real(eigs(H_k, S_k, nev=num_band, sigma=(fermi_level + lowest_band), which=:LR, ritzvec=false, maxiter=max_iter)[1])
            end
            egvals[:, idx_k] = egval
            if which_k == 0
                # println(egval .- fermi_level)
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

        open(joinpath(parsed_args["output_dir"], "egvals.dat"), "w") do f
            writedlm(f, egvals)
        end

        ϵ = config["epsilon"]
        ωs = genlist(config["omegas"])
        nωs = length(ωs)
        dos = zeros(nωs)
        factor = 1/((2π)^3*ϵ*√π)
        for idx_k in 1:nks, idx_band in 1:num_band, (idx_ω, ω) in enumerate(ωs)
            dos[idx_ω] += exp(-(egvals[idx_band, idx_k] - ω - fermi_level) ^ 2 / ϵ ^ 2) * factor
        end
        open(joinpath(parsed_args["output_dir"], "dos.dat"), "w") do f
            writedlm(f, [ωs dos])
        end
    end
end


main()
