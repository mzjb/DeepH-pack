using StaticArrays
using LinearAlgebra
using HDF5
using JSON
using DelimitedFiles
using Statistics
using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = ""
            arg_type = String
            default = "./"
        "--output_dir", "-o"
            help = ""
            arg_type = String
            default = "./output"
        "--if_DM", "-d"
            help = ""
            arg_type = Bool
            default = false
       "--save_overlap", "-s"
            help = ""
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end
parsed_args = parse_commandline()

# @info string("get data from: ", parsed_args["input_dir"])
periodic_table = JSON.parsefile(joinpath(@__DIR__, "periodic_table.json"))

#=
struct Site_list
    R::Array{StaticArrays.SArray{Tuple{3},Int16,1,3},1}
    site::Array{Int64,1}
    pos::Array{Float64,2}
end

function _cal_neighbour_site(e_ij::Array{Float64,2},Rcut::Float64)
    r_ij = sum(dims=1,e_ij.^2)[1,:]
    p = sortperm(r_ij)
    j_cut = searchsorted(r_ij[p],Rcut^2).stop
    return p[1:j_cut]
end

function cal_neighbour_site(site_positions::Matrix{<:Real}, lat::Matrix{<:Real}, R_list::Union{Vector{SVector{3, Int64}}, Vector{Vector{Int64}}}, nsites::Int64, Rcut::Float64)
    # writed by lihe
    neighbour_site = Array{Site_list,1}([])
    # R_list = collect(keys(tm.hoppings))
    pos_R_list = map(R -> collect(lat * R), R_list)
    pos_j_list = cat(dims=2, map(pos_R -> pos_R .+ site_positions, pos_R_list)...)
    for site_i in 1:nsites
        pos_i = site_positions[:, site_i]
        e_ij = pos_j_list .- pos_i
        p = _cal_neighbour_site(e_ij, Rcut)
        R_ordered = R_list[map(x -> div(x + (nsites - 1), nsites),p)]
        site_ordered = map(x -> mod(x - 1, nsites) + 1,p)
        pos_ordered = e_ij[:,p]
        @assert pos_ordered[:,1] ≈ [0,0,0]
        push!(neighbour_site, Site_list(R_ordered, site_ordered, pos_ordered))
    end
    return neighbour_site
end

function _get_local_coordinate(e_ij::Array{Float64,1},site_list_i::Site_list)
    if e_ij != [0,0,0]
        r1 = e_ij
    else
        r1 = site_list_i.pos[:,2]
    end
    nsites_i = length(site_list_i.R)
    r2 = [0,0,0]
    for j in 1:nsites_i
        r2 = site_list_i.pos[:,j]
        if norm(cross(r1,r2)) != 0
            break
        end
        if j == nsites_i
            for k in 1:3
                r2 = [[1,0,0],[0,1,0],[0,0,1]][k]
                if norm(cross(r1,r2)) != 0
                    break
                end
            end
        end
    end
    if r2 == [0,0,0]
        error("there is no linear independent chemical bond in the Rcut range, this may be caused by a too small  Rcut or the structure  is1D")
    end
    local_coordinate = zeros(Float64,(3,3))
    local_coordinate[:,1] = r1/norm(r1)

    local_coordinate[:,2] = cross(r1,r2)/norm(cross(r1,r2))
    local_coordinate[:,3] = cross(local_coordinate[:,1],local_coordinate[:,2])
    return local_coordinate
end

function get_local_coordinates(site_positions::Matrix{<:Real}, lat::Matrix{<:Real}, R_list::Vector{SVector{3, Int64}}, Rcut::Float64)::Dict{Array{Int64,1},Array{Float64,2}}
    nsites = size(site_positions, 2)
    neighbour_site = cal_neighbour_site(site_positions, lat, R_list, nsites, Rcut)
    local_coordinates = Dict{Array{Int64,1},Array{Float64,2}}()
    for site_i = 1:nsites
        site_list_i = neighbour_site[site_i]
        nsites_i = length(site_list_i.R)
        for j = 1:nsites_i
            R = site_list_i.R[j]; site_j = site_list_i.site[j]; e_ij = site_list_i.pos[:,j]
            local_coordinate = _get_local_coordinate(e_ij, site_list_i)
            local_coordinates[cat(dims=1, R, site_i, site_j)] = local_coordinate
        end
    end
    return local_coordinates
end
=#

# The function parse_openmx below is come from "https://github.com/HopTB/HopTB.jl"
function parse_openmx(filepath::String; return_DM::Bool = false)
    # define some helper functions for mixed structure of OpenMX binary data file.
    function multiread(::Type{T}, f, size)::Vector{T} where T
        ret = Vector{T}(undef, size)
        read!(f, ret);ret
    end
    multiread(f, size) = multiread(Int32, f, size)

    function read_mixed_matrix(::Type{T}, f, dims::Vector{<:Integer}) where T
        ret::Vector{Vector{T}} = []
        for i = dims; t = Vector{T}(undef, i);read!(f, t);push!(ret, t); end; ret
    end

    function read_matrix_in_mixed_matrix(::Type{T}, f, spins, atomnum, FNAN, natn, Total_NumOrbs) where T
        ret = Vector{Vector{Vector{Matrix{T}}}}(undef, spins)
        for spin = 1:spins;t_spin = Vector{Vector{Matrix{T}}}(undef, atomnum)
            for ai = 1:atomnum;t_ai = Vector{Matrix{T}}(undef, FNAN[ai])
                for aj_inner = 1:FNAN[ai]
                    t = Matrix{T}(undef, Total_NumOrbs[natn[ai][aj_inner]], Total_NumOrbs[ai])
                    read!(f, t);t_ai[aj_inner] = t
                end;t_spin[ai] = t_ai
            end;ret[spin] = t_spin
        end;return ret
    end
    read_matrix_in_mixed_matrix(f, spins, atomnum, FNAN, natn, Total_NumOrbs) = read_matrix_in_mixed_matrix(Float64, f, spins, atomnum, FNAN, natn, Total_NumOrbs)

    read_3d_vecs(::Type{T}, f, num) where T = reshape(multiread(T, f, 4 * num), 4, Int(num))[2:4,:]
    read_3d_vecs(f, num) = read_3d_vecs(Float64, f, num)
    # End of helper functions

    bound_multiread(T, size) = multiread(T, f, size)
    bound_multiread(size) = multiread(f, size)
    bound_read_mixed_matrix() = read_mixed_matrix(Int32, f, FNAN)
    bound_read_matrix_in_mixed_matrix(spins) = read_matrix_in_mixed_matrix(f, spins, atomnum, FNAN, natn, Total_NumOrbs)
    bound_read_3d_vecs(num) = read_3d_vecs(f, num)
    bound_read_3d_vecs(::Type{T}, num) where T = read_3d_vecs(T, f, num)
    # End of bound helper functions

    f = open(filepath)
    atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell, order_max = bound_multiread(7)
    @assert (SpinP_switch >> 2) == 3 "DeepH-pack only supports OpenMX v3.9. Please check your OpenMX version"
    SpinP_switch &= 0x03

    atv, atv_ijk = bound_read_3d_vecs.([Float64,Int32], TCpyCell + 1)

    Total_NumOrbs, FNAN = bound_multiread.([atomnum,atomnum])
    FNAN .+= 1
    natn = bound_read_mixed_matrix()
    ncn = ((x)->x .+ 1).(bound_read_mixed_matrix()) # These is to fix that atv and atv_ijk starts from 0 in original C code.

    tv, rtv, Gxyz = bound_read_3d_vecs.([3,3,atomnum])

    Hk = bound_read_matrix_in_mixed_matrix(SpinP_switch + 1)
    iHk = SpinP_switch == 3 ? bound_read_matrix_in_mixed_matrix(3) : nothing
    OLP = bound_read_matrix_in_mixed_matrix(1)[1]
    OLP_r = []
    for dir in 1:3, order in 1:order_max
        t = bound_read_matrix_in_mixed_matrix(1)[1]
        if order == 1 push!(OLP_r, t) end
    end
    OLP_p = bound_read_matrix_in_mixed_matrix(3)
    DM = bound_read_matrix_in_mixed_matrix(SpinP_switch + 1)
    iDM = bound_read_matrix_in_mixed_matrix(2)
    solver = bound_multiread(1)[1]
    chem_p, E_temp = bound_multiread(Float64, 2)
    dipole_moment_core, dipole_moment_background = bound_multiread.(Float64, [3,3])
    Valence_Electrons, Total_SpinS = bound_multiread(Float64, 2)
    dummy_blocks = bound_multiread(1)[1]
    for i in 1:dummy_blocks
        bound_multiread(UInt8, 256)
    end

    # we suppose that the original output file(.out) was appended to the end of the scfout file.
    function strip1(s::Vector{UInt8})
        startpos = 0
        for i = 1:length(s) + 1
            if i > length(s) || s[i] & 0x80 != 0 || !isspace(Char(s[i] & 0x7f))
                startpos = i
                break
            end
        end
        return s[startpos:end]
    end
    function startswith1(s::Vector{UInt8}, prefix::Vector{UInt8})
        return length(s) >= length(prefix) && s[1:length(prefix)] == prefix
    end
    function isnum(s::Char)
        if s >= '1' && s <= '9'
            return true
        else
            return false
        end
    end

    function isorb(s::Char)
        if s in ['s','p','d','f']
            return true
        else
            return false
        end
    end

    function orbital_types_str2num(str::AbstractString)
        orbs = split(str, isnum, keepempty = false)
        nums = map(x->parse(Int, x), split(str, isorb, keepempty = false))
        orb2l = Dict("s" => 0, "p" => 1, "d" => 2, "f" => 3)
        @assert length(orbs) == length(nums)
        orbital_types = Array{Int64,1}(undef, 0)
        for (orb, num) in zip(orbs, nums)
            for i = 1:num
                push!(orbital_types, orb2l[orb])
            end
        end
        return orbital_types
    end

    function find_target_line(target_line::String)
        target_line_UInt8 = Vector{UInt8}(target_line)
        while !startswith1(strip1(Vector{UInt8}(readline(f))), target_line_UInt8)
            if eof(f)
                error(string(target_line, "not found. Please check if the .out file was appended to the end of .scfout file!"))
            end
        end
    end

#     @info """get orbital setting of element:orbital_types_element::Dict{String,Array{Int64,1}} "element" => orbital_types"""
    find_target_line("<Definition.of.Atomic.Species")
    orbital_types_element = Dict{String,Array{Int64,1}}([])
    while true
        str = readline(f)
        if str == "Definition.of.Atomic.Species>"
            break
        end
        element = split(str)[1]
        orbital_types_str = split(split(str)[2], "-")[2]
        orbital_types_element[element] = orbital_types_str2num(orbital_types_str)
    end

    
#     @info "get Chemical potential (Hartree)"
    find_target_line("(see also PRB 72, 045121(2005) for the energy contributions)")
    readline(f)
    readline(f)
    readline(f)
    str = split(readline(f))
    @assert "Chemical" == str[1]
    @assert "potential" == str[2]
    @assert "(Hartree)" == str[3]
    ev2Hartree = 0.036749324533634074
    fermi_level = parse(Float64, str[length(str)])/ev2Hartree

    # @info "get Chemical potential (Hartree)"
    # find_target_line("Eigenvalues (Hartree)")
    # for i = 1:2;@assert readline(f) == "***********************************************************";end
    # readline(f)
    # str = split(readline(f))
    # ev2Hartree = 0.036749324533634074
    # fermi_level = parse(Float64, str[length(str)])/ev2Hartree
    

#     @info "get Fractional coordinates & orbital types:"
    find_target_line("Fractional coordinates of the final structure")
    target_line = Vector{UInt8}("Fractional coordinates of the final structure")
    for i = 1:2;@assert readline(f) == "***********************************************************";end
    @assert readline(f) == ""
    orbital_types = Array{Array{Int64,1},1}(undef, 0) #orbital_types
    element = Array{Int64,1}(undef, 0) #orbital_types
    atom_frac_pos = zeros(3, atomnum) #Fractional coordinates
    for i = 1:atomnum
        str = readline(f)
        element_str = split(str)[2]
        push!(orbital_types, orbital_types_element[element_str])
        m = match(r"^\s*\d+\s+\w+\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)\s+([0-9+-.Ee]+)", str)
        push!(element, periodic_table[element_str]["Atomic no"])
        atom_frac_pos[:,i] = ((x)->parse(Float64, x)).(m.captures)
    end
    atom_pos = tv * atom_frac_pos
    close(f)

    # use the atom_pos to fix
    # TODO: Persuade wangc to accept the following code, which seems hopeless and meaningless.
    """
    for axis = 1:3
        ((x2, y2, z)->((x, y)->x .+= z * y).(x2, y2)).(OLP_r[axis], OLP, atom_pos[axis,:])
    end
    """
    for axis in 1:3,i in 1:atomnum, j in 1:FNAN[i]
        OLP_r[axis][i][j] .+= atom_pos[axis,i] * OLP[i][j]
    end

    # fix type mismatch
    atv_ijk = Matrix{Int64}(atv_ijk)

    if return_DM
        return element, atomnum, SpinP_switch, atv, atv_ijk, Total_NumOrbs, FNAN, natn, ncn, tv, Hk, iHk, OLP, OLP_r, orbital_types, fermi_level, atom_pos, DM
    else
        return element, atomnum, SpinP_switch, atv, atv_ijk, Total_NumOrbs, FNAN, natn, ncn, tv, Hk, iHk, OLP, OLP_r, orbital_types, fermi_level, atom_pos, nothing
    end
end

function get_data(filepath_scfout::String, Rcut::Float64; if_DM::Bool = false)
    element, nsites, SpinP_switch, atv, atv_ijk, Total_NumOrbs, FNAN, natn, ncn, lat, Hk, iHk, OLP, OLP_r, orbital_types, fermi_level, site_positions, DM = parse_openmx(filepath_scfout; return_DM=if_DM)

    for t in [Hk, iHk]
        if t != nothing
            ((x)->((y)->((z)->z .*= 27.2113845).(y)).(x)).(t) # Hartree to eV
        end
    end
    site_positions .*= 0.529177249 # Bohr to Ang
    lat .*= 0.529177249 # Bohr to Ang

    # get R_list
    R_list = Set{Vector{Int64}}()
    for atom_i in 1:nsites, index_nn_i in 1:FNAN[atom_i]
        atom_j = natn[atom_i][index_nn_i]
        R = atv_ijk[:, ncn[atom_i][index_nn_i]]
        push!(R_list, SVector{3, Int64}(R))
    end
    R_list = collect(R_list)

    # get neighbour_site
    nsites = size(site_positions, 2)
    # neighbour_site = cal_neighbour_site(site_positions, lat, R_list, nsites, Rcut)
    # local_coordinates = Dict{Array{Int64, 1}, Array{Float64, 2}}()

    # process hamiltonian
    norbits = sum(Total_NumOrbs)
    overlaps = Dict{Array{Int64, 1}, Array{Float64, 2}}()
    if SpinP_switch == 0
        spinful = false
        hamiltonians = Dict{Array{Int64, 1}, Array{Float64, 2}}()
        if if_DM
            density_matrixs = Dict{Array{Int64, 1}, Array{Float64, 2}}()
        else
            density_matrixs = nothing
        end
    elseif SpinP_switch == 1
        error("Collinear spin is not supported currently")
    elseif SpinP_switch == 3
        @assert if_DM == false
        density_matrixs = nothing
        spinful = true
        for i in 1:length(Hk[4]),j in 1:length(Hk[4][i])
            Hk[4][i][j] += iHk[3][i][j]
            iHk[3][i][j] = -Hk[4][i][j]
        end
        hamiltonians = Dict{Array{Int64, 1}, Array{Complex{Float64}, 2}}()
    else
        error("SpinP_switch is $SpinP_switch, rather than valid values 0, 1 or 3")
    end

    for site_i in 1:nsites, index_nn_i in 1:FNAN[site_i]
        site_j = natn[site_i][index_nn_i]
        R = atv_ijk[:, ncn[site_i][index_nn_i]]
        e_ij = lat * R + site_positions[:, site_j] - site_positions[:, site_i]
        # if norm(e_ij) > Rcut
            # continue
        # end
        key = cat(dims=1, R, site_i, site_j)
        # site_list_i = neighbour_site[site_i]
        # local_coordinate = _get_local_coordinate(e_ij, site_list_i)
        # local_coordinates[key] = local_coordinate
        
        overlap = permutedims(OLP[site_i][index_nn_i])
        overlaps[key] = overlap
        if SpinP_switch == 0
            hamiltonian = permutedims(Hk[1][site_i][index_nn_i])
            hamiltonians[key] = hamiltonian
            if if_DM
                density_matrix = permutedims(DM[1][site_i][index_nn_i])
                density_matrixs[key] = density_matrix
            end
        elseif SpinP_switch == 1
            error("Collinear spin is not supported currently")
        elseif SpinP_switch == 3
            key_inv = cat(dims=1, -R, site_j, site_i)

            len_i_wo_spin = Total_NumOrbs[site_i]
            len_j_wo_spin = Total_NumOrbs[site_j]
    
            if !(key in keys(hamiltonians))
                @assert !(key_inv in keys(hamiltonians))
                hamiltonians[key] = zeros(Complex{Float64}, len_i_wo_spin * 2, len_j_wo_spin * 2)
                hamiltonians[key_inv] = zeros(Complex{Float64}, len_j_wo_spin * 2, len_i_wo_spin * 2)
            end
            for spini in 0:1,spinj in spini:1
                Hk_real, Hk_imag = spini == 0 ? spinj == 0 ? (Hk[1][site_i][index_nn_i], iHk[1][site_i][index_nn_i]) : (Hk[3][site_i][index_nn_i], Hk[4][site_i][index_nn_i]) : spinj == 0 ? (Hk[3][site_i][index_nn_i], iHk[3][site_i][index_nn_i]) : (Hk[2][site_i][index_nn_i], iHk[2][site_i][index_nn_i])
                hamiltonians[key][spini * len_i_wo_spin + 1 : (spini + 1) * len_i_wo_spin, spinj * len_j_wo_spin + 1 : (spinj + 1) * len_j_wo_spin] = permutedims(Hk_real) + im * permutedims(Hk_imag)
                if spini == 0 && spinj == 1
                    hamiltonians[key_inv][1 * len_j_wo_spin + 1 : (1 + 1) * len_j_wo_spin, 0 * len_i_wo_spin + 1 : (0 + 1) * len_i_wo_spin] = (permutedims(Hk_real) + im * permutedims(Hk_imag))'
                end
            end
        else
            error("SpinP_switch is $SpinP_switch, rather than valid values 0, 1 or 3")
        end
    end

    return element, overlaps, density_matrixs, hamiltonians, fermi_level, orbital_types, lat, site_positions, spinful, R_list
end

parsed_args["input_dir"] = abspath(parsed_args["input_dir"])
mkpath(parsed_args["output_dir"])
cd(parsed_args["output_dir"])

element, overlaps, density_matrixs, hamiltonians, fermi_level, orbital_types, lat, site_positions, spinful, R_list = get_data(joinpath(parsed_args["input_dir"], "openmx.scfout"), -1.0; if_DM=parsed_args["if_DM"])

if parsed_args["if_DM"]
    h5open("density_matrixs.h5", "w") do fid
        for (key, density_matrix) in density_matrixs
            write(fid, string(key), permutedims(density_matrix))
        end
    end
end
if parsed_args["save_overlap"]
    h5open("overlaps.h5", "w") do fid
        for (key, overlap) in overlaps
            write(fid, string(key), permutedims(overlap))
        end
    end
end
h5open("hamiltonians.h5", "w") do fid
    for (key, hamiltonian) in hamiltonians
        write(fid, string(key), permutedims(hamiltonian))
    end
end

info_dict = Dict(
    "fermi_level" => fermi_level,
    "isspinful" => spinful
    )
open("info.json", "w") do f
    write(f, json(info_dict, 4))
end
open("site_positions.dat", "w") do f
    writedlm(f, site_positions)
end
open("R_list.dat", "w") do f
    writedlm(f, R_list)
end
open("lat.dat", "w") do f
    writedlm(f, lat)
end
rlat = 2pi * inv(lat)'
open("rlat.dat", "w") do f
    writedlm(f, rlat)
end
open("orbital_types.dat", "w") do f
    writedlm(f, orbital_types)
end
open("element.dat", "w") do f
    writedlm(f, element)
end
