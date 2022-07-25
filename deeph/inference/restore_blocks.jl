using JSON
using LinearAlgebra
using DelimitedFiles
using HDF5
using ArgParse


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "path of block_without_restoration, element.dat, site_positions.dat, orbital_types.dat, and info.json"
            arg_type = String
            default = "./"
        "--output_dir", "-o"
            help = "path of output rh_pred.h5"
            arg_type = String
            default = "./"
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


if isfile(joinpath(parsed_args["input_dir"],"info.json"))
    spinful = JSON.parsefile(joinpath(parsed_args["input_dir"],"info.json"))["isspinful"]
else
    spinful = false
end

spinful = JSON.parsefile(joinpath(parsed_args["input_dir"],"info.json"))["isspinful"]
numbers = readdlm(joinpath(parsed_args["input_dir"], "element.dat"), Int64)
lattice = readdlm(joinpath(parsed_args["input_dir"], "lat.dat"))
inv_lattice = inv(lattice)
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
atom_num_orbital = (x->sum(x .* 2 .+ 1)).(orbital_types)

fid = h5open(joinpath(parsed_args["input_dir"], "block_without_restoration", "block_without_restoration.h5"), "r")
num_model = read(fid["num_model"])
T_pytorch = eltype(fid["output_0"])
if spinful
    T_Hamiltonian = Complex{T_pytorch}
else
    T_Hamiltonian = T_pytorch
end
hoppings_pred = Dict{Array{Int64,1}, Array{T_Hamiltonian, 2}}()
println("Found $num_model models, spinful:$spinful")
edge_attr = read(fid["edge_attr"])
edge_index = read(fid["edge_index"])
for index_model in 0:(num_model-1)
    output = read(fid["output_$index_model"])
    orbital = JSON.parsefile(joinpath(parsed_args["input_dir"], "block_without_restoration", "orbital_$index_model.json"))
    orbital = convert(Vector{Dict{String, Vector{Int}}}, orbital)
    for index in 1:size(edge_index, 1)
        R = Int.(round.(inv_lattice * edge_attr[5:7, index] - inv_lattice * edge_attr[8:10, index]))
        i = edge_index[index, 1] + 1
        j = edge_index[index, 2] + 1
        key_term = cat(R, i, j, dims=1)
        for (index_orbital, orbital_dict) in enumerate(orbital)
            atomic_number_pair = "$(numbers[i]) $(numbers[j])"
            if !(atomic_number_pair ∈ keys(orbital_dict))
                continue
            end
            orbital_i, orbital_j = orbital_dict[atomic_number_pair]
            orbital_i += 1
            orbital_j += 1

            if !(key_term ∈ keys(hoppings_pred))
                if spinful
                    hoppings_pred[key_term] = fill(NaN + NaN * im, 2 * atom_num_orbital[i], 2 * atom_num_orbital[j])
                else
                    hoppings_pred[key_term] = fill(NaN, atom_num_orbital[i], atom_num_orbital[j])
                end
            end
            if spinful
                hoppings_pred[key_term][orbital_i, orbital_j] = output[index_orbital * 8 - 7, index] + output[index_orbital * 8 - 6, index] * im
                hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, atom_num_orbital[j] + orbital_j] = output[index_orbital * 8 - 5, index] + output[index_orbital * 8 - 4, index] * im
                hoppings_pred[key_term][orbital_i, atom_num_orbital[j] + orbital_j] = output[index_orbital * 8 - 3, index] + output[index_orbital * 8 - 2, index] * im
                hoppings_pred[key_term][atom_num_orbital[i] + orbital_i, orbital_j] = output[index_orbital * 8 - 1, index] + output[index_orbital * 8, index] * im
            else
                hoppings_pred[key_term][orbital_i, orbital_j] = output[index_orbital, index]
            end
        end
    end
end
close(fid)

h5open(joinpath(parsed_args["output_dir"], "rh_pred.h5"), "w") do fid
    for (key, rh_pred) in hoppings_pred
        write(fid, string(key), permutedims(rh_pred))
    end
end
