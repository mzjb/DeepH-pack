using DelimitedFiles, LinearAlgebra
using HDF5
using ArgParse
using StaticArrays


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input_dir", "-i"
            help = "path of site_positions.dat, lat.dat, element.dat, and R_list.dat (overlaps.h5)"
            arg_type = String
            default = "./"
        "--output_dir", "-o"
            help = "path of output rc.h5"
            arg_type = String
            default = "./"
        "--radius", "-r"
            help = "cutoff radius"
            arg_type = Float64
            default = 8.0
        "--create_from_DFT"
            help = "retain edges by DFT overlaps neighbour"
            arg_type = Bool
            default = true
        "--output_text"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "--Hop_dir"
            help = "path of Hop.jl"
            arg_type = String
            default = "/home/lihe/DeepH/process_ham/Hop.jl/"
    end
    return parse_args(s)
end
parsed_args = parse_commandline()

using Pkg
Pkg.activate(parsed_args["Hop_dir"])
using Hop


site_positions = readdlm(joinpath(parsed_args["input_dir"], "site_positions.dat"))
lat = readdlm(joinpath(parsed_args["input_dir"], "lat.dat"))
R_list_read = convert(Matrix{Int64}, readdlm(joinpath(parsed_args["input_dir"], "R_list.dat")))
num_R = size(R_list_read, 1)
R_list = Vector{SVector{3, Int64}}()
for index_R in 1:num_R
    push!(R_list, SVector{3, Int64}(R_list_read[index_R, :]))
end

@info "get local coordinate"
begin_time = time()
rcoordinate = Hop.Deeph.rotate_system(site_positions, lat, R_list, parsed_args["radius"])
println("time for calculating local coordinate is: ", time() - begin_time)

if parsed_args["output_text"]
    @info "output txt"
    mkpath(joinpath(parsed_args["output_dir"], "rresult"))
    mkpath(joinpath(parsed_args["output_dir"], "rresult/rc"))
    for (R, coord) in rcoordinate
        open(joinpath(parsed_args["output_dir"], "rresult/rc/", R, "_real.dat"), "w") do f
            writedlm(f, coord)
        end
    end
end

@info "output h5"
h5open(joinpath(parsed_args["input_dir"], "overlaps.h5"), "r") do fid_OLP
    graph_key = Set(keys(fid_OLP))
    h5open(joinpath(parsed_args["output_dir"], "rc.h5"), "w") do fid
        for (key, coord) in rcoordinate
            if (parsed_args["create_from_DFT"] == true) && (!(string(key) in graph_key))
                continue
            end
            write(fid, string(key), permutedims(coord))
        end
    end
end
