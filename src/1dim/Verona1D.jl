module Verona1D
    using ..EosTypes
    include("../fluxlimiter.jl")
    include("structs.jl")
    include("algos.jl")
end
