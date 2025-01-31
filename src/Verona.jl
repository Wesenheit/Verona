module Verona
    include("eos.jl")
    include("1dim/Verona1D.jl")
    include("2dim/Verona2D.jl")

    export Verona1D
    export Verona2D
end
