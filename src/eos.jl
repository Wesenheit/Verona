module EosTypes

    export EOS,Polytrope,SoundSpeed

    abstract type EOS{T} end

    struct Polytrope{T} <:EOS{T}
        gamma::T
    end

    function Pressure(u::T,eos::Polytrope{T})::T where T
        return (eos.gamma-1)*u
    end

    function SoundSpeed(rho::T,u::T,eos::Polytrope{T})::T where T
        return sqrt((eos.gamma * (eos.gamma - 1) * u )/(rho + eos.gamma * u))
    end

    function SoundSpeed(rho::T,u::T,gamma::T)::T where T
        return sqrt((gamma * (gamma - 1) * u )/(rho + gamma * u))
    end
end