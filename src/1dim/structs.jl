using StaticArrays
using LinearAlgebra
using Base.Threads

# Used scheme
# U - conserved varaibles
# U1 = rho ut - mass conservation
# U2 = T^t_t - energy conservation
# U3 = T^t_x - momentum conservation

# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = ux four-velocity in x

mutable struct ParVector1D{T <:Real,N}
    # Parameter Vector
    arr1::MVector{N,T}
    arr2::MVector{N,T}
    arr3::MVector{N,T}
    size::Int64
    function ParVector1D{T,N}() where {T, N}
        arr1 = MVector{N,T}(zeros(N))
        arr2 = MVector{N,T}(zeros(N))
        arr3 = MVector{N,T}(zeros(N))
        new(arr1,arr2,arr3,N)
    end
end

Base.copy(s::ParVector1D) = ParVector1D(s.arr1, s.arr2,s.arr3,s.size)

function Jacobian(x::MVector{3,Float64},buffer::MMatrix{3,3,Float64},eos::Polytrope)
    gam::Float64 = sqrt(1+x[3]^2) ### gamma factor
    w::Float64 = eos.gamma * x[2] + x[1] ### enthalpy w = p + u + rho
    buffer[1,1] = gam
    buffer[1,2] = 0
    buffer[1,3] = x[1] * x[3] / gam

    buffer[2,1] =  - gam^2
    buffer[2,2] =  (eos.gamma - 1) - gam ^2 * eos.gamma
    buffer[2,3] =  -2*x[3] * w
    
    buffer[3,1] = gam * x[3]
    buffer[3,2] = gam * x[3] * eos.gamma
    buffer[3,3] = (2 * x[3]^2 + 1) / gam * w
end

function F_ptoU(x::MVector{3,Float64}, buffer::MVector{3,Float64},eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam * x[1]
    buffer[2] = (eos.gamma-1) * x[2] - gam^2 * w
    buffer[3] = x[3] * gam * w
end



function PtoU(P::ParVector1D,U::ParVector1D,eos::Polytrope)
    @threads :static for i in 1:P.size
        gamma = sqrt(P.arr3[i]^2+1)
        pressure = (eos.gamma-1)* P.arr2[i]
        U.arr1[i] = P.arr1[i]*gamma
        U.arr2[i] = -(P.arr2[i] + pressure + P.arr1[i]) * gamma^2 + pressure
        U.arr3[i] = (P.arr2[i] + pressure + P.arr1[i]) * gamma * P.arr3[i]
    end
end

function PtoF(P::ParVector1D,F::ParVector1D,eos::Polytrope)
    @threads :static for i in 1:P.size
        gamma::Float64 = sqrt(P.arr3[i]^2+1)
        pressure::Float64 = (eos.gamma-1)* P.arr2[i]
        F.arr1[i] = P.arr1[i] * P.arr3[i]
        F.arr2[i] = -(P.arr2[i] + pressure + P.arr1[i]) * gamma * P.arr3[i]
        F.arr3[i] = pressure + (P.arr2[i] + pressure + P.arr1[i]) * P.arr3[i]^2
    end
end

function UtoP(U::ParVector1D,P::ParVector1D,eos::EOS,n_iter::Int64,tol::Float64=1e-10)
    @threads :static for i in 1:P.size
        buff_start::MVector{3,Float64} = MVector(0.,0.,0.)
        buff_fun::MVector{3,Float64} = MVector(0.,0.,0.)
        buff_jac::MMatrix{3,3,Float64} = @MMatrix zeros(3,3)

        buff_start[1] = P.arr1[i]
        buff_start[2] = P.arr2[i]
        buff_start[3] = P.arr3[i]
        buff_fun = MVector(0.,0.,0.)
        for num in 1:n_iter
            F_ptoU(buff_start,buff_fun,eos)
            Jacobian(buff_start,buff_jac,eos)
            buff_fun[1] = buff_fun[1] - U.arr1[i]
            buff_fun[2] = buff_fun[2] - U.arr2[i]
            buff_fun[3] = buff_fun[3] - U.arr3[i]
            buff_fun = buff_jac \ buff_fun
            if norm(buff_fun) < tol
                break
            end
            buff_start = buff_start .- buff_fun
            #println(buff_fun)
        end
        P.arr1[i] = buff_start[1]
        P.arr2[i] = buff_start[2]
        P.arr3[i] = buff_start[3]
    end
end

