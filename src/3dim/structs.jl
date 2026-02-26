using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5

abstract type VeronaArr{T} end

mutable struct ParVector3D{T<:Real} <: VeronaArr{T}
    arr::Array{T,4}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function ParVector3D{T}(Nx, Ny, Nz) where {T}
        arr = zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)
        new(arr, Nx + 6, Ny + 6, Nz + 6)
    end
    function ParVector3D{T}(arr::VeronaArr{T}) where {T}
        new(Array{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end
end

mutable struct CuParVector3D{T<:Real} <: VeronaArr{T}
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function CuParVector3D{T}(arr::VeronaArr{T}) where {T<:Real}
        new(CuArray{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end
    function CuParVector3D{T}(Nx::Int64, Ny::Int64, Nz::Int64) where {T<:Real}
        new(CuArray{T}(zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)), Nx + 6, Ny + 6, Nz + 6)
    end
end

function VectorLike(X::VeronaArr{T}) where {T}
    if typeof(X.arr) <: CuArray
        return CuParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    else
        return ParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    end
end

@kernel inbounds = true function kernel_PtoU(
    @Const(P::AbstractArray{T}),
    U::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    begin
        ρ = P[1, i, j, k]
        v¹ = P[2, i, j, k]
        v² = P[3, i, j, k]
        v³ = P[4, i, j, k]
        u = P[5, i, j, k]
        W = 1 / sqrt(1 - (v¹*v¹ + v²*v² + v³*v³))
        h = 1 + eos.gamma*u
        U[1, i, j, k] = ρ*W
        U[2, i, j, k] = (ρ*h) * W^2 * v¹
        U[3, i, j, k] = (ρ*h) * W^2 * v²
        U[4, i, j, k] = (ρ*h) * W^2 * v³
        U[5, i, j, k] = (ρ*h)*W^2 - ρ*u*(eos.gamma - 1) - ρ*W
    end
end

@inline function function_PtoU(
    P::AbstractVector{T},
    U::AbstractVector{T},
    eos::Polytrope{T},
) where {T<:Real}
    ρ = P[1]
    v¹ = P[2]
    v² = P[3]
    v³ = P[4]
    u = P[5]
    W = 1 / sqrt(1 - (v¹*v¹ + v²*v² + v³*v³))
    h = 1 + eos.gamma*u
    U[1] = ρ*W
    U[2] = ρ*h*W^2*v¹
    U[3] = ρ*h*W^2*v²
    U[4] = ρ*h*W^2*v³
    U[5] = ρ*h*W^2 - ρ*u*(eos.gamma - 1) - ρ*W
end

@inline function function_PtoFx(
    P::AbstractVector{T},
    Fx::AbstractVector{T},
    eos::Polytrope{T},
) where {T<:Real}
    ρ = P[1]
    v¹ = P[2]
    v² = P[3]
    v³ = P[4]
    u = P[5]
    W = 1 / sqrt(1 - (v¹*v¹ + v²*v² + v³*v³))
    h = 1 + eos.gamma*u
    D = ρ*W
    S₁ = ρ*h*W^2*v¹
    S₂ = ρ*h*W^2*v²
    S₃ = ρ*h*W^2*v³
    τ = ρ*h*W^2 - ρ*u*(eos.gamma - 1) - D
    Fx[1] = D*v¹
    Fx[2] = S₁*v¹ + ρ*u*(eos.gamma - 1)
    Fx[3] = S₂*v¹
    Fx[4] = S₃*v¹
    Fx[5] = τ*v¹ + ρ*u*(eos.gamma - 1)*v¹
end

@inline function function_PtoFy(
    P::AbstractArray{T},
    Fy::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    ρ = P[1]
    v¹ = P[2]
    v² = P[3]
    v³ = P[4]
    u = P[5]
    W = 1 / sqrt(1 - (v¹*v¹ + v²*v² + v³*v³))
    h = 1 + eos.gamma*u
    D = ρ*W
    S₁ = ρ*h*W^2*v¹
    S₂ = ρ*h*W^2*v²
    S₃ = ρ*h*W^2*v³
    τ = ρ*h*W^2 - ρ*u*(eos.gamma - 1) - D
    Fy[1] = D*v²
    Fy[2] = S₁*v²
    Fy[3] = S₂*v² + ρ*u*(eos.gamma - 1)
    Fy[4] = S₃*v²
    Fy[5] = τ*v² + ρ*u*(eos.gamma - 1)*v²
end

@inline function function_PtoFz(
    P::AbstractArray{T},
    Fz::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    ρ = P[1]
    v¹ = P[2]
    v² = P[3]
    v³ = P[4]
    u = P[5]
    W = 1 / sqrt(1 - (v¹*v¹ + v²*v² + v³*v³))
    h = 1 + eos.gamma*u
    D = ρ*W
    S₁ = ρ*h*W^2*v¹
    S₂ = ρ*h*W^2*v²
    S₃ = ρ*h*W^2*v³
    τ = ρ*h*W^2 - ρ*u*(eos.gamma - 1) - D
    Fz[1] = D*v³
    Fz[2] = S₁*v³
    Fz[3] = S₂*v³
    Fz[4] = S₃*v³ + ρ*u*(eos.gamma - 1)
    Fz[5] = τ*v³ + ρ*u*(eos.gamma - 1)*v³
end

@inline function _f_UtoP(z::T, τ::T, D::T, S²::T, γ::T) where {T<:Real}
    z2 = z*z
    x = one(T) - S² / z2
    x = ifelse(x > zero(T), x, zero(T))
    root = sqrt(x)
    τ + D - z + ((γ - one(T))/γ) * (z*x - D*root)
end

@kernel inbounds = true function function_UtoP(
    @Const(U::AbstractArray{T}),
    P::AbstractArray{T},
    eos::Polytrope{T},
    n_iter::Int64,
    tol::T = T(1e-10),
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    il, jl, kl = @index(Local, NTuple)

    @uniform begin
        N, M, L = @groupsize()
        Nx, Ny, Nz = @ndrange()
    end

    Ploc = @localmem eltype(U) (5, N, M, L)
    Uloc = @localmem eltype(U) (5, N, M, L)

    for idx = 1:5
        Ploc[idx, il, jl, kl] = P[idx, i, j, k]
        Uloc[idx, il, jl, kl] = U[idx, i, j, k]
    end

    if i > 3 && i < Nx - 2 && j > 3 && j < Ny - 2 && k > 3 && k < Nz - 2
        D  = Uloc[1, il, jl, kl]
        S₁ = Uloc[2, il, jl, kl]
        S₂ = Uloc[3, il, jl, kl]
        S₃ = Uloc[4, il, jl, kl]
        τ  = Uloc[5, il, jl, kl]

        eps_guard = max(T(1e-12), T(100) * eps(T))
        v2max = one(T) - eps_guard

        ρmin = T(1e-8)
        umin = T(1e-8)

        okU = isfinite(D) && isfinite(S₁) && isfinite(S₂) && isfinite(S₃) && isfinite(τ) && (D > zero(T))
        if !okU
            Ploc[1, il, jl, kl] = max(Ploc[1, il, jl, kl], ρmin)
            Ploc[2, il, jl, kl] = zero(T)
            Ploc[3, il, jl, kl] = zero(T)
            Ploc[4, il, jl, kl] = zero(T)
            Ploc[5, il, jl, kl] = max(Ploc[5, il, jl, kl], umin)
        else
            S² = S₁*S₁ + S₂*S₂ + S₃*S₃

            v2_prev = Ploc[2, il, jl, kl]^2 + Ploc[3, il, jl, kl]^2 + Ploc[4, il, jl, kl]^2
            if !isfinite(v2_prev) || v2_prev >= v2max
                v2_prev = min(max(v2_prev, zero(T)), v2max)
            end
            denom_prev = one(T) - v2_prev
            denom_prev = ifelse(denom_prev > eps_guard, denom_prev, eps_guard)

            Z_guess = Ploc[1, il, jl, kl] * (one(T) + eos.gamma*Ploc[5, il, jl, kl]) / denom_prev
            if !isfinite(Z_guess) || Z_guess <= zero(T)
                Z_guess = max(D, sqrt(S²) * (one(T) + eps_guard))
            end

            Z_min = max(D, sqrt(S²) * (one(T) + eps_guard))
            a = Z_min
            b = max(T(2)*Z_guess, T(2)*a)

            fa = _f_UtoP(a, τ, D, S², eos.gamma)
            fb = _f_UtoP(b, τ, D, S², eos.gamma)

            bracketed = false
            if isfinite(fa) && isfinite(fb)
                for _ = 1:50
                    if fa * fb < 0
                        bracketed = true
                        break
                    end
                    b = T(2) * b
                    if !isfinite(b)
                        break
                    end
                    fb = _f_UtoP(b, τ, D, S², eos.gamma)
                    if !isfinite(fb)
                        break
                    end
                end
            end

            converged = false
            if bracketed
                if abs(fa) < abs(fb)
                    a, b = b, a
                    fa, fb = fb, fa
                end

                c = a
                fc = fa
                d = zero(T)
                mflag = true

                atol = tol
                rtol = sqrt(eps(T))

                nit = n_iter
                nit = ifelse(nit > 0, nit, 1)
                nit = min(nit, 1000)

                for _ = 1:nit
                    tol_ba = atol + rtol*max(abs(a), abs(b))
                    if fb == 0 || abs(b - a) <= tol_ba
                        converged = true
                        break
                    end

                    s = if (fa != fc) && (fb != fc)
                        (a*fb*fc)/((fa - fb)*(fa - fc)) +
                        (b*fa*fc)/((fb - fa)*(fb - fc)) +
                        (c*fa*fb)/((fc - fa)*(fc - fb))
                    else
                        b - fb*(b - a)/(fb - fa)
                    end

                    if ((s - (T(3)*a + b)/T(4))*(s - b) >= 0) ||
                       (mflag && abs(s - b) >= abs(b - c)/T(2)) ||
                       (!mflag && abs(s - b) >= abs(c - d)/T(2)) ||
                       (mflag && abs(b - c) < tol_ba) ||
                       (!mflag && abs(c - d) < tol_ba)
                        s = (a + b)/T(2)
                        mflag = true
                    else
                        mflag = false
                    end

                    d, c, fc = c, b, fb
                    fs = _f_UtoP(s, τ, D, S², eos.gamma)
                    if !isfinite(fs)
                        converged = false
                        break
                    end

                    if fa * fs < 0
                        b, fb = s, fs
                    else
                        a, fa = s, fs
                    end

                    if abs(fa) < abs(fb)
                        a, b = b, a
                        fa, fb = fb, fa
                    end
                end
            end

            okZ = bracketed && converged && isfinite(b) && (b >= Z_min)
            if !okZ
                ρ = max(D, ρmin)
                u = (τ + D)/ρ - one(T)
                if !isfinite(u) || u < umin
                    u = umin
                end
                Ploc[1, il, jl, kl] = ρ
                Ploc[2, il, jl, kl] = zero(T)
                Ploc[3, il, jl, kl] = zero(T)
                Ploc[4, il, jl, kl] = zero(T)
                Ploc[5, il, jl, kl] = u
            else
                Z_SOL = b

                v1 = S₁ / Z_SOL
                v2 = S₂ / Z_SOL
                v3 = S₃ / Z_SOL

                vsq = v1*v1 + v2*v2 + v3*v3
                if !isfinite(vsq) || vsq >= v2max
                    if isfinite(vsq) && vsq > zero(T)
                        s = sqrt(v2max / vsq)
                        v1 *= s
                        v2 *= s
                        v3 *= s
                        vsq = v1*v1 + v2*v2 + v3*v3
                    else
                        v1 = zero(T); v2 = zero(T); v3 = zero(T)
                        vsq = zero(T)
                    end
                end

                W = inv(sqrt(one(T) - vsq))
                Wmax = T(50)
                if !isfinite(W) || W < one(T)
                    W = one(T)
                elseif W > Wmax
                    W = Wmax
                end

                ρ = D / W
                if !isfinite(ρ) || ρ < ρmin
                    ρ = ρmin
                end

                denom = ρ * W * W
                denom = ifelse(denom > ρmin, denom, ρmin)
                u = (Z_SOL/denom - one(T)) / eos.gamma
                if !isfinite(u) || u < umin
                    u = umin
                end

                Ploc[1, il, jl, kl] = ρ
                Ploc[2, il, jl, kl] = v1
                Ploc[3, il, jl, kl] = v2
                Ploc[4, il, jl, kl] = v3
                Ploc[5, il, jl, kl] = u
            end
        end
    end

    @synchronize

    for idx = 1:5
        P[idx, i, j, k] = Ploc[idx, il, jl, kl]
    end
end
