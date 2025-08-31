@inline function compute_B_from_A!(
    B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T},
    A1::AbstractArray{T}, A2::AbstractArray{T}, A3::AbstractArray{T},
    dx::T, dy::T, dz::T) where T<:Real

    @threads for i in 1:Nx+1
        for j in 1:Ny
            for k in 1:Nz
                B1[i,j,k] = (A3[i,j+1,k] - A3[i,j,k]) / dy -
                             (A2[i,j,k+1] - A2[i,j,k]) / dz
            end
        end
    end

    @threads for i in 1:Nx
        for j in 1:Ny+1
            for k in 1:Nz
                B2[i,j,k] = (A1[i,j,k+1] - A1[i,j,k]) / dz -
                             (A3[i+1,j,k] - A3[i,j,k]) / dx
            end
        end
    end

    @threads for i in 1:Nx
        for j in 1:Ny
            for k in 1:Nz+1
                B3[i,j,k] = (A2[i+1,j,k] - A2[i,j,k]) / dx -
                             (A1[i,j+1,k] - A1[i,j,k]) / dy
            end
        end
    end
end


@inline function divB(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, dx::T, dy::T, dz::T) where T<:Real
    total_div::T = 0.0
 for i in 1:Nx
        for j in 1:Ny
            for k in 1:Nz
                dBx_dx = (B1[i+1,j,k] - B1[i,j,k]) / dx
                dBy_dy = (B2[i,j+1,k] - B2[i,j,k]) / dy
                dBz_dz = (B3[i,j,k+1] - B3[i,j,k]) / dz
                total_div += dBx_dx + dBy_dy + dBz_dz
            end
        end
    end

    return total_div
end

@inline function floors!(P::AbstractArray{T},B1::AbstractArray{T},B2::AbstractArray{T},B3::AbstractArray{T}) where T<:Real
@threads for i in 1:Nx
            for j in 1:Ny
                for k in 1:Nz
                    #Primitive variables
                    ρ  = P[1,i,j,k]                                    
                    u  = P[2,i,j,k]                                   
                    u¹ = P[3,i,j,k]                                   
                    u² = P[4,i,j,k]                                     
                    u³ = P[5,i,j,k]                                       
                    B¹ = 1/2 *(B1[i,j,k]+B1[i+1,j,k])
                    B² = 1/2 *(B2[i,j,k]+B2[i,j+1,k])
                    B³ = 1/2 *(B3[i,j,k]+B3[i,j,k+1])

                    # Contravariant time component of the four-velocity
                    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)
                    
                    #LORENTZ FACTOR LIMITER
                    v¹ = u¹/u⁰
                    v² = u²/u⁰
                    v³ = u³/u⁰
                    u⁰ = min(u⁰,GAMMAMAX)

                    u¹ = v¹*u⁰
                    u² = v²*u⁰
                    u³ = v³*u⁰

                    # Covariant components of the four-velocity
                    u₀ = -u⁰ 
                    u₁ =  u¹
                    u₂ =  u²
                    u₃ =  u³

                    # Contravariant components of the magnetic four-vector
                    b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
                    b¹ = (B¹ + b⁰*u¹)/u⁰
                    b² = (B² + b⁰*u²)/u⁰
                    b³ = (B³ + b⁰*u³)/u⁰	

                    # Covariant components of the magnetic four-vector
                    b₀ = -b⁰
                    b₁ =  b¹
                    b₂ =  b²
                    b₃ =  b³	

                    # Magnetic four-vector contraction            
                    bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
                    
                    ρ = max(ρ,floor)                    
                    u = max(u,floor)
                    
                    ρ = max(ρ, bsq/MAXSIGMARHO)
                    u = max(u, bsq/MAXSIGMAUU)
                    
                    P[1,i,j,k] = ρ                            
                    P[2,i,j,k] = u                            
                    P[3,i,j,k] = u¹                          
                    P[4,i,j,k] = u²                       
                    P[5,i,j,k] = u³
                end
            end
        end
end


function divB_inner(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, dx::T, dy::T, dz::T) where T<:Real
    total = 0.0
    for i in 5:Nx-5
        for j in 5:Ny-5
            for k in 5:Nz-5
                dBx_dx = (B1[i+1,j,k] - B1[i,j,k]) / dx
                dBy_dy = (B2[i,j+1,k] - B2[i,j,k]) / dy
                dBz_dz = (B3[i,j,k+1] - B3[i,j,k]) / dz

                total += dBx_dx + dBy_dy + dBz_dz
            end
        end
    end

    return total
end


function find_bad_cellssss(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, dx::T, dy::T, dz::T; threshold::T = T(1e-8)) where T<:Real
    bad = Tuple{Int,Int,Int}[]
    Nx = size(B1,1) - 2
    Ny = size(B1,2) - 2
    Nz = size(B1,3) - 2

    for i in 2:Nx+1, j in 2:Ny+1, k in 2:Nz+1
        dBx_dx = (B1[i+1,j  ,k  ] - B1[i,j,k]) / dx
        dBy_dy = (B2[i  ,j+1,k  ] - B2[i,j,k]) / dy
        dBz_dz = (B3[i  ,j  ,k+1] - B3[i,j,k]) / dz

        div = dBx_dx + dBy_dy + dBz_dz
       # println("∇*B = ",div)
        if abs(div) > threshold
            push!(bad, (i-1, j-1, k-1))  
        end
    end

    return bad
end

function find_bad_cellssss(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, dx::T, dy::T, dz::T; threshold::T = T(1e-8)) where T<:Real
    bad = Tuple{Int,Int,Int}[]
    Nx = size(B1,1) - 2
    Ny = size(B1,2) - 2
    Nz = size(B1,3) - 2

    max_div = 0.0
    max_loc = (0, 0, 0)

    for i in 2:Nx+1, j in 2:Ny+1, k in 2:Nz+1
        dBx_dx = (B1[i+1,j  ,k  ] - B1[i,j,k]) / dx
        dBy_dy = (B2[i  ,j+1,k  ] - B2[i,j,k]) / dy
        dBz_dz = (B3[i  ,j  ,k+1] - B3[i,j,k]) / dz

        div = dBx_dx + dBy_dy + dBz_dz

        if abs(div) > threshold
            coords = (i-1, j-1, k-1)
            println("∇·B = ", div, " w komórce ", coords)
            push!(bad, coords)
        end

        if abs(div) > abs(max_div)
            max_div = div
            max_loc = (i-1, j-1, k-1)
        end
    end

    println("Największa divergencja ∇·B = ", max_div, " w komórce ", max_loc)

    return bad, max_loc, max_div
end


function find_bad_cells(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, dx::T, dy::T, dz::T; threshold::T = T(1e-8)) where T<:Real
    bad = Tuple{Int,Int,Int}[]
    Nx = size(B1,1) - 2
    Ny = size(B1,2) - 2
    Nz = size(B1,3) - 2

    max_div = 0.0
    max_loc = (0, 0, 0)

    for i in 2:Nx+1, j in 2:Ny+1, k in 2:Nz+1
        dBx_dx = (B1[i+1,j  ,k  ] - B1[i,j,k]) / dx
        dBy_dy = (B2[i  ,j+1,k  ] - B2[i,j,k]) / dy
        dBz_dz = (B3[i  ,j  ,k+1] - B3[i,j,k]) / dz

        div = dBx_dx + dBy_dy + dBz_dz

        if abs(div) > threshold
            coords = (i-1, j-1, k-1)
            println("∇·B = ", div, " w komórce ", coords)
            push!(bad, coords)
        end

        if abs(div) > abs(max_div)
            max_div = div
            max_loc = (i-1, j-1, k-1)
        end
    end

    println("Największa divergencja ∇·B = ", max_div, " w komórce ", max_loc)

    return bad, max_loc, max_div
end

