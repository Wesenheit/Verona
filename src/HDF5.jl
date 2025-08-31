using HDF5
using Printf

@inline function save_dump(P, B1, B2, B3, iter)
    # Rozmiary siatki
    Nx = size(P, 2)
    Ny = size(P, 3)
    Nz = size(P, 4)

    # Uśrednianie pól magnetycznych do centrów komórek
    B1c = 0.5 .* (B1[1:Nx,   :, :] .+ B1[2:Nx+1, :, :])
    B2c = 0.5 .* (B2[:, 1:Ny,   :] .+ B2[:, 2:Ny+1, :])
    B3c = 0.5 .* (B3[:,   :, 1:Nz] .+ B3[:,   :, 2:Nz+1])

    found_nan = false
    P_names = ["rho", "u", "ux", "uy", "uz"]

    # Sprawdzenie NaN w B1
    for i in 1:Nx+1, j in 1:Ny, k in 1:Nz
        if isnan(B1[i, j, k])
            @error "NaN w B1 na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
    end

    # Sprawdzenie NaN w B2
    for i in 1:Nx, j in 1:Ny+1, k in 1:Nz
        if isnan(B2[i, j, k])
            @error "NaN w B2 na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
    end

    # Sprawdzenie NaN w B3
    for i in 1:Nx, j in 1:Ny, k in 1:Nz+1
        if isnan(B3[i, j, k])
            @error "NaN w B3 na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
    end

    # Sprawdzenie NaN w uśrednionych B1c, B2c, B3c (opcjonalne)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        if isnan(B1c[i, j, k])
            @error "NaN w B1c na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
        if isnan(B2c[i, j, k])
            @error "NaN w B2c na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
        if isnan(B3c[i, j, k])
            @error "NaN w B3c na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            found_nan = true
        end
    end
    
    for n in 1:size(P, 1), i in 1:Nx, j in 1:Ny, k in 1:Nz
        if isnan(P[n, i, j, k])
            @error "NaN w P[$n] ($(P_names[n])) na pozycji (i=$i, j=$j, k=$k) przy iteracji $iter"
            println("rho: ",P[1, i, j, k]) 
            println("u: ",P[2, i, j, k])           
            println("B1: ",B1c[i, j, k])
            println("B2: ",B2c[i, j, k])
            println("B3: ",B3c[i, j, k])
            found_nan = true
        end
    end
    
    if found_nan
        error("save_dump: wykryto NaN w tablicach P lub B przy iteracji $iter — szczegóły powyżej")
    end

    # Wektory współrzędnych
    xs = ((collect(1:Nx) .- 0.5) .* dx)
    ys = ((collect(1:Ny) .- 0.5) .* dy)
    zs = ((collect(1:Nz) .- 0.5) .* dz)

    # Zapis do pliku HDF5
    fname = @sprintf("dump_%05d.h5", iter)
    @info "Zapisuję snapshot do $fname"
    h5open(fname, "w") do file
        write(file, "rho", P[1, :, :, :])
        write(file, "u",   P[2, :, :, :])
        write(file, "ux",  P[3, :, :, :])
        write(file, "uy",  P[4, :, :, :])
        write(file, "uz",  P[5, :, :, :])
        write(file, "B1",  B1c)
        write(file, "B2",  B2c)
        write(file, "B3",  B3c)
        write(file, "x",   xs)
        write(file, "y",   ys)
        write(file, "z",   zs)
    end
    println(" → Gotowe: zapisano wszystkie pola w pliku $fname")
end

