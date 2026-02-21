using Base.Threads

function LaxFriedrich(
    P::ParVector1D,
    N::Int64,
    dt::Float64,
    dx::Float64,
    T::Float64,
    eos::EOS,
    drops::Float64,
    kwargs...,
)
    U::ParVector1D = ParVector1D{Float64,N}()

    F::ParVector1D = ParVector1D{Float64,N}()
    t::Float64 = 0
    PtoU(P, U, eos)
    Ubuffer = deepcopy(U)

    out::Vector{ParVector1D} = []
    thres_to_dump::Float64 = drops
    push!(out, deepcopy(P))
    while t < T

        PtoF(P, F, eos) #calculating fluxes

        for i = 2:(N-1)
            @inbounds Ubuffer.arr1[i] =
                0.5 * (U.arr1[i-1] + U.arr1[i+1]) -
                0.5 * dt/dx * (F.arr1[i+1] - F.arr1[i-1])
            @inbounds Ubuffer.arr2[i] =
                0.5 * (U.arr2[i-1] + U.arr2[i+1]) -
                0.5 * dt/dx * (F.arr2[i+1] - F.arr2[i-1])
            @inbounds Ubuffer.arr3[i] =
                0.5 * (U.arr3[i-1] + U.arr3[i+1]) -
                0.5 * dt/dx * (F.arr3[i+1] - F.arr3[i-1])
        end

        for i = 1:N
            @inbounds U.arr1[i] = Ubuffer.arr1[i]
            @inbounds U.arr2[i] = Ubuffer.arr2[i]
            @inbounds U.arr3[i] = Ubuffer.arr3[i]
        end

        UtoP(U, P, eos, kwargs...) #Conversion to primitive variables

        t += dt
        if t > thres_to_dump
            push!(out, deepcopy(P))
            thres_to_dump += drops
            println(t)
        end
    end
    return out
end


function HARM_HLL(
    P::ParVector1D,
    N::Int64,
    dt::Float64,
    dx::Float64,
    T::Float64,
    eos::EOS,
    drops::Float64,
    rec::Symbol,
    kwargs...,
)
    U::ParVector1D = ParVector1D{Float64}(N)

    PR = deepcopy(P)
    PL = deepcopy(P)

    UL::ParVector1D = ParVector1D{Float64}(N)
    UR::ParVector1D = ParVector1D{Float64}(N)

    FL::ParVector1D = ParVector1D{Float64}(N) #Left flux
    FR::ParVector1D = ParVector1D{Float64}(N) #Right flux
    F::ParVector1D = ParVector1D{Float64}(N) # HLL flux

    t::Float64 = 0
    PtoU(P, U, eos)
    Uhalf = deepcopy(U)
    Phalf = deepcopy(P)

    out::Vector{ParVector1D} = []
    thres_to_dump::Float64 = drops
    push!(out, deepcopy(P))
    while t < T

        if rec == :minmod
            @threads :static for i = 2:(N-1) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] = MINMOD(P.arr1[i-1], P.arr1[i], P.arr1[i+1])
                PR.arr2[i-1], PL.arr2[i] = MINMOD(P.arr2[i-1], P.arr2[i], P.arr2[i+1])
                PR.arr3[i-1], PL.arr3[i] = MINMOD(P.arr3[i-1], P.arr3[i], P.arr3[i+1])
            end
        elseif rec == :ppm
            @threads :static for i = 3:(N-2) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] =
                    PPM(P.arr1[i-2], P.arr1[i-1], P.arr1[i], P.arr1[i+1], P.arr1[i+2])
                PR.arr2[i-1], PL.arr2[i] =
                    PPM(P.arr2[i-2], P.arr2[i-1], P.arr2[i], P.arr2[i+1], P.arr2[i+2])
                PR.arr3[i-1], PL.arr3[i] =
                    PPM(P.arr3[i-2], P.arr3[i-1], P.arr3[i], P.arr3[i+1], P.arr3[i+2])
            end
        elseif rec == :wenoz
            @threads :static for i = 3:(N-2) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] =
                    WENOZ(P.arr1[i-2], P.arr1[i-1], P.arr1[i], P.arr1[i+1], P.arr1[i+2])
                PR.arr2[i-1], PL.arr2[i] =
                    WENOZ(P.arr2[i-2], P.arr2[i-1], P.arr2[i], P.arr2[i+1], P.arr2[i+2])
                PR.arr3[i-1], PL.arr3[i] =
                    WENOZ(P.arr3[i-2], P.arr3[i-1], P.arr3[i], P.arr3[i+1], P.arr3[i+2])
            end
        else
            println("reconstruction not supported!")
        end
        if minimum(PL.arr1) < 0
            println(argmin(PL.arr1))
            #return
        end
        PtoU(PR, UR, eos)
        PtoU(PL, UL, eos)
        PtoF(PR, FR, eos)
        PtoF(PL, FL, eos)

        for i = 1:N
            vL::Float64 = PL.arr3[i] / sqrt(PL.arr3[i]^2 + 1)
            vR::Float64 = PR.arr3[i] / sqrt(PR.arr3[i]^2 + 1)
            CL::Float64 = SoundSpeed(PL.arr1[i], PL.arr2[i], eos)
            CR::Float64 = SoundSpeed(PR.arr1[i], PR.arr2[i], eos)

            sigma_S_L::Float64 = CL^2 / ((PL.arr3[i]^2 + 1) * (1-CL^2))
            sigma_S_R::Float64 = CR^2 / ((PR.arr3[i]^2 + 1) * (1-CR^2))

            C_max::Float64 = max(
                (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L))) / (1 + sigma_S_L),
                (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R))) / (1 + sigma_S_R),
            ) # velocity composition
            C_min::Float64 =
                -min(
                    (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L))) / (1 + sigma_S_L),
                    (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R))) / (1 + sigma_S_R),
                ) # velocity composition

            if C_max < 0
                F.arr1[i] = FR.arr1[i]
                F.arr2[i] = FR.arr2[i]
                F.arr3[i] = FR.arr3[i]
            elseif C_min < 0
                F.arr1[i] = FL.arr1[i]
                F.arr2[i] = FL.arr2[i]
                F.arr3[i] = FL.arr3[i]
            else
                @inbounds F.arr1[i] =
                    (
                        FR.arr1[i] * C_min + FL.arr1[i] * C_max -
                        C_max * C_min * (UR.arr1[i] - UL.arr1[i])
                    ) / (C_max + C_min)
                @inbounds F.arr2[i] =
                    (
                        FR.arr2[i] * C_min + FL.arr2[i] * C_max -
                        C_max * C_min * (UR.arr2[i] - UL.arr2[i])
                    ) / (C_max + C_min)
                @inbounds F.arr3[i] =
                    (
                        FR.arr3[i] * C_min + FL.arr3[i] * C_max -
                        C_max * C_min * (UR.arr3[i] - UL.arr3[i])
                    ) / (C_max + C_min)
            end
        end

        @threads :static for i = 2:(N-2)
            @inbounds Uhalf.arr1[i] = U.arr1[i] - 0.5 * dt/dx * (F.arr1[i] - F.arr1[i-1])
            @inbounds Uhalf.arr2[i] = U.arr2[i] - 0.5 * dt/dx * (F.arr2[i] - F.arr2[i-1])
            @inbounds Uhalf.arr3[i] = U.arr3[i] - 0.5 * dt/dx * (F.arr3[i] - F.arr3[i-1])
        end

        Phalf = deepcopy(P)
        UtoP(Uhalf, Phalf, eos, kwargs...) #Conversion to primitive variables
        #### Second iteration 


        if rec == :minmod
            @threads :static for i = 2:(N-1) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] =
                    MINMOD(Phalf.arr1[i-1], Phalf.arr1[i], Phalf.arr1[i+1])
                PR.arr2[i-1], PL.arr2[i] =
                    MINMOD(Phalf.arr2[i-1], Phalf.arr2[i], Phalf.arr2[i+1])
                PR.arr3[i-1], PL.arr3[i] =
                    MINMOD(Phalf.arr3[i-1], Phalf.arr3[i], Phalf.arr3[i+1])
            end
        elseif rec == :ppm
            @threads :static for i = 3:(N-2) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] = PPM(
                    Phalf.arr1[i-2],
                    Phalf.arr1[i-1],
                    Phalf.arr1[i],
                    Phalf.arr1[i+1],
                    Phalf.arr1[i+2],
                )
                PR.arr2[i-1], PL.arr2[i] = PPM(
                    Phalf.arr2[i-2],
                    Phalf.arr2[i-1],
                    Phalf.arr2[i],
                    Phalf.arr2[i+1],
                    Phalf.arr2[i+2],
                )
                PR.arr3[i-1], PL.arr3[i] = PPM(
                    Phalf.arr3[i-2],
                    Phalf.arr3[i-1],
                    Phalf.arr3[i],
                    Phalf.arr3[i+1],
                    Phalf.arr3[i+2],
                )
            end
        elseif rec == :wenoz
            @threads :static for i = 3:(N-2) # interpolating left and right
                PR.arr1[i-1], PL.arr1[i] = WENOZ(
                    Phalf.arr1[i-2],
                    Phalf.arr1[i-1],
                    Phalf.arr1[i],
                    Phalf.arr1[i+1],
                    Phalf.arr1[i+2],
                )
                PR.arr2[i-1], PL.arr2[i] = WENOZ(
                    Phalf.arr2[i-2],
                    Phalf.arr2[i-1],
                    Phalf.arr2[i],
                    Phalf.arr2[i+1],
                    Phalf.arr2[i+2],
                )
                PR.arr3[i-1], PL.arr3[i] = WENOZ(
                    Phalf.arr3[i-2],
                    Phalf.arr3[i-1],
                    Phalf.arr3[i],
                    Phalf.arr3[i+1],
                    Phalf.arr3[i+2],
                )
            end
        else
            println("reconstruction not supported!")
        end
        if minimum(PL.arr1) < 0
            println(argmin(PL.arr1))
        end
        PtoU(PR, UR, eos)
        PtoU(PL, UL, eos)
        PtoF(PR, FR, eos)
        PtoF(PL, FL, eos)

        for i = 1:N
            vL::Float64 = PL.arr3[i] / sqrt(PL.arr3[i]^2 + 1)
            vR::Float64 = PR.arr3[i] / sqrt(PR.arr3[i]^2 + 1)
            CL::Float64 = SoundSpeed(PL.arr1[i], PL.arr2[i], eos)
            CR::Float64 = SoundSpeed(PR.arr1[i], PR.arr2[i], eos)

            sigma_S_L::Float64 = CL^2 / ((PL.arr3[i]^2 + 1) * (1-CL^2))
            sigma_S_R::Float64 = CR^2 / ((PR.arr3[i]^2 + 1) * (1-CR^2))

            C_max::Float64 = max(
                (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L))) / (1 + sigma_S_L),
                (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R))) / (1 + sigma_S_R),
            ) # velocity composition
            C_min::Float64 =
                -min(
                    (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L))) / (1 + sigma_S_L),
                    (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R))) / (1 + sigma_S_R),
                ) # velocity composition

            if C_max < 0
                F.arr1[i] = FR.arr1[i]
                F.arr2[i] = FR.arr2[i]
                F.arr3[i] = FR.arr3[i]
            elseif C_min < 0
                F.arr1[i] = FL.arr1[i]
                F.arr2[i] = FL.arr2[i]
                F.arr3[i] = FL.arr3[i]
            else
                @inbounds F.arr1[i] =
                    (
                        FR.arr1[i] * C_min + FL.arr1[i] * C_max -
                        C_max * C_min * (UR.arr1[i] - UL.arr1[i])
                    ) / (C_max + C_min)
                @inbounds F.arr2[i] =
                    (
                        FR.arr2[i] * C_min + FL.arr2[i] * C_max -
                        C_max * C_min * (UR.arr2[i] - UL.arr2[i])
                    ) / (C_max + C_min)
                @inbounds F.arr3[i] =
                    (
                        FR.arr3[i] * C_min + FL.arr3[i] * C_max -
                        C_max * C_min * (UR.arr3[i] - UL.arr3[i])
                    ) / (C_max + C_min)
            end
        end

        @threads :static for i = 2:(N-2)
            @inbounds U.arr1[i] = U.arr1[i] - dt/dx * (F.arr1[i] - F.arr1[i-1])
            @inbounds U.arr2[i] = U.arr2[i] - dt/dx * (F.arr2[i] - F.arr2[i-1])
            @inbounds U.arr3[i] = U.arr3[i] - dt/dx * (F.arr3[i] - F.arr3[i-1])
        end
        UtoP(U, P, eos, kwargs...) #Conversion to primitive variables

        t += dt
        if t > thres_to_dump
            push!(out, deepcopy(P))
            #println(P.arr3)
            thres_to_dump += drops
            println(t)
        end
        #update(pbar))
    end
    return out
end
