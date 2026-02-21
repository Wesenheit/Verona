
@inline function local_to_global(i, p, Size, MPI)
    halo = 3
    interior_size = Size - 2 * halo
    if MPI == 1
        return i
    end
    if p == 0
        # First process: valid local indices are 1:(Size - halo)
        return (i >= 1 && i <= Size - halo) ? i : 0

    elseif p == MPI - 1
        # Last process: valid local indices are halo:Size
        offset = (Size - halo) + interior_size * (MPI - 2) - 1
        return (i > halo && i <= Size) ? offset + i - halo + 1 : 0

    else
        # Middle processes: valid local indices are halo:(Size - halo)
        offset = (Size - halo) + interior_size * (p - 1) - 1
        return (i > halo && i <= Size - halo) ? offset + i - halo + 1 : 0
    end
end
