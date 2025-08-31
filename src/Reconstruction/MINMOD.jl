@inline function MINMOD(q_im2::T,q_im1::T,q_i::T,q_ip1::T,q_ip2::T) where T<:Real
    @inbounds begin
        sp = q_ip1 - q_i
        sm = q_i - q_im1
        ssp = sign(sp)
        ssm = sign(sm)
        asp = abs(sp)
        asm = abs(sm)
        dU = T(0.25) * (ssp + ssm) * min(asp,asm)
        return q_i - dU, q_i + dU   
    end 
end
