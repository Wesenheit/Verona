function MINMOD(q_im2, q_im1, q_i, q_ip1, q_ip2)
    """
    simple minmod reconstruction
    """
    sp = q_ip1 - q_i
    sm = q_i - q_im1
    ssp = sign(sp)
    ssm = sign(sm)
    asp = abs(sp)
    asm = abs(sm)
    dU = 0.25 * (ssp + ssm) * min(asp, asm)
    return q_i - dU, q_i + dU
end

function SQR(x)
    return x^2
end

function WENOZ(q_im2::T, q_im1::T, q_i::T, q_ip1::T, q_ip2::T) where {T}
    """
    Implementation of WENOZ reconstruction, taken from AthenaPK
    """

    beta_coeff = (T(13.0) / T(12.0), T(0.25))
    beta = (
        beta_coeff[1] * SQR(q_im2 + q_i - T(2) * q_im1) +
        beta_coeff[2] * SQR(q_im2 + T(3) * q_i - T(4) * q_im1),
        beta_coeff[1] * SQR(q_im1 + q_ip1 - T(2) * q_i) +
        beta_coeff[2] * SQR(q_im1 - q_ip1),
        beta_coeff[1] * SQR(q_ip2 + q_i - T(2) * q_ip1) +
        beta_coeff[2] * SQR(q_ip2 + T(3) * q_i - T(4) * q_ip1),
    )

    epsL = T(1.0e-42)
    tau_5 = abs(beta[1] - beta[3])

    indicator =
        (tau_5 / (beta[1] + epsL), tau_5 / (beta[2] + epsL), tau_5 / (beta[3] + epsL))

    f = (
        T(2.0) * q_im2 - T(7.0) * q_im1 + T(11.0) * q_i,
        -q_im1 + T(5.0) * q_i + T(2.0) * q_ip1,
        T(2.0) * q_i + T(5.0) * q_ip1 - q_ip2,
    )

    alpha = (
        T(0.1) * (T(1.0) + SQR(indicator[1])),
        T(0.6) * (T(1.0) + SQR(indicator[2])),
        T(0.3) * (T(1.0) + SQR(indicator[3])),
    )

    alpha_sum = T(6.0) * sum(alpha)
    ql_ip1 = sum(f .* alpha) / alpha_sum

    f = (
        T(2.0) * q_ip2 - T(7.0) * q_ip1 + T(11.0) * q_i,
        -q_ip1 + T(5.0) * q_i + T(2.0) * q_im1,
        T(2.0) * q_i + T(5.0) * q_im1 - q_im2,
    )

    alpha = (
        T(0.1) * (T(1.0) + SQR(indicator[3])),
        T(0.6) * (T(1.0) + SQR(indicator[2])),
        T(0.3) * (T(1.0) + SQR(indicator[1])),
    )

    alpha_sum = T(6.0) * sum(alpha)
    qr_i = sum(f .* alpha) / alpha_sum

    return qr_i, ql_ip1
end


function PPM(q_im2::T, q_im1::T, q_i::T, q_ip1::T, q_ip2::T) where {T}
    """
    Implementation of PPM limiter, taken from the AthenaPK
    """

    C2 = T(1.25)

    # Step 1: Compute differences and interface averages
    qa = q_i - q_im1
    qb = q_ip1 - q_i
    dd_im1 = T(0.5) * qa + T(0.5) * (q_im1 - q_im2)
    dd = T(0.5) * qb + T(0.5) * qa
    dd_ip1 = T(0.5) * (q_ip2 - q_ip1) + T(0.5) * qb

    dph = T(0.5) * (q_im1 + q_i) + (dd_im1 - dd) / T(6)
    dph_ip1 = T(0.5) * (q_i + q_ip1) + (dd - dd_ip1) / T(6)

    # Step 2a: Limit interpolated states
    d2qc_im1 = q_im2 + q_i - T(2) * q_im1
    d2qc = q_im1 + q_ip1 - T(2) * q_i
    d2qc_ip1 = q_i + q_ip2 - T(2) * q_ip1

    # i-1/2 adjustments
    qa_tmp = dph - q_im1
    qb_tmp = q_i - dph
    qa = T(3) * (q_im1 + q_i - T(2) * dph)
    qb = d2qc_im1
    qc = d2qc
    qd = T(0)

    if sign(qa) == sign(qb) && sign(qa) == sign(qc)
        qd = sign(qa) * min(C2 * abs(qb), min(C2 * abs(qc), abs(qa)))
    end

    dph_tmp = T(0.5) * (q_im1 + q_i) - qd / T(6)
    if qa_tmp * qb_tmp < T(0)
        dph = dph_tmp
    end

    # i+1/2 adjustments
    qa_tmp = dph_ip1 - q_i
    qb_tmp = q_ip1 - dph_ip1
    qa = T(3) * (q_i + q_ip1 - T(2) * dph_ip1)
    qb = d2qc
    qc = d2qc_ip1
    qd = T(0)

    if sign(qa) == sign(qb) && sign(qa) == sign(qc)
        qd = sign(qa) * min(C2 * abs(qb), min(C2 * abs(qc), abs(qa)))
    end

    dphip1_tmp = T(0.5) * (q_i + q_ip1) - qd / T(6)
    if qa_tmp * qb_tmp < T(0)
        dph_ip1 = dphip1_tmp
    end

    d2qf = T(6) * (dph + dph_ip1 - T(2) * q_i)
    qr_i = dph
    ql_ip1 = dph_ip1

    # Step 3: Compute cell-centered difference stencils
    dqf_minus = q_i - qr_i
    dqf_plus = ql_ip1 - q_i

    # Step 4: Apply limiters
    qa_tmp = dqf_minus * dqf_plus
    qb_tmp = (q_ip1 - q_i) * (q_i - q_im1)

    qa = d2qc_im1
    qb = d2qc
    qc = d2qc_ip1
    qd = d2qf
    qe = T(0)

    if sign(qa) == sign(qb) && sign(qa) == sign(qc) && sign(qa) == sign(qd)
        qe = sign(qd) * min(min(C2 * abs(qa), C2 * abs(qb)), min(C2 * abs(qc), abs(qd)))
    end

    qa = max(abs(q_im1), abs(q_im2))
    qb = max(max(abs(q_i), abs(q_ip1)), abs(q_ip2))

    rho = T(0)
    if abs(qd) > T(1.0e-12) * max(qa, qb)
        rho = qe / qd
    end

    tmp_m = q_i - rho * dqf_minus
    tmp_p = q_i + rho * dqf_plus
    tmp2_m = q_i - T(2) * dqf_plus
    tmp2_p = q_i + T(2) * dqf_minus

    if qa_tmp <= T(0) || qb_tmp <= T(0)
        if rho <= T(1) - T(1.0e-12)
            qr_i = tmp_m
            ql_ip1 = tmp_p
        end
    else
        if abs(dqf_minus) >= T(2) * abs(dqf_plus)
            qr_i = tmp2_m
        end
        if abs(dqf_plus) >= T(2) * abs(dqf_minus)
            ql_ip1 = tmp2_p
        end
    end

    return qr_i, ql_ip1
end


function reconstruction(
    method::Val{M},
    q_im2::T,
    q_im1::T,
    q_i::T,
    q_ip1::T,
    q_ip2::T,
) where {M,T}
    if M === :MINMOD
        return MINMOD(q_im1, q_i, q_ip1)
    elseif M === :WENOZ
        return WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2)
    elseif M === :PPM
        return PPM(q_im2, q_im1, q_i, q_ip1, q_ip2)
    else
        error("Unsupported reconstruction method: ", M)
    end
end
