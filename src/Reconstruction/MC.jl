@inline function MC(q_im2,q_im1,q_i,q_ip1,q_ip2)
    @inbounds begin
	    Dqm = 2.0 * (q_i - q_im1)
	    Dqp = 2.0 * (q_ip1 - q_i)
	    Dqc = 0.5 * (q_ip1 - q_im1)
	    s = Dqm * Dqp
	    if s <= 0.0
		dq = 0.0
	    else
		if abs(Dqm) < abs(Dqp) && abs(Dqm) < abs(Dqc)
		    dq = Dqm
		elseif abs(Dqp) < abs(Dqc)
		    dq = Dqp
		else
		    dq = Dqc
		end
	    end
	    qr_i    = q_i + 0.5 * dq
	    ql_ip1  = q_i - 0.5 * dq

	    return qr_i, ql_ip1
    end
end

