function compute_disp_spatial_n_var_n(X, ∂Xkl∂u, U, ϕ, δF_in_ω)
    @tullio Φ[i] := ϕ[i,k] * ϕ[i,k]
    @tullio no_derivative_term[i] := X[i] * (1 - Φ[i])
    @tullio Xklmn[k,l,m,n] := conj(∂Xkl∂u[k,l,ω,j]) * ∂Xkl∂u[m,n,ω,j]
    @tullio shot_noise[i] := ϕ[i,k] * ϕ[i,l] * ϕ[i,m] * ϕ[i,n] * Xklmn[k,l,m,n]
    @tullio ∂Xkl∂u_U[k,l,ω] := ∂Xkl∂u[k,l,ω,j] * U[j]
    @tullio XUklmn[k,l,m,n] := δF_in_ω[ω] * conj(∂Xkl∂u_U[k,l,ω]) * ∂Xkl∂u_U[m,n,ω]
    @tullio excess_noise[i] := ϕ[i,k] * ϕ[i,l] * ϕ[i,m] * ϕ[i,n] * XUklmn[k,l,m,n]
    var_X = real.(no_derivative_term + shot_noise + excess_noise)
    return var_X
end

function compute_cw_spatial_n_var_n(u0, uf, μf, νf, δF_in, ϕ, P)
    α = ϕ * uf
    Uk1 = u0/√P
    ∂ni∂uk = α .* (ϕ * conj.(νf)) + conj.(α) .* (ϕ * μf)
    no_derivative_term = abs2.(α) .* (1 .- sum(abs2.(ϕ), dims=2))
    shot_noise = sum(abs2.(∂ni∂uk), dims=2)
    excess_noise = δF_in * abs2.(∂ni∂uk * Uk1)
    return abs2.(α), no_derivative_term + shot_noise + excess_noise
end

function compute_cw_superpixel_n_var_n(u0, uf, μf, νf, pixel_indices, δF_in, ϕ, P)
    ϕ_sp = ϕ[pixel_indices,:]
    
    α_sp = ϕ_sp * uf
    U = u0/√P
    
    ∂ni∂uk = α_sp .* (ϕ_sp * conj.(νf)) + conj.(α_sp) .* (ϕ_sp * μf)
    no_derivative_term = α_sp' * α_sp - sum(abs2.(ϕ_sp' * α_sp))
    shot_noise = sum(abs2.(sum(∂ni∂uk, dims=1)))
    excess_noise = δF_in * abs2.(sum(∂ni∂uk * U))

    return real(α_sp' * α_sp), real(no_derivative_term + shot_noise + excess_noise)
end

function compute_cw_modal_n_var_n(u0, uf, μf, νf, δF_in, P)
    U = u0/√P
    ∂ni∂uj = uf .* conj.(νf) + conj.(uf) .* μf
    return abs2.(uf), sum(abs2.(∂ni∂uj), dims=2)[:,1] + δF_in*abs2.(∂ni∂uj * U)
end