function cw_mmf_u_μ_ν!(dΓ̃, Γ̃, p, z)

    δβ, γ, M, ũ, μ̃, ν̃, dũ, dμ̃, dν̃, exp_p, exp_m, u, v, w, μ, ν, γvv, γww, γvw, γ_abs2_u, γ_sq_u = p
    
    ũ .= view(Γ̃, :, 1)
    μ̃ .= view(Γ̃, :, 2:M+1)
    ν̃ .= view(Γ̃, :, M+2:2*M+1)

    @. exp_p = exp(1im * δβ * z)
    @. exp_m = exp(-1im * δβ * z)
    
    @. u = exp_p * ũ
    @. v = real(u)
    @. w = imag(u)

    @tullio γvv[i,j] = γ[l,k,j,i] * v[k] * v[l]
    @tullio γww[i,j] = γ[l,k,j,i] * w[k] * w[l]
    @tullio γvw[i,j] = γ[l,k,j,i] * v[k] * w[l]

    @. γ_abs2_u = γvv + γww
    @. γ_sq_u = γvv - γww + 2im*γvw

    mul!(dũ, γ_abs2_u, u)
    @. dũ *= 1im * exp_m

    @. μ = exp_p * μ̃
    @. ν = exp_p * ν̃
    
    dμ̃ .= 1im .* exp_m .* ( 2 .* γ_abs2_u * μ .+ γ_sq_u * conj.(ν))
    dν̃ .= 1im .* exp_m .* ( 2 .* γ_abs2_u * ν .+ γ_sq_u * conj.(μ))

    dΓ̃[:,1] .= dũ
    dΓ̃[:,2:M+1] .= dμ̃
    dΓ̃[:,M+2:2*M+1] .= dν̃
    
end

function cw_mmf_u_reim!(dũ, ũ, p, z)
    
    δβ, γ, M, c, s, v, w, γvv, γww, δ_abs2_u, δ_abs2_u_v, δ_abs2_u_w, dṽ, dw̃ = p
    
    @. c = cos(δβ * z)
    @. s = sin(δβ * z)

    ṽ = view(ũ, 1:M)
    w̃ = view(ũ, M+1:2*M)
    @. v = c * ṽ - s * w̃
    @. w = s * ṽ + c * w̃

    @tullio γvv[i,j] = γ[l,k,j,i] * v[k] * v[l]
    @tullio γww[i,j] = γ[l,k,j,i] * w[k] * w[l]
    @. δ_abs2_u = γvv + γww
    mul!(δ_abs2_u_v, δ_abs2_u, v)
    mul!(δ_abs2_u_w, δ_abs2_u, w)

    @. dṽ = -c * δ_abs2_u_w + s * δ_abs2_u_v
    @. dw̃ = c * δ_abs2_u_v + s * δ_abs2_u_w
    dũ[1:M] .= dṽ
    dũ[M+1:2*M] .= dw̃
    
end

function cw_mmf_adjoint_reim!(dλ̃, λ̃, p, z)
    
    ũ, δβ, γ, M, c, s, ψ, χ, ũ_z, ṽ, w̃, v, w, γvv, γww, γvw, δ_abs2_u, δ_sq_u_re, δ_sq_u_im, δ_abs2_u_ψ, δ_abs2_u_χ, δ_sq_u_re_ψ, δ_sq_u_re_χ, δ_sq_u_im_ψ, δ_sq_u_im_χ, dψ̃, dχ̃ = p

    @. c = cos(δβ * z)
    @. s = sin(δβ * z)

    ψ̃ = view(λ̃, 1:M)
    χ̃ = view(λ̃, M+1:2*M)
    @. ψ = c * ψ̃ - s * χ̃
    @. χ = s * ψ̃ + c * χ̃

    ũ_z .= ũ(z)
    ṽ .= view(ũ_z, 1:M)
    w̃ .= view(ũ_z, M+1:2*M)
    @. v = c * ṽ - s * w̃
    @. w = s * ṽ + c * w̃

    @tullio γvv[i,j] = γ[l,k,j,i] * v[k] * v[l]
    @tullio γww[i,j] = γ[l,k,j,i] * w[k] * w[l]
    @tullio γvw[i,j] = γ[l,k,j,i] * v[k] * w[l]

    @. δ_abs2_u = γvv + γww
    @. δ_sq_u_re = γvv - γww 
    @. δ_sq_u_im = 2 * γvw

    mul!(δ_abs2_u_ψ, δ_abs2_u, ψ)
    mul!(δ_abs2_u_χ, δ_abs2_u, χ)
    mul!(δ_sq_u_re_ψ, δ_sq_u_re, ψ)
    mul!(δ_sq_u_re_χ, δ_sq_u_re, χ)
    mul!(δ_sq_u_im_ψ, δ_sq_u_im, ψ)
    mul!(δ_sq_u_im_χ, δ_sq_u_im, χ)

    @. dψ̃ = 2 * s * δ_abs2_u_ψ - 2 * c * δ_abs2_u_χ - s * δ_sq_u_re_ψ - s * δ_sq_u_im_χ + c * δ_sq_u_im_ψ - c * δ_sq_u_re_χ
    @. dχ̃ = 2 * c * δ_abs2_u_ψ + 2 * s * δ_abs2_u_χ - s * δ_sq_u_im_ψ + s * δ_sq_u_re_χ - c * δ_sq_u_re_ψ - c * δ_sq_u_im_χ
    dλ̃[1:M] .= dψ̃
    dλ̃[M+1:2*M] .= dχ̃
    
end

function get_p_cw_mmf_u_reim(δβ, γ, M, x0)
    return (δβ, γ, M, zeros(M), zeros(M), similar(x0, M), similar(x0, M), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M))
end

function get_p_cw_mmf_adjoint_reim(δβ, γ, M, x0, sol)
    return (sol, δβ, γ, M, similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, 2*M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, (M,M)), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M), similar(x0, M))
end

function get_p_cw_mmf_u_μ_ν(δβ, γ, M)
    return (δβ, γ, M, zeros(ComplexF64, M), zeros(ComplexF64, M, M), zeros(ComplexF64, M, M), zeros(ComplexF64, M), zeros(ComplexF64, M, M), zeros(ComplexF64, M, M), zeros(ComplexF64, M), zeros(ComplexF64, M), zeros(ComplexF64, M), zeros(M), zeros(M), zeros(ComplexF64, M, M), zeros(ComplexF64, M, M), zeros(M, M), zeros(M, M), zeros(M, M), zeros(M, M), zeros(ComplexF64, M, M))
end

function rf_to_lab_reim(ũ, δβ, z)
    c = diagm(cos.(δβ * z))
    s = diagm(sin.(δβ * z))
    O = [[c -s]; [s c]]
    return O * ũ
end

function lab_to_rf_reim(u, δβ, z)
    c = diagm(cos.(δβ * z))
    s = diagm(sin.(δβ * z))
    O = [[c -s]; [s c]]
    return O' * u
end

function solve_cw_mmf(u0::Vector{ComplexF64}, fiber, sim)
    M = sim["M"]
    zsave = fiber["zsave"]
    δβ = fiber["dbeta"]
    Γ0 = [u0 diagm(ones(M)) zeros(M,M)]
    p_mmf_u_μ_ν = get_p_cw_mmf_u_μ_ν(δβ, fiber["gamma"], M)
    prob_mmf_u_μ_ν = ODEProblem(cw_mmf_u_μ_ν!, Γ0, (0, fiber["L"]), p_mmf_u_μ_ν)

    if isnothing(fiber["zsave"])
        println("Error: No zsave")
        return nothing
    else
        sol_ũ_μ̃_ν̃ = solve(prob_mmf_u_μ_ν, Tsit5(), reltol=1e-5, saveat=zsave)

        uz = zeros(ComplexF64, length(zsave), M)
        μz = zeros(ComplexF64, length(zsave), M, M)
        νz = zeros(ComplexF64, length(zsave), M, M)

        for i in 1:length(zsave)
            uz[i,:] = exp.(1im*δβ*sol_ũ_μ̃_ν̃.t[i]) .* sol_ũ_μ̃_ν̃.u[i][:,1]
            μz[i,:,:] = exp.(1im*δβ*sol_ũ_μ̃_ν̃.t[i]) .* sol_ũ_μ̃_ν̃.u[i][:,2:M+1]
            νz[i,:,:] = exp.(1im*δβ*sol_ũ_μ̃_ν̃.t[i]) .* sol_ũ_μ̃_ν̃.u[i][:,M+2:2*M+1]
        end

        return Dict("uz" => uz, "μz" => μz, "νz" => νz)
    end
end

function solve_∇n_modes_cw_mmf_adjoint_reim(u0_reim, p; return_u=false)
    mode, fiber, sim = p
    δβ = fiber["dbeta"]
    γ = fiber["gamma"]
    M = sim["M"]
    L = fiber["L"]
    
    p_u_reim = get_p_cw_mmf_u_reim(δβ, γ, M, u0_reim)
    prob_u_reim = ODEProblem(cw_mmf_u_reim!, u0_reim, (0, L), p_u_reim)
    sol_ũ_reim = solve(prob_u_reim, Tsit5(), reltol=1e-5)
    uf_reim = rf_to_lab_reim(sol_ũ_reim(L), δβ, L)

    p_adj_reim = get_p_cw_mmf_adjoint_reim(δβ, γ, M, u0_reim, sol_ũ_reim)
    λL = uf_reim .* [diagm(ones(M)); diagm(ones(M))][:,mode]
    λ̃L = lab_to_rf_reim(λL, δβ, L)
    prob_adj_reim = ODEProblem(cw_mmf_adjoint_reim!, λ̃L, (L, 0), p_adj_reim)
    sol_adj_reim = solve(prob_adj_reim, Tsit5(), reltol=1e-5)
    λ0 = sol_adj_reim(0)
    
    if return_u == false
        return λ0
    elseif return_u == true
        return uf_reim, λ0
    end
end

function solve_∇n_pixels_cw_mmf_adjoint_reim(u0_reim, p; return_u=false)
    pixel_indices, fiber, sim = p
    δβ = fiber["dbeta"]
    γ = fiber["gamma"]
    ϕ = fiber["phi"]
    M = sim["M"]
    L = fiber["L"]
    δF_in = sim["dF_in"]
    ϕ_pixel = ϕ[pixel_indices,:]
    
    p_u_reim = get_p_cw_mmf_u_reim(δβ, γ, M, u0_reim)
    prob_u_reim = ODEProblem(cw_mmf_u_reim!, u0_reim, (0, L), p_u_reim)
    sol_ũ_reim = solve(prob_u_reim, Tsit5(), reltol=1e-5)
    uf_reim = rf_to_lab_reim(sol_ũ_reim(L), δβ, L)

    p_adj_reim = get_p_cw_mmf_adjoint_reim(δβ, γ, M, u0_reim, sol_ũ_reim)
    λL = 2 * [(ϕ_pixel * uf_reim[1:M])' * ϕ_pixel (ϕ_pixel * uf_reim[M+1:2*M])' * ϕ_pixel][1,:]
    λ̃L = lab_to_rf_reim(λL, δβ, L)
    prob_adj_reim = ODEProblem(cw_mmf_adjoint_reim!, λ̃L, (L, 0), p_adj_reim)
    sol_adj_reim = solve(prob_adj_reim, Tsit5(), reltol=1e-5) # Vern9()
    λ0 = sol_adj_reim(0)
    
    if return_u == false
        return λ0
    elseif return_u == true
        return uf_reim, λ0
    end
end