function compute_noise_map(X, ∂Xkl∂u, U, ϕ, δF_in_ω)
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