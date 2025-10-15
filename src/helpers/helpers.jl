function meshgrid(x, y)
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(reshape(y, :, 1), 1, length(x))
    return X, Y
end

function lin_to_dB(x)
    return 10*log10(x)
end

function get_center_square_pixels(N, k, center)
    # Calculate the center of the grid
    center_x = center[1]
    center_y = center[2]
    
    # Calculate the starting and ending indices based on k
    start_x = max(center_x - div(k-1, 2), 1)
    end_x = min(center_x + div(k, 2), N)
    start_y = max(center_y - div(k-1, 2), 1)
    end_y = min(center_y + div(k, 2), N)
    
    # Collect the k x k pixels
    pixels = [[i, j] for i in start_x:end_x for j in start_y:end_y]
    
    return pixels
end

function get_center_square_ids_flat(XX, YY, pixels)
    return [argmin(vec((XX .- XX[pixels[i][2],pixels[i][1]]).^2 + (YY .- YY[pixels[i][2],pixels[i][1]]).^2)) for i in 1:length(pixels)]
end

function gauss(XX, YY, σ)
    return -exp.(-(XX.^2 + YY.^2)/(2*σ^2)) / sqrt(sum(abs2.(exp.(-(XX.^2 + YY.^2)/(2*σ^2)))))
end
    
function insert_matrix_into_center(large_matrix, small_matrix)
    # Get the dimensions of the large and small matrices
    large_rows, large_cols = size(large_matrix)
    small_rows, small_cols = size(small_matrix)
    
    # Calculate the starting indices to place the small matrix in the center of the large matrix
    start_row = div(large_rows - small_rows, 2)
    start_col = div(large_cols - small_cols, 2)
    
    # Create a copy of the large matrix to avoid modifying the original
    augmented_matrix = copy(large_matrix)
    
    # Insert the small matrix into the center of the large matrix
    augmented_matrix[start_row+1:start_row+small_rows, start_col+1:start_col+small_cols] = small_matrix
    
    return augmented_matrix
end
    
function fourier_coefs_to_u0(x, fiber, sim, k_grid_width, k_grid_height, beam; fourier_zoom=1, shift=0)
    nx = length(fiber["x"])
    ϕ = fiber["phi"]
    P = sim["P"]
    #k_grid_width, k_grid_height, nx, ϕ, modes_smf, P = p
    
    fourier_coeffs = reshape(x, (k_grid_width, k_grid_height))
    fourier_host = insert_matrix_into_center(zeros((nx,nx)), fourier_coeffs)
    temp_ft = ifftshift(fourier_host)
    temp_ft = fft(temp_ft)

    if fourier_zoom == 1
        slm_pattern = angle.(fftshift(temp_ft))
    else
        slm_pattern = angle.(fftshift(temp_ft))/π
        slm_pattern = π*(mod.(fourier_zoom*slm_pattern .+ 1, 2) .- 1)
    end
    
    beam_after_slm = beam.*exp.(1im*slm_pattern)
    u0_norm = normalize(sum( reshape(circshift(beam_after_slm, (0,shift)), nx^2) .* ϕ, dims=1)[1,:])
    u0 = √P*u0_norm
    return u0
end

function get_disp_sim_params(λ0, M, Nt, time_window, β_order)
    c0 = 2.99792458e8 # m/s
    h = 6.62607e-34 # Js
    f0 = c0/λ0/1e12 # THz
    Δt = time_window/Nt
    ts = 1e-12*[-time_window/2+i*Δt for i in 0:Nt-1] # s
    fs = f0 .+ fftshift(fftfreq(Nt, 1/Δt)) # THz
    ω0 = 2*π*f0 # rad / ps
    ωs = 2*π*fs # rad / ps
    ε = 1e-12*Δt/(h*1e12*f0)

    r_attenuation = 0.85 * time_window/2
    n_attenuation = 30
    σ_attenuation = r_attenuation/log(2)^(1/n_attenuation)
    r_hm = σ_attenuation*log(2)^(1/n_attenuation)
    attenuator = exp.(-(abs.(1e12*ts)/σ_attenuation).^n_attenuation) * ones(M)';

    return Dict("λ0" => λ0, "f0" => f0, "M" => M, "Nt" => Nt, "time_window" => time_window, "Δt" => Δt, "ts" => ts, "fs" => fs, "ω0" => ω0, 
        "ωs" => ωs, "attenuator" => attenuator, "c0" => c0, "h" => h, "ε" => ε, "β_order" => β_order)
end

function get_cw_sim_params(λ0, M, P, δF_in)
    c0 = 2.99792458e8 # m/s
    h = 6.62607e-34 # Js
    f0 = c0/λ0/1e12 # THz
    ω0 = 2*π*f0 # rad / ps

    return Dict("λ0" => λ0, "f0" => f0, "M" => M, "ω0" => ω0, "c0" => c0, "h" => h, "P" => P, "dF_in" => δF_in)
end

function get_disp_fiber_params(L, radius, core_NA, alpha, nx, sim, fiber_fname; spatial_window = 100, fR = 0.18, τ1 = 12.2, τ2 = 32)
    Δt = sim["Δt"]
    ts = sim["ts"]
    one_m_fR = (1 - fR)
    hRt = fR*Δt*1e3*(τ1^2 + τ2^2)/(τ1*τ2^2) * exp.(-ts*1e15/τ2) .* sin.(ts*1e15/τ1) .* (sign.(ts) .+ 1)/2
    hRω = fft(hRt)

    f0 = sim["f0"]
    c0 = sim["c0"]
    M = sim["M"]
    Nt = sim["Nt"]
    β_order = sim["β_order"]

    if isfile(fiber_fname) == true
        println("Load fiber params"); flush(stdout)
        fiber = npzread(fiber_fname)
        γ = fiber["gamma"]
        ϕ = fiber["phi"]
        Dω = fiber["D_w"]
        x = fiber["x"]
        βn_ω = fiber["betas"]
    else
        println("Compute fiber params"); flush(stdout)
        βn_ω, Dω, γ, ϕ, x = get_params(f0, c0, nx, spatial_window, radius, core_NA, alpha, M, Nt, Δt, β_order)
        npzwrite(fiber_fname, Dict("gamma" => γ, "phi" => ϕ, "x" => x, "D_w" => Dω, "betas" => βn_ω))
    end

    return Dict("ϕ" => ϕ, "Dω" => Dω, "γ" => γ, "L" => L, "hRω" => hRω, "one_m_fR" => one_m_fR, "zsave" => nothing, "x" => x)
end

function get_cw_fiber_params(L, radius, core_NA, alpha, nx, sim, fiber_fname; spatial_window = 100)

    f0 = sim["f0"]
    c0 = sim["c0"]
    M = sim["M"]

    if isfile(fiber_fname) == true
        println("Load fiber params"); flush(stdout)
        fiber = npzread(fiber_fname)
        γ = fiber["gamma"]
        ϕ = fiber["phi"]
        x = fiber["x"]
        δβ = fiber["dbeta"]
    else
        println("Compute fiber params"); flush(stdout)
        δβ, γ, ϕ, x = get_params(f0, c0, nx, spatial_window, radius, core_NA, alpha, M)
        npzwrite(fiber_fname, Dict("gamma" => γ, "phi" => ϕ, "x" => x, "dbeta" => δβ))
    end
    
    return Dict("phi" => ϕ, "dbeta" => δβ, "gamma" => γ, "L" => L, "zsave" => nothing, "x" => x)
end