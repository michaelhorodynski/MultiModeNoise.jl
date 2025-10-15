function plot_fiber(x, u_xy, x0, r; vmin=nothing, cbarlabel=nothing, cmap="viridis", norm=nothing)
    pcolormesh(x, x, u_xy, cmap=cmap, rasterized=true, vmin=vmin, norm=norm)
    colorbar(label=cbarlabel)
    gca().set_aspect("equal")
    xlim(-x0, x0)
    ylim(-x0, x0)
    
    plot(r*cos.(LinRange(0,2*π,100)), r*sin.(LinRange(0,2*π,100)), "white")

    xlabel("x (μm)")
    ylabel("y (μm)")
end