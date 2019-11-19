using Documenter, JuliaKitchenSink

makedocs(;
    modules=[JuliaKitchenSink],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/daryoush/JuliaKitchenSink.jl/blob/{commit}{path}#L{line}",
    sitename="JuliaKitchenSink.jl",
    authors="Daryoush Mehrtash <dmehrtash@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/daryoush/JuliaKitchenSink.jl",
)
