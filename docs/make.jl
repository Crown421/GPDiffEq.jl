using Documenter, GPDiffEq

makedocs(;
    sitename="GPDiffEq.jl",
    pages=[
        "Home"=> "index.md",
        "theory.md",
        "symmetries.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/Crown421/GPDiffEq.jl.git",
    devbranch = "main",
)