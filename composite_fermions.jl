import Pkg
Pkg.instantiate()
for package in ["LinearAlgebra", "Zygote", "AdvancedHMC", "LogDensityProblems", "DelimitedFiles"]
	Pkg.add(package)
end
using LinearAlgebra, Zygote, Random, AdvancedHMC, LogDensityProblems, DelimitedFiles
const global P::Int64 = 1
const global n::Int64 = 2
const global nchains::Int64 = Threads.nthreads()
const global n_samples::Int64 = div(4096, nchains)
const global n_adapts::Int64 = 256

# function fromroots(roots::AbstractVector{T}) where {T<:Number}
#     n = length(roots)
#     c = zeros(T, (n + 1))
#     c[1] = one(T)
#     for j in 1:n, i in j:-1:1
#         setindex!(c, c[(i+1)] - roots[j] * c[i], i + 1)
#     end
#     return reverse(c)
# end

### Implementation copied from source code of Polynomials.jl
function fromroots(roots)
    n = length(roots)
    c = zeros(ComplexF64, (n + 1))
    c[1] = one(ComplexF64)
    for j in 1:n, i in j:-1:1
        setindex!(c, c[(i+1)] - roots[j] * c[i], i + 1)
    end
    return reverse(c)
end


Zygote.@adjoint fromroots(roots) = fromroots(roots), c̄ -> (reduce(hcat, [-1 .* append!(fromroots(roots[1:end.!=i]), zero(ComplexF64)) for i in eachindex(roots)])' * c̄,)


# function derivative(z, coeffs::AbstractVector{T}, order) where {T<:Number}
#     order == 0 && return evalpoly(z, coeffs)
#     order > length(coeffs) - 1 && return zero(T)
#     return evalpoly(z, [reduce(*, (i-order):i-1, init=coeffs[i]) for i in eachindex(coeffs)])
# end

### Implementation copied from source code of Polynomials.jl
function derivative(z, coeffs, order)
    order == 0 && return evalpoly(z, coeffs)
    order > length(coeffs) - 1 && return zero(ComplexF64)
    return evalpoly(z, [reduce(*, (i-order):i-1, init=coeffs[i]) for i in eachindex(coeffs)])
end

# function wavefunction(Z::AbstractVector{T}, N, nlist, p) where {T<:Number} ### cumsum can be given to function.
#     slater_elems = Zygote.Buffer(zeros(T, N, N))
#     for i in 1:N
#         poly::AbstractVector{T} = fromroots(repeat(Z[1:end.!=i], p))
#         iter_count::Int64 = 1
#         for n in eachindex(nlist)
#             deriv_stored::T = derivative(Z[i], poly, n - 1)
#             for j in 1:nlist[n]
#                 slater_elems[iter_count, i] = deriv_stored * Z[i]^(j - n)
#                 iter_count = iter_count + 1
#             end
#         end
#     end
#     return logdet(copy(slater_elems)) - dot(Z, Z) / 4
# end
function wavefunction(Z, N, nlist, p) 
    slater_elems = Zygote.Buffer(zeros(ComplexF64, N, N))
    for i in 1:N
        poly::AbstractVector{ComplexF64} = fromroots(repeat(Z[1:end.!=i], p))
        iter_count::Int64 = 1
        for n in eachindex(nlist)
            deriv_stored::ComplexF64 = derivative(Z[i], poly, n - 1)
            for j in 1:nlist[n]
                slater_elems[iter_count, i] = deriv_stored * Z[i]^(j - n)
                iter_count = iter_count + 1
            end
        end
    end
    return logdet(copy(slater_elems)) - dot(Z, Z) / 4
end

# function wavefunction(Z, N, nlist, p) 
#     slater_elems = Zygote.Buffer(zeros(ComplexF64, N-1, N-1))
#     for i in 1:N-1
#         poly::AbstractVector{ComplexF64} = fromroots(repeat(append!(Z[1:end.!=i], zero(ComplexF64)), p))
#         iter_count::Int64 = 1
#         for n in eachindex(nlist)
#             deriv_stored::ComplexF64 = derivative(Z[i], poly, n - 1)
#             for j in 1:nlist[n]
#                 slater_elems[iter_count, i] = deriv_stored * Z[i]^(j - n)
#                 iter_count = iter_count + 1
#             end
#         end
#     end
#     return logdet(copy(slater_elems)) - dot(Z, Z) / 4
# end


struct LogTargetDensity
    dim::Int
end

function LogDensityProblems.logdensity(p::LogTargetDensity, Z)
    N::Int64 = div(p.dim, 2)
    return 2 * real(wavefunction(Z[begin:2:end] + 1.0im * Z[begin+1:2:end], N, fill(div(N, n), n), P))
end

LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

function cf_sampler(N::Int64)
    RN::Float64 = sqrt(2 * N * (2 * n * P + 1) / n)
    ℓπ = LogTargetDensity(2 * N)

    metric = DiagEuclideanMetric(2 * N)
    hamiltonian = Hamiltonian(metric, ℓπ, Zygote)
    chains = Vector{Any}(undef, nchains)

    time1 = time_ns()
    Threads.@threads for i in 1:nchains
        initial_z = Array{Float64,1}(undef, 2 * N)
        for i in range(0, N - 1)
            r::Float64 = rand(Float64) * RN + 0.5 * RN
            arr = randn(Float64, 2)
            arr = arr / norm(arr)
            initial_z[2*i+1] = r * arr[1]
            initial_z[2*i+2] = r * arr[2]
        end

        initial_ϵ = find_good_stepsize(hamiltonian, initial_z)
        integrator = Leapfrog(initial_ϵ)

        proposal = NUTS{MultinomialTS,GeneralisedNoUTurn}(integrator)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        samples, stats = sample(hamiltonian, proposal, initial_z, n_samples, adaptor, n_adapts; progress=false, verbose=false, drop_warmup=false)

        chains[i] = map(x -> x[begin:2:end] + 1.0im * x[begin+1:2:end], samples)
    end
    time2 = time_ns()
    delta = 1e-9 * (time2 - time1)
    open("./times.csv", "a") do file
    	print(file, "$N,$delta\n")
    end
    final_data = collect(Iterators.flatten(chains))
#     println(final_data)
    writedlm("./data_$N.csv", final_data, ',')
    return
end


for N in 4:2:64
    cf_sampler(N)
end
# cf_sampler(4)
