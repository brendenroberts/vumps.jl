#!/usr/bin/env julia
using Distributions,JLD,CSV
include("vumps.jl")

function polarU(A::AbstractMatrix{T})::Matrix{T} where {T<:Number}
    M = similar(A);
    polarU!(M,A);
    return M;
end

function do_vumps(params::Vector{Float64},
                  Td::Dict{String,Vector{Array{Float64,3}}},
                  N::Int64,
                  d::Int64,
                  m::Int64)
    # Simulation parameters
    M::Int64 = 12;         # MPO bond dim
    conv::Float64 = 1e-10; # overall convergence traget

    # use Cartan-Weyl operators to specify MPO
    w::Complex{Float64} = exp(2.0im*π/3.);
    X::Matrix{Float64} = [ 0. 0. 1. ; 1. 0. 0. ; 0. 1. 0. ];
    Z::Matrix{Complex{Float64}} = [ 1. 0. 0. ; 0. w 0. ; 0. 0. w*w ];
    T3::Matrix{Float64} = [ 1. 0. 0. ; 0. -1. 0. ; 0. 0. 0. ]/2.;
    T8::Matrix{Float64} = [ 1. 0. 0. ; 0. 1. 0. ; 0. 0. -2. ]/(2.0*sqrt(3.0));
    Ip::Matrix{Float64} = [ 0. 1. 0. ; 0. 0. 0. ; 0. 0. 0. ];
    Im::Matrix{Float64} = [ 0. 0. 0. ; 1. 0. 0. ; 0. 0. 0. ];
    Up::Matrix{Float64} = [ 0. 0. 1. ; 0. 0. 0. ; 0. 0. 0. ];
    Um::Matrix{Float64} = [ 0. 0. 0. ; 0. 0. 0. ; 1. 0. 0. ];
    Vp::Matrix{Float64} = [ 0. 0. 0. ; 0. 0. 1. ; 0. 0. 0. ];
    Vm::Matrix{Float64} = [ 0. 0. 0. ; 0. 0. 0. ; 0. 1. 0. ];

    # Initialize MPO tensor
    delta::Float64 = params[1];
    K::Float64 = params[2];
    Jx::Float64 = 1.0-delta;
    Jz::Float64 = 1.0+delta;
    W::Array{Float64,4} = zeros((M,d,d,M));
    W[ 1,:,:, 1] += I;
    W[ 1,:,:, 2] = X;
    W[ 1,:,:, 3] = X';
    W[ 1,:,:, 4] = T3;
    W[ 1,:,:, 5] = T8;
    W[ 1,:,:, 6] = Ip;
    W[ 1,:,:, 7] = Im;
    W[ 1,:,:, 8] = Up;
    W[ 1,:,:, 9] = Um;
    W[ 1,:,:,10] = Vp;
    W[ 1,:,:,11] = Vm;
    W[ 2,:,:,12] = -Jx*X;
    W[ 3,:,:,12] = -Jx*X';
    W[ 4,:,:,12] = -6.0*(Jz+K)*T3;
    W[ 5,:,:,12] = -6.0*(Jz+K)*T8;
    W[ 6,:,:,12] = -3.0*K*Ip;
    W[ 7,:,:,12] = -3.0*K*Im;
    W[ 8,:,:,12] = -3.0*K*Up;
    W[ 9,:,:,12] = -3.0*K*Um;
    W[10,:,:,12] = -3.0*K*Vp;
    W[11,:,:,12] = -3.0*K*Vm;
    W[12,:,:,12] += I;
   
    # Initialization
    if(length(Td) == 0)
        Td["AL"] = [zeros(Float64,m,d,m) for i=1:N];
        Td["AR"] = [zeros(Float64,m,d,m) for i=1:N];
        Td["C"]  = [randn(Float64,(m,1,m)) for i=1:N];
 
        for (l,r,c) in zip(Td["AL"],Td["AR"],Td["C"])
            C = reshape(c,(m,m));
            A = randn(Float64,m,d,m);

            UAL = polarU(reshape(A,(m*d,m)));
            UAR = polarU(reshape(A,(m,d*m)));
            UC = polarU(C);

            l .= reshape(UAL*UC',(m,d,m));
            r .= reshape(UC'*UAR,(m,d,m));
        end
    end

    for i = 1:N
        @tensor Res[c,d] := (Td["AL"][i])[a,b,c]*conj.(Td["AL"][i])[a,b,d];
        @assert (norm(Res-I) < 1e-12) "AL tensor not orthogonal!";
        @tensor Res[a,d] := (Td["AR"][i])[a,b,c]*conj.(Td["AR"][i])[d,b,c];
        @assert (norm(Res-I) < 1e-12) "AR tensor not orthogonal!";
    end
    
    # filename convention
    handle = @sprintf("%s_%s_delta%.7f_K%.7f_m%03d",ARGS[5],ARGS[4],delta,K,m);
    
    # observables to track during vumps convergence
    measOps = [HermitianOp{Float64}("Re(X)",[totype(Val(Float64),(X+X')/2.0)]),
               HermitianOp{Float64}("Im(X)",[totype(Val(Float64),(X-X')/2.0im)]),
               HermitianOp{Float64}("Re(Z)",[totype(Val(Float64),(Z+Z')/2.0)]),
               HermitianOp{Float64}("Im(Z)",[totype(Val(Float64),(Z-Z')/2.0im)])];
    
    (E,λ,S,tLS,tEV,_),t,_,_,_ = @timed vumps(W,Td,measOps,conv,handle);
    @printf(stdout,"elapsed %.1f s, linear solver %.1f s, eigensolver %.1f s (delta=%.7f, K=%.7f)\n",t,tLS,tEV,delta,K);

    save(handle*".jld",Td);

    f1 = open(ARGS[5],"a");
    @printf(f1,"%.7f %.7f %s %03d %13.10f ",delta,K,ARGS[4],m,sum(E)/N/N);
    join(f1,[@sprintf("%10.8f",s) for s in S]," ");
    println(f1);
    close(f1);

    f2 = open(ARGS[5]*"_tm","a");
    for tmEig in λ
        @printf(f2,"%.8f",real(tmEig));
        if imag(tmEig) >= 0.0
            @printf(f2,"+%.8fj",imag(tmEig));
        else
            @printf(f2,"%.8fj",imag(tmEig));
        end
        
        if tmEig != λ[end]
            print(f2,",");
        else
            println(f2);
        end
    end
    close(f2);

    return nothing;
end

function scan()
    # Simulation parameters
    d::Int64 = 3;                    # local Hilbert space dimension
    N::Int64 = parse(Int64,ARGS[1]); # unit cell size
    m::Int64 = parse(Int64,ARGS[2]); # MPS bond dimension

    # Non-IEEE behavior for super small numbers, OK for TN calcs
    set_zero_subnormals(true);
    
    # Initialize MPS tensors inside dict for ease of writing as JLD (HDF5)
    # Dummy index added to C for stronger typing
    Td = Dict{String,Vector{Array{Float64,3}}}();  
    
    if(length(ARGS) == 6)
        Td::Dict{String,Vector{Array{Float64,3}}} = load(ARGS[6]);
        println(stdout,"loaded initial data from $(ARGS[6])");
    end
 
    f1 = open(ARGS[5],"a");
    print(f1,"# delta K tag M E S1 ... SN\n");
    close(f1);
 
    for row in CSV.File(ARGS[3],delim=',',header=1,comment="#")
        do_vumps([row.d,row.K],Td,N,d,m);
    end
    
    return nothing;
end

#-------------------------------
if(abspath(PROGRAM_FILE) == @__FILE__)
    if(length(ARGS) != 5 && length(ARGS) != 6)
        println(stdout,"usage: $PROGRAM_FILE N_uc chi pfile tag fname (init.jld)");
    else
        scan();
    end
end
