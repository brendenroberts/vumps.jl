# vumps implementation of multi-site sequential algorithm without symmetries
# specializes at compile time to either real- or complex-valued MPS/MPO tensors
# Brenden Roberts, 2019
using LinearAlgebra,TensorOperations,LinearMaps,Arpack,IterativeSolvers,Printf

struct uMPS{T<:Number}
    N::Int64;
    m::Int64;
    d::Int64;
    A::NamedTuple{(:L,:R),Tuple{Vector{Array{T,3}},Vector{Array{T,3}}}};
    B::NamedTuple{(:L,:R),Tuple{Vector{Array{T,3}},Vector{Array{T,3}}}};
    C::Vector{Array{T,2}};
    AC::Vector{Array{T,3}}; 
    ρL::Vector{Array{T,2}};
    ρR::Vector{Array{T,2}};
    vv::Vector{Array{T,2}};
    ping::Array{Int64,1};
end

struct FixedPtTen{T<:Number}
    N::Int64;
    m::Int64;
    M::Int64;
    L::Vector{Array{T,3}};
    R::Vector{Array{T,3}}; 
    YL::Vector{Array{T,2}};
    YR::Vector{Array{T,2}};
    HL::Vector{Array{T,1}};
    HR::Vector{Array{T,1}};
    vv::Vector{Array{T,3}};
    ping::Array{Int64,1};
end

struct Hamiltonian{T<:Number}
    M::Int64;
    L::Vector{Array{T,3}};
    R::Vector{Array{T,3}};
    W::Array{T,4};
end

struct HermitianOp{T<:Number}
    name::String;
    op::Vector{Matrix{T}};
end

struct Observables{T<:Number}
    op::Vector{HermitianOp{T}};
    ev::Vector{Float64};
    vv::Vector{Array{T,2}};
    ping::Array{Int64,1};
end

@inline function to(obj::Union{uMPS{T},Observables{T}})::Array{T,2} where {T<:Number}
    @inbounds return obj.vv[2-obj.ping[1]];
end

@inline function from(obj::Union{uMPS{T},Observables{T}})::Array{T,2} where {T<:Number}
    @inbounds return obj.vv[obj.ping[1]+1];
end

@inline function to(obj::FixedPtTen{T})::Array{T,3} where {T<:Number}
    @inbounds return obj.vv[2-obj.ping[1]];
end

@inline function from(obj::FixedPtTen{T})::Array{T,3} where {T<:Number}
    @inbounds return obj.vv[obj.ping[1]+1];
end

@inline function pingpong(obj::Union{uMPS,FixedPtTen,Observables})
    @inbounds obj.ping[1] = 1-obj.ping[1];
    return nothing;
end

totype(::Val{<:Real} , A::AbstractArray{T}) where {T<:Number} = real.(A);
totype(::Val{T} , A::AbstractArray{<:Number}) where {T<:Complex} = T.(A);

function polarU!(R::Matrix{T},A::AbstractMatrix{T}) where {T<:Number}
    (U,~,Vt) = LAPACK.gesvd!('S','S',Matrix(A));
    R .= U*Vt;
end

@inline function tl_mult(w::AbstractArray{T,2} , v::AbstractArray{T,2} , A::Array{T,3} , B::Array{T,3}) where {T<:Number}
    @tensor w[δ,γ] = A[α,σ,δ]*v[α,β]*B[γ,σ,β];
    return nothing;
end

@inline function tr_mult(w::AbstractArray{T,2} , v::AbstractArray{T,2} , A::Array{T,3} , B::Array{T,3}) where {T<:Number}
    @tensor w[α,β] = A[α,σ,δ]*v[δ,γ]*B[γ,σ,β];
    return nothing;
end

@inline function TL(w::AbstractVector{T} , v::AbstractVector{T} , psi::uMPS{T} , lr::Symbol , seq::Vector{Int64}) where {T<:Number}
    for j in seq
        @inbounds tl_mult(j == seq[end] ? reshape(w,psi.m,psi.m) : to(psi) ,
                          j == seq[1]   ? reshape(v,psi.m,psi.m) : from(psi) , psi.A[lr][j] , psi.B[lr][j]);
        pingpong(psi);
    end
   
    return nothing;
end

@inline function TR(w::AbstractVector{T} , v::AbstractVector{T} , psi::uMPS{T} , lr::Symbol , seq::Vector{Int64}) where {T<:Number}
    for j in seq
        @inbounds tr_mult(j == seq[end] ? reshape(w,psi.m,psi.m) : to(psi) ,
                          j == seq[1]   ? reshape(v,psi.m,psi.m) : from(psi) , psi.A[lr][j] , psi.B[lr][j]);
        pingpong(psi);
    end
   
    return nothing;
end

@inline function tl_MPO_mult(w::AbstractArray{T,3},
                             v::AbstractArray{T,3},
                             A::Array{T,3},
                             B::Array{T,3},
                             W::AbstractArray{T,4}) where {T<:Number}
    @tensor w[δ,γ,ν] = (A[α,σ,δ]*v[α,β,μ])*W[μ,σ,ρ,ν]*B[γ,ρ,β];
    return nothing;
end

@inline function tr_MPO_mult(w::AbstractArray{T,3},
                             v::AbstractArray{T,3},
                             A::Array{T,3},
                             B::Array{T,3},
                             W::AbstractArray{T,4}) where {T<:Number}
    @tensor w[α,β,μ] = (A[α,σ,δ]*v[δ,γ,ν])*W[μ,σ,ρ,ν]*B[γ,ρ,β];
    return nothing;
end

@inline function TL_MPO(i::Int64,
                        a::Int64,
                        psi::uMPS{T},
                        F::FixedPtTen{T},
                        W::Array{T,4},
                        seq::Vector{Int64}) where {T<:Number}
    @inbounds Wa = view(W,1:a,:,:,1:a);
    
    for j in seq
        @inbounds tl_MPO_mult(view(to(F),:,:,1:a) ,
                              view(j == seq[1] ? F.L[i] : from(F),:,:,1:a) ,
                              psi.A[:L][j] , psi.B[:L][j] , Wa);
        pingpong(F);
    end

    @inbounds F.L[i][:,:,a] .= view(from(F),:,:,a);
end

@inline function TR_MPO(i::Int64,
                        a::Int64,
                        psi::uMPS{T},
                        F::FixedPtTen{T},
                        W::Array{T,4},
                        seq::Vector{Int64}) where {T<:Number}
    M::Int64 = F.M;
    @inbounds Wa = view(W,a:M,:,:,a:M);
    
    for j in seq
        @inbounds tr_MPO_mult(view(to(F),:,:,a:M) ,
                              view(j == seq[1] ? F.R[i] : from(F),:,:,a:M) ,
                              psi.A[:R][j] , psi.B[:R][j] , Wa);
        pingpong(F);
    end

    @inbounds F.R[i][:,:,a] .= view(from(F),:,:,a);
end

@inline function dotu(a::AbstractArray,b::AbstractArray)
    l = length(a);
    @assert (l == length(b)) "Incompatible vectors in dotu!";
    ret = 0.0;
    for i = 1:l
        ret += a[i]*b[i];
    end
    
    return ret;
end

@inline function addDiag!(A::AbstractMatrix{T},x::T2) where {T<:Number,T2<:Number}
    m = minimum(size(A));
    for j = 1:m
        @inbounds A[j,j] += x;
    end

    return nothing;
end

@inline function ImTL(w::AbstractVector{T},
                      v::AbstractVector{T},
                      i::Int64,
                      psi::uMPS{T},
                      seq::Vector{Int64}) where {T<:Number}
    TL(w,v,psi,:L,seq);
    w .= w .* -1.0;
    addDiag!(reshape(w,psi.m,psi.m),dotu(reshape(v,psi.m,psi.m),psi.ρR[i]));
    w .= w .+ v;

    return nothing;
end

@inline function ImTR(w::AbstractVector{T},
                      v::AbstractVector{T},
                      i::Int64,
                      psi::uMPS{T}, 
                      seq::Vector{Int64}) where {T<:Number}
    TR(w,v,psi,:R,seq);
    w .= w .* -1.0;
    addDiag!(reshape(w,psi.m,psi.m),dotu(psi.ρL[i],reshape(v,psi.m,psi.m)));
    w .= w .+ v;

    return nothing;
end
    
function setFixedPt(i::Int64,
                    psi::uMPS{T},
                    F::FixedPtTen{T},
                    W::Array{T,4})::Tuple{Float64,Float64} where {T<:Number}
    N::Int64 = F.N;
    M::Int64 = F.M;
    m::Int64 = F.m;
    seqL::Vector{Int64} = [mod1(i+j,N)   for j=1:N];
    seqR::Vector{Int64} = [mod1(i-j+1,N) for j=1:N];

    fill!(F.L[i],0.0);
    addDiag!(view(F.L[i],:,:,1),1.0);
    for a = 2:M
        TL_MPO(i,a,psi,F,W,seqL);
    end
    F.YL[i] .= copy(view(F.L[i],:,:,M));
    EL::Float64 = real(dotu(F.YL[i],psi.ρR[i]));
    addDiag!(F.YL[i],-EL);
    Lop = LinearMap{T}((w,v)->ImTL(w,v,i,psi,seqL),m*m;ishermitian=false);
    ~,info = bicgstabl!(F.HL[i],Lop,reshape(F.YL[i],m*m),2;reltol=1e-9,log=true);
    @assert (info.isconverged) "Linear solver not converged (L) after $(info.mvps) prods!";
    F.L[i][:,:,M] .= copy(reshape(F.HL[i],m,m));
 
    fill!(F.R[i],0.0);
    addDiag!(view(F.R[i],:,:,M),1.0);
    for a = M-1:-1:1
        TR_MPO(i,a,psi,F,W,seqR);
    end
    F.YR[i] .= copy(view(F.R[i],:,:,1));
    ER::Float64 = real(dotu(psi.ρL[i],F.YR[i]));
    addDiag!(F.YR[i],-ER);
    Rop = LinearMap{T}((w,v)->ImTR(w,v,i,psi,seqR),m*m;ishermitian=false);
    ~,info = bicgstabl!(F.HR[i],Rop,reshape(F.YR[i],m*m),2;reltol=1e-9,log=true);
    @assert (info.isconverged) "Linear solver not converged (R) after $(info.mvps) prods!";
    F.R[i][:,:,1] .= copy(reshape(F.HR[i],m,m));
    
    return (EL,ER);
end

struct HAChelper{T<:Number}
    T1::Array{T,4};
    T1p::Array{T,4};
    T2::Array{T,4};
    T2p::Array{T,4};
end

@inline function HAC(w::AbstractVector{T},
                     v::AbstractVector{T},
                     psi::uMPS{T},
                     H::Hamiltonian{T},
                     h::HAChelper{T},
                     j::Int64,
                     i::Int64) where {T<:Number}
    d = psi.d;
    m = psi.m;
    M = H.M;
    Lp = reshape(H.L[j],m*M,m);
    vp = reshape(v,m,d*m);
    BLAS.gemm!('N','N',T(1.0),Lp,vp,T(0.0),reshape(h.T1,m*M,d*m));
    transpose!(reshape(h.T1p,m,m*M*d),reshape(h.T1,m*M*d,m));
    Wp = reshape(H.W,M*d,d*M);
    BLAS.gemm!('N','N',T(1.0),reshape(h.T1p,m*m,M*d),Wp,T(0.0),reshape(h.T2,m*m,d*M));
    transpose!(reshape(h.T2p,m*d*M,m),reshape(h.T2,m,m*d*M));
    Rp = reshape(H.R[i],M*m,m);
    BLAS.gemm!('N','N',T(1.0),reshape(h.T2p,m*d,M*m),Rp,T(0.0),reshape(w,m*d,m));
    
    return nothing;
end

@inline function HC(w::AbstractVector{T},
                    v::AbstractVector{T},
                    psi::uMPS{T},
                    L::Array{T,3},
                    R::Array{T,3}) where {T<:Number}
    wp = reshape(w,psi.m,psi.m);
    @tensor wp[δ,γ] = L[δ,μ,α]*reshape(v,psi.m,psi.m)[α,β]*R[μ,β,γ];

    return nothing;
end

function HA2C(v::Array{Float64,4},L::Array{Float64,3},R::Array{Float64,3},W::Array{Float64,4})
    @tensor w[δ,π,τ,γ] := L[α,δ,μ]*(v[α,σ,ρ,β]*W[μ,σ,π,ω]*W[ω,ρ,τ,ν])*R[β,γ,ν];
    return w;
end

function convert(V::Val{T} , O::HermitianOp{T2})::HermitianOp{T} where {T<:Number,T2<:Number}
    return HermitianOp(O.name,[totype(V,ob) for ob in O.op]);
end

@inline function tl_mult(w::AbstractArray{T,2} , A::Array{T2,3} , B::Array{T2,3} , O::Matrix{T}) where {T<:Number,T2<:Number}
    @tensor w[δ,γ] = A[α,σ,δ]*(O[σ,ρ]*B[γ,ρ,α]);
    return nothing;
end

@inline function tl_mult(w::AbstractArray{T,2} , v::AbstractArray{T,2} , A::Array{T2,3} , B::Array{T2,3} , O::Matrix{T}) where {T<:Number,T2<:Number}
    @tensor w[δ,γ] = A[α,σ,δ]*v[α,β]*(O[σ,ρ]*B[γ,ρ,β]);
    return nothing;
end

@inline function measureObs(psi::uMPS{T} , seq::Vector{Int64} , obs::Observables{T}) where {T<:Number}
    for x = 1:length(obs.op)
        O = obs.op[x];
        for i = 1:length(O.op)
            j = seq[mod1(i,length(seq))];
            if(i == 1)
                @inbounds tl_mult(to(obs) , psi.A[:L][j] , psi.B[:L][j] , O.op[i]);
            else
                @inbounds tl_mult(to(obs) , from(obs) , psi.A[:L][j] , psi.B[:L][j] , O.op[i]);
            end
            pingpong(obs);
        end
        obs.ev[x] = real(dotu(from(obs),psi.ρR[seq[mod1(length(O.op),length(seq))]]));
    end

    return nothing;
end

function vumps(W::Array{T,4},
               Td::Dict{String,Vector{Array{T,3}}},
               measOps::Vector{HermitianOp{T2}},
               conv::Float64,
               logn::String) where {T<:Number,T2<:Number} 
    # extract simulation parameters
    N::Int64 = length(Td["AL"]);     # size of unit cell
    m::Int64 = size(Td["AL"][1])[1]; # MPS bond dim
    M::Int64 = size(W)[1];           # MPO bond dim
    d::Int64 = size(W)[2];           # local Hilbert sp dim
    nSteps::Int64 = 10000;           # upper bound on steps
    thresh::Float64 = 1e-12;         # threshold for float comps

    # initialize uMPS struct container
    C::Vector{Array{T,2}}  = [reshape(c,(m,m)) for c in Td["C"]];
    psi = uMPS{T}(
        #= N = =# N,
        #= m = =# m,
        #= d = =# d,
        #= A = =# (L=Td["AL"],R=Td["AR"]),
        #= B = =# (L=conj.([permutedims(a,(3,2,1)) for a in Td["AL"]]),
                   R=conj.([permutedims(a,(3,2,1)) for a in Td["AR"]])),
        #= C = =# C,
        #= AC= =# [reshape(reshape(Td["AL"][i],m*d,m)*C[i],m,d,m) for i=1:N],
        #= ρL= =# [transpose(c)*conj.(c) for c in C],
        #= ρR= =# [c*c' for c in C],
        #= vv= =# [Array{T,2}(undef,m,m) , Array{T,2}(undef,m,m)], 
        #= p = =# [0] 
        );

    # initialize FixedPtTen struct container
    F = FixedPtTen{T}(
        #= N  =# N,
        #= m  =# m,
        #= M  =# M,
        #= L  =# [Array{T,3}(undef,m,m,M) for i=1:N],
        #= R  =# [Array{T,3}(undef,m,m,M) for i=1:N],
        #= YL =# [Array{T,2}(undef,m,m) for i=1:N],
        #= YR =# [Array{T,2}(undef,m,m) for i=1:N],
        #= HL =# [zeros(T,m*m) for i=1:N],
        #= HR =# [zeros(T,m*m) for i=1:N],
        #= vv =# [Array{T,3}(undef,m,m,M) , Array{T,3}(undef,m,m,M)],
        #= p  =# [0] 
        );

    # initialize Hamiltonian struct container
    H = Hamiltonian{T}(
        #= M =# M,
        #= L =# [Array{T,3}(undef,m,M,m) for i=1:N],
        #= R =# [Array{T,3}(undef,M,m,m) for i=1:N],
        #= W =# W
        );

    # initialize HAChelper which provides intermediate storage for H ops
    h = HAChelper{T}(
        #= T1  =# Array{T,4}(undef,m,M,d,m),
        #= T1p =# Array{T,4}(undef,m,m,M,d),
        #= T2  =# Array{T,4}(undef,m,m,d,M),
        #= T2p =# Array{T,4}(undef,m,d,M,m)
        );

    # initialize diagnostic observables data container
    OT = T<:Complex ? T : T2;
    Ob = Observables{OT}(
        #= op =# [convert(Val(OT),O) for O in measOps],
        #= ev =# Vector{Float64}(undef,length(measOps)),
        #= vv =# [Array{OT,2}(undef,m,m) , Array{OT,2}(undef,m,m)],
        #= p  =# [0] 
        );

    # allocate memory for data
    ACt  = Matrix{T}(undef,m*d,m);       # used to calculate B param
    NL   = Matrix{T}(undef,m*d,m*(d-1)); # used to calculate B param
    UAL  = Matrix{T}(undef,m*d,m);       # polar decomp of AC_L
    UAR  = Matrix{T}(undef,m,d*m);       # polar decomp of AC_R
    UCL  = Matrix{T}(undef,m,m);         # polar decomp of C_L
    UCR  = Matrix{T}(undef,m,m);         # polar decomp of C_R
    enable_cache(maxsize=Int64(1e9));

    # initialize algorithm metrics
    e = zeros(Float64,N); # e convergence parameter
    B = zeros(Float64,N); # B convergence parameter
    E = zeros(Float64,N); # trial state energy
    S = zeros(Float64,N); # entanglement entropy
    tLS::Float64 = 0.0;   # linear solve time
    tEV::Float64 = 0.0;   # eigensolve time
    t::Float64   = 0.0;   # individual step time

    # write log file header
    logf = open("$(logn).log","w");
    print(logf,"# i t");
    for i = 1:N
        print(logf," e$i B$i E$i S$i");
        for O in Ob.op
            print(logf," $(O.name)$i");
        end
    end
    print(logf,"\n");

    # solve for leading eigenvectors of L,R transfer operators
    for i = 1:N
        seqL::Vector{Int64} = [mod1(i+j,N)   for j=1:N];
        seqR::Vector{Int64} = [mod1(i-j+1,N) for j=1:N];
       
        TmL = LinearMap{T}((w,v)->TR(w,v,psi,:L,seqR),m*m;ishermitian=false);
        V,U = eigs(TmL;nev=1,v0=reshape(psi.ρR[i],m*m));
        @assert (abs(V[1] - 1.0+0.0im) < thresh) "leading eigenvalue of TL not unity";
        psi.ρR[i] .= totype(Val(T),reshape(U[:,1],m,m));
        psi.ρR[i] /= tr(psi.ρR[i]);
     
        TmR = LinearMap{T}((w,v)->TL(w,v,psi,:R,seqL),m*m;ishermitian=false);
        V,U = eigs(TmR;nev=1,v0=reshape(psi.ρL[i],m*m));
        @assert (abs(V[1] - 1.0+0.0im) < thresh) "leading eigenvalue of TR not unity";
        psi.ρL[i] .= totype(Val(T),reshape(U[:,1],m,m));
        psi.ρL[i] /= tr(psi.ρL[i]);
    end 

    for n = 1:nSteps
        for i = 1:N
            j = mod1(i-1,N);
            
            # error measure of convergence
            eL::Float64 = norm(reshape(psi.AC[i],(m*d,m))-reshape(psi.A[:L][i],(m*d,m))*psi.C[i]); 
            eR::Float64 = norm(reshape(psi.AC[i],(m,d*m))-psi.C[j]*reshape(psi.A[:R][i],(m,d*m)));
            e[i] = max(eL,eR);
        end
           
        # convergence tolerance for eigensolvers
        tol = min(1e-8,maximum(e)*1e-2);

        t = @elapsed for i = 1:N
            j = mod1(i-1,N);

            # set MPO fixed point tensors based on updated AL, AR
            tLS += @elapsed begin
                (j != i) && setFixedPt(j,psi,F,W);
                E_vals =    setFixedPt(i,psi,F,W);
            end
            E[i] = (E_vals[1]+E_vals[2])/2.0;
         
            # another structure stores tensors in contraction-friendly way
            for x = 1:N
                H.L[x] .= permutedims(F.L[x],(2,3,1));
                H.R[x] .= permutedims(F.R[x],(3,1,2));
            end

            # geometric convergence metric
            HAC(reshape(ACt,m*d*m),reshape(psi.AC[i],m*d*m),psi,H,h,j,i);
            NL .= nullspace(transpose(reshape(psi.A[:L][i],(m*d,m))));
            B[i] = norm(transpose(NL)*ACt);

            # use fixed point tensors L, R and MPO to solve for new center tensors
            HACop = LinearMap{T}((w,v)->HAC(w,v,psi,H,h,j,i),m*d*m;issymmetric=true,ishermitian=true);
            tEV += @elapsed begin
                V,U,nconv,niter,nmult,resid = eigs(HACop;nev=1,which=:SR,v0=reshape(psi.AC[i],m*d*m),tol=tol);
            end
            psi.AC[i] .= totype(Val(T),reshape(U[:,1],m,d,m));

            if(j != i)
                HCLop = LinearMap{T}((w,v)->HC(w,v,psi,H.L[j],H.R[j]),m*m;issymmetric=true,ishermitian=true);
                tEV += @elapsed begin
                    V,U,nconv,niter,nmult,resid = eigs(HCLop;nev=1,which=:SR,v0=reshape(psi.C[j],m*m),tol=tol);
                end
                psi.C[j] .= totype(Val(T),reshape(U[:,1],m,m).*exp(-1.0im*angle(U[1,1])));
                psi.ρL[j] .= transpose(psi.C[j])*conj.(psi.C[j]);
                psi.ρR[j] .= psi.C[j]*psi.C[j]';
            end

            HCRop = LinearMap{T}((w,v)->HC(w,v,psi,H.L[i],H.R[i]),m*m;issymmetric=true,ishermitian=true);
            tEV += @elapsed begin
               V,U,nconv,niter,nmult,resid = eigs(HCRop;nev=1,which=:SR,v0=reshape(psi.C[i],m*m),tol=tol);
            end
            psi.C[i] .= totype(Val(T),reshape(U[:,1],m,m).*exp(-1.0im*angle(U[1,1])));
            psi.ρL[i] .= transpose(psi.C[i])*conj.(psi.C[i]);
            psi.ρR[i] .= psi.C[i]*psi.C[i]';

            # update AL, AR by polar decomposition of center tensors
            polarU!(UAL,reshape(psi.AC[i],(m*d,m)));
            polarU!(UAR,reshape(psi.AC[i],(m,d*m)));
            polarU!(UCL,psi.C[i]);
            polarU!(UCR,psi.C[j]);

            psi.A[:L][i] .= reshape(UAL*UCL',(m,d,m));
            psi.A[:R][i] .= reshape(UCR'*UAR,(m,d,m));

            psi.B[:L][i] .= conj.(permutedims(psi.A[:L][i],(3,2,1)));
            psi.B[:R][i] .= conj.(permutedims(psi.A[:R][i],(3,2,1)));
        end

        # print convergence metrics to log file
        @printf(logf,"%05d %.2f ",n,t);
        for i = 1:N
            seq::Vector{Int64} = [mod1(i+j-1,N) for j=1:N];
            S[i] = mapreduce(x->x*x*log(x*x),-,svdvals(psi.C[i]);init=0.0);
            @printf(logf,"%.1e %.14f %.14f %.10f ",e[i],B[i],E[i]/N,S[i]);
            measureObs(psi,seq,Ob);
            ob_str = [@sprintf("%12.9f",v) for v in Ob.ev];
            join(logf,ob_str," ");
            print(logf,i == N ? "\n" : " ");
        end 
        flush(logf);

        if(all(B .< conv))
            break;
        end
    end

    # compute leading eigenvalues of N-site transfer matrix
    nEigs::Int64 = 12;
    seq = [mod1(j,N) for j=1:N];
    TmL = LinearMap{T}((w,v)->TL(w,v,psi,:L,seq),m*m;ishermitian=false);
    λ,~ = eigs(TmL;nev=nEigs,v0=reshape(Matrix{T}(I,m,m),m*m));
    @assert (abs(λ[1] - 1.0+0.0im) < thresh) "leading eigenvalue of TL not unity";
    close(logf);

    A2Cp = [Array{T,4}(undef,m,d,d,m) for i=1:N];
    for i = 1:N
        j = mod1(i+1,N);
        k = mod1(i+2,N);
        @tensor A2C[α,σ,ρ,γ] := psi.A[:L][i][α,σ,β]*psi.C[i][β,δ]*psi.A[:R][j][δ,ρ,γ];
        A2Cp[i] .= HA2C(A2C,F.L[i],F.R[k],W);
    end

    Td["C"] .= reshape.(psi.C,m,1,m);
    clear_cache();

    return E,λ,S,tLS,tEV,A2Cp;
end
